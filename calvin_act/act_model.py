# -*- coding: utf-8 -*-
"""ACT - Action Chunking with Transformers (a faithful, self-contained port).

This mirrors the policy used in LeRobot / the ALOHA paper: a CVAE whose decoder is
a DETR-style transformer that, from a single observation (multi-camera images +
proprioceptive state), predicts a **chunk** of the next ``H`` actions at once.

    images ─► CNN backbone ─► feature tokens ┐
    state  ─► linear ─► 1 token              ├─► Transformer encoder ─► memory
    z (CVAE latent) ─► linear ─► 1 token     ┘                              │
    H learned query tokens ─► Transformer decoder ◄───────────────────────┘
                                   │
                                   └─► linear ─► action chunk  [B, H, action_dim]

Training maximises an action-reconstruction (L1) likelihood plus a KL term on the
latent; at inference the latent is set to its prior mean (zeros). Action chunking
is the mechanism the assignment asks us to analyse under visual distribution
shift - it is the decoder's fixed set of ``H`` query tokens.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .act_config import ACTConfig


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------
def sinusoid_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """2-D sinusoidal position embedding, shape [h*w, dim].

    Half the channels encode the row (y) coordinate, half the column (x); each
    half is an interleaved sin/cos of geometrically-spaced frequencies.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2-D sincos"
    d = dim // 4                                       # freqs per axis per sin/cos
    omega = torch.exp(torch.arange(d).float() * (-math.log(10000.0) / max(d, 1)))
    yy = torch.arange(h).float()[:, None] * omega[None, :]     # [h, d]
    xx = torch.arange(w).float()[:, None] * omega[None, :]     # [w, d]
    pe = torch.zeros(h, w, dim)
    pe[..., 0:d] = torch.sin(yy)[:, None, :].expand(h, w, d)
    pe[..., d:2 * d] = torch.cos(yy)[:, None, :].expand(h, w, d)
    pe[..., 2 * d:3 * d] = torch.sin(xx)[None, :, :].expand(h, w, d)
    pe[..., 3 * d:4 * d] = torch.cos(xx)[None, :, :].expand(h, w, d)
    return pe.reshape(h * w, dim)


# ---------------------------------------------------------------------------
# Vision backbone
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """A lightweight conv stack (no pretrained weights) -> /16 feature map."""

    def __init__(self, out_channels: int = 256):
        super().__init__()
        c = [3, 32, 64, 128, out_channels]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                       nn.GroupNorm(8, c[i + 1]) if c[i + 1] >= 8 else nn.Identity(),
                       nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):                              # [B,3,H,W] -> [B,C,H/16,W/16]
        return self.net(x)


def build_backbone(cfg: ACTConfig):
    if cfg.vision_backbone == "small":
        return SmallCNN(out_channels=256), 256
    if cfg.vision_backbone == "resnet18":
        import torchvision
        weights = torchvision.models.ResNet18_Weights.DEFAULT if cfg.pretrained_backbone else None
        net = torchvision.models.resnet18(weights=weights)
        backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                 net.layer1, net.layer2, net.layer3, net.layer4)
        return backbone, 512
    raise ValueError(f"unknown vision_backbone: {cfg.vision_backbone}")


# ---------------------------------------------------------------------------
# ACT policy
# ---------------------------------------------------------------------------
class ACTPolicy(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        # vision
        self.backbone, feat_ch = build_backbone(cfg)
        self.input_proj = nn.Conv2d(feat_ch, D, kernel_size=1)

        # tokens for state and latent
        self.state_proj = nn.Linear(cfg.state_dim, D)
        self.latent_proj = nn.Linear(cfg.latent_dim, D)

        # work out how many image tokens per camera (dry run)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
            fmap = self.input_proj(self.backbone(dummy))
            _, _, fh, fw = fmap.shape
        self.feat_hw = (fh, fw)
        tokens_per_cam = fh * fw
        self.num_mem_tokens = 2 + cfg.num_cameras * tokens_per_cam   # z + state + imgs

        # image position embedding (2-D sincos), registered as buffer
        self.register_buffer("img_pos", sinusoid_2d(fh, fw, D), persistent=False)
        # learned embeddings for the z and state tokens
        self.extra_pos = nn.Parameter(torch.zeros(2, D))
        nn.init.normal_(self.extra_pos, std=0.02)

        # transformer encoder (obs -> memory)
        enc_layer = nn.TransformerEncoderLayer(
            D, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.n_enc_layers)

        # transformer decoder (H queries -> action chunk)
        dec_layer = nn.TransformerDecoderLayer(
            D, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.n_dec_layers)
        self.query_embed = nn.Parameter(torch.zeros(cfg.chunk_size, D))
        nn.init.normal_(self.query_embed, std=0.02)

        self.action_head = nn.Linear(D, cfg.action_dim)

        # CVAE encoder (action chunk + state -> latent)
        if cfg.use_vae:
            self.cls_token = nn.Parameter(torch.zeros(1, D))
            nn.init.normal_(self.cls_token, std=0.02)
            self.action_embed = nn.Linear(cfg.action_dim, D)
            self.vae_state_proj = nn.Linear(cfg.state_dim, D)
            self.vae_pos = nn.Parameter(torch.zeros(2 + cfg.chunk_size, D))
            nn.init.normal_(self.vae_pos, std=0.02)
            vae_layer = nn.TransformerEncoderLayer(
                D, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, batch_first=True)
            self.vae_encoder = nn.TransformerEncoder(vae_layer, 3)
            self.latent_head = nn.Linear(D, 2 * cfg.latent_dim)

    # -- pieces ------------------------------------------------------------
    def _image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """[B, num_cam, 3, H, W] -> [B, num_cam*h*w, D] with position embedding."""
        B, K = images.shape[:2]
        x = images.flatten(0, 1)                       # [B*K, 3, H, W]
        f = self.input_proj(self.backbone(x))          # [B*K, D, h, w]
        f = f.flatten(2).transpose(1, 2)               # [B*K, h*w, D]
        f = f + self.img_pos[None]                     # add 2-D pos
        return f.reshape(B, K * f.shape[1], -1)

    def _encode_latent(self, state, action):
        """CVAE encoder -> (z, mu, logvar). Used only in training."""
        B = state.shape[0]
        cls = self.cls_token.expand(B, 1, -1)
        s = self.vae_state_proj(state).unsqueeze(1)
        a = self.action_embed(action)                  # [B, H, D]
        seq = torch.cat([cls, s, a], dim=1) + self.vae_pos[None]
        h = self.vae_encoder(seq)[:, 0]                # CLS
        mu, logvar = self.latent_head(h).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def _decode(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        q = self.query_embed[None].expand(B, -1, -1)
        h = self.decoder(q, memory)                    # [B, H, D]
        return self.action_head(h)                     # [B, H, action_dim]

    def _memory(self, images, state, z):
        img_tok = self._image_tokens(images)           # [B, N, D]
        ztok = self.latent_proj(z).unsqueeze(1) + self.extra_pos[0]
        stok = self.state_proj(state).unsqueeze(1) + self.extra_pos[1]
        tokens = torch.cat([ztok, stok, img_tok], dim=1)
        return self.encoder(tokens)

    # -- forward (training) ------------------------------------------------
    def forward(self, batch: dict) -> dict:
        images, state = batch["images"], batch["state"]
        action, pad = batch["action"], batch["pad_mask"]
        B = state.shape[0]

        if self.cfg.use_vae:
            z, mu, logvar = self._encode_latent(state, action)
        else:
            z = torch.zeros(B, self.cfg.latent_dim, device=state.device)
            mu = logvar = None

        memory = self._memory(images, state, z)
        pred = self._decode(memory)

        valid = (~pad).unsqueeze(-1).float()           # [B, H, 1]
        l1 = (F.l1_loss(pred, action, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
        loss = l1
        kl = torch.zeros((), device=state.device)
        if self.cfg.use_vae:
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
            loss = l1 + self.cfg.kl_weight * kl
        return {"loss": loss, "l1": l1.detach(), "kl": kl.detach()}

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict_chunk(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Predict one normalised action chunk [B, H, action_dim] (latent = prior mean)."""
        self.eval()
        B = state.shape[0]
        z = torch.zeros(B, self.cfg.latent_dim, device=state.device)
        return self._decode(self._memory(images, state, z))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
