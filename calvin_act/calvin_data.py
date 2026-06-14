# -*- coding: utf-8 -*-
"""LeRobot-format CALVIN dataset access for the ACT cross-environment study.

The data for HW3 Task 2 is a set of **LeRobot v2.x datasets**, *one directory per
CALVIN environment*, all under a single root (default
``task2/data/calvin-lerobot``)::

    data/calvin-lerobot/
      splitA/                                     <- environment A
        meta/info.json                            feature schema, fps, path template
        meta/episodes.jsonl                       one line/episode: index, length, scene
        meta/tasks.jsonl
        data/chunk-000/episode_000000.parquet     ONE parquet PER EPISODE
        data/chunk-000/episode_000001.parquet
        ...
      splitB/  splitC/  splitD/                   <- same layout (added later)

Each parquet row is one timestep with columns ``image`` (HxWx3, PNG-encoded),
``wrist_image``, ``state`` (float32[15]) and ``actions`` (float32[7] = CALVIN
``rel_actions``). One parquet == one episode == one contiguous segment, so action
chunks must never cross a parquet boundary (handled by :meth:`EnvData.chunk`).

Environment selection is by **folder**: env ``"A"`` -> ``splitA`` (Task 1 trains
on A, Task 2 on A+B+C, Task 3 zero-shot tests on the unseen D).

This module exposes:

* :func:`default_data_root` / :func:`find_data_root` / :func:`env_dir` - locate the
  per-environment datasets (portable, machine-independent paths);
* :class:`EnvData` - preloads the small low-dim arrays (state + action) for the
  selected environments into RAM (with an on-disk cache) and reads the (large)
  images on demand, exposing per-row action-chunk windows with padding masks.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Canonical LeRobot column keys in this dataset ------------------------------
IMAGE_KEYS = ("image", "wrist_image")     # static (200x200) + wrist (84x84)
STATE_KEY = "state"
ACTION_KEY = "actions"                     # CALVIN rel_actions, 7-D
ENVS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Locating the data (portable / relative paths)
# ---------------------------------------------------------------------------
def default_data_root() -> Path:
    """Root holding the per-environment LeRobot datasets (``split{A,B,C,D}``).

    Resolved **relative to this file** (``<repo>/task2/data/calvin-lerobot``) so the
    code is portable across machines/GPUs without editing paths. Override with the
    ``CALVIN_DATA_ROOT`` environment variable or the CLI ``--data-root`` flag.
    """
    env = os.environ.get("CALVIN_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "calvin-lerobot"


def find_data_root(data_root: str | Path | None = None) -> Path:
    root = Path(data_root).expanduser() if data_root is not None else default_data_root()
    if not root.exists():
        raise FileNotFoundError(
            f"data root not found: '{root}'. Put the per-environment LeRobot "
            f"datasets there as split{{A,B,C,D}}/ (each with meta/info.json), set "
            f"$CALVIN_DATA_ROOT, or pass --data-root. See README (数据准备).")
    return root


def _is_lerobot_dataset(d: Path) -> bool:
    return (d / "meta" / "info.json").exists()


def env_dir(data_root: str | Path | None, env: str) -> Path:
    """Resolve an environment letter (``"A"``..``"D"``) to its dataset directory.

    Tries common folder names (``splitA``, ``A``, ...) then falls back to scanning
    for a dataset whose ``meta/info.json`` records ``"scene": "<env>"``.
    """
    root = find_data_root(data_root)
    e = env.strip().upper()
    candidates = [f"split{e}", f"split{e.lower()}", e, e.lower(),
                  f"calvin_scene_{e}", f"scene_{e}", f"env{e}", f"task_{e}"]
    for name in candidates:
        d = root / name
        if _is_lerobot_dataset(d):
            return d
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if _is_lerobot_dataset(d):
            try:
                info = load_info(d)
                if str(info.get("scene", "")).strip().upper() == e:
                    return d
            except Exception:
                continue
    present = [p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []
    raise FileNotFoundError(
        f"environment {env!r} not found under {root}. Expected a LeRobot dataset "
        f"directory like 'split{e}/' containing meta/info.json. Present: {present}")


def available_envs(data_root: str | Path | None = None) -> list[str]:
    """Letters of the environments currently present under ``data_root``."""
    root = find_data_root(data_root)
    found: list[str] = []
    for e in ENVS:
        try:
            env_dir(root, e)
            found.append(e)
        except FileNotFoundError:
            pass
    return found


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def load_info(ds_dir: str | Path) -> dict:
    return json.loads((Path(ds_dir) / "meta" / "info.json").read_text(encoding="utf-8"))


def load_episodes_meta(ds_dir: str | Path) -> list[dict]:
    """Parse ``meta/episodes.jsonl`` -> list of per-episode dicts."""
    out: list[dict] = []
    p = Path(ds_dir) / "meta" / "episodes.jsonl"
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def episode_parquet_path(ds_dir: str | Path, info: dict, episode_index: int) -> Path:
    """Resolve an episode's parquet path from the ``data_path`` template."""
    chunks_size = int(info.get("chunks_size", 1000)) or 1000
    template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel = template.format(episode_chunk=episode_index // chunks_size,
                          episode_index=episode_index)
    return Path(ds_dir) / rel


def feature_dim(info: dict, key: str, default: int | None = None) -> int | None:
    feats = info.get("features", {})
    if key in feats and feats[key].get("shape"):
        return int(np.prod(feats[key]["shape"]))
    return default


# ---------------------------------------------------------------------------
# Parquet IO helpers
# ---------------------------------------------------------------------------
def _read_columns(path: str | Path, cols):
    """Read selected columns of a parquet file as a pandas DataFrame."""
    import pandas as pd
    return pd.read_parquet(path, columns=list(cols))


def _stack_vectors(series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in series])


def _decode_image(cell) -> np.ndarray:
    """A LeRobot image cell -> HxWx3 uint8 array.

    Handles the HuggingFace image dict ``{"bytes":..., "path":...}``, raw PNG/JPEG
    bytes, a path string, or an already-decoded array.
    """
    from PIL import Image
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            cell = cell["bytes"]
        elif cell.get("path"):
            cell = cell["path"]
        else:
            raise ValueError("image cell dict has neither 'bytes' nor 'path'")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return np.asarray(Image.open(io.BytesIO(bytes(cell))).convert("RGB"))
    if isinstance(cell, str):
        return np.asarray(Image.open(cell).convert("RGB"))
    arr = np.asarray(cell)
    if arr.ndim == 2:                                  # HxW grayscale -> HxWx3
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr


# ---------------------------------------------------------------------------
# On-disk cache for the preloaded low-dim arrays
# ---------------------------------------------------------------------------
def _cache_signature(env_dirs: list[Path], **params) -> str:
    parts = []
    for d in env_dirs:
        ep = Path(d) / "meta" / "episodes.jsonl"
        info = Path(d) / "meta" / "info.json"
        st = ep.stat() if ep.exists() else None
        parts.append(f"{d.name}:{st.st_size if st else 0}:{int(st.st_mtime) if st else 0}:"
                     f"{info.stat().st_mtime_ns if info.exists() else 0}")
    parts.append(json.dumps(params, sort_keys=True))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Preloaded environment data (low-dim in RAM, images lazy)
# ---------------------------------------------------------------------------
@dataclass
class EnvData:
    """Low-dim arrays (preloaded) + lazy image access for a set of episodes.

    Rows are a flat concatenation of all selected episodes; ``segments`` holds the
    half-open ``[row_start, row_end)`` range of each episode so action chunks stay
    within a single episode. ``files[i]`` is the parquet backing ``segments[i]``.
    """
    states: np.ndarray                       # [T, S]
    actions: np.ndarray                      # [T, A]
    segments: list[tuple[int, int]]          # half-open row ranges, one per episode
    files: list[Path]                        # parquet path per segment
    image_keys: tuple[str, ...] = ("image",)
    state_key: str = STATE_KEY
    action_key: str = ACTION_KEY
    envs: tuple[str, ...] = ()
    _seg_starts: np.ndarray = field(default=None, repr=False, compare=False)
    _img_cache: "OrderedDict | None" = field(default=None, repr=False, compare=False)
    cache_capacity: int = field(default=128, repr=False, compare=False)

    def __post_init__(self):
        self._seg_starts = np.asarray([s for s, _ in self.segments], dtype=np.int64)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, data_root: str | Path | None, envs,
              image_keys: tuple[str, ...] = ("image",),
              state_key: str = STATE_KEY, action_key: str = ACTION_KEY,
              max_episodes: int | None = None, max_steps: int | None = None,
              subset: str = "all", val_fraction: float = 0.0, split_seed: int = 0,
              use_disk_cache: bool = True, verbose: bool = True) -> "EnvData":
        """Load the selected environments.

        ``subset`` is one of ``"all" | "train" | "val"``; when ``val_fraction>0`` a
        deterministic per-environment episode hold-out is carved out so a clean
        validation Action-L1 curve can be logged during training (assignment §4.1).
        """
        root = find_data_root(data_root)
        envs = [e.strip().upper() for e in envs]
        env_dirs = [env_dir(root, e) for e in envs]

        cached = None
        sig_params = dict(envs=envs, image_keys=list(image_keys), state_key=state_key,
                          action_key=action_key, max_episodes=max_episodes,
                          max_steps=max_steps, subset=subset,
                          val_fraction=val_fraction, split_seed=split_seed)
        cache_file = None
        if use_disk_cache:
            sig = _cache_signature(env_dirs, **sig_params)
            cache_file = root / ".cache" / "calvin_act" / f"lowdim_{sig}.npz"
            cached = _load_lowdim_cache(cache_file)

        if cached is not None:
            states, actions, segments, files = cached
        else:
            states, actions, segments, files = _preload_lowdim(
                env_dirs, envs, state_key, action_key, max_episodes, max_steps,
                subset, val_fraction, split_seed, verbose)
            if cache_file is not None:
                _save_lowdim_cache(cache_file, states, actions, segments, files)

        # validate the requested image keys exist
        info0 = load_info(env_dirs[0])
        feats = info0.get("features", {})
        if feats:
            for k in image_keys:
                if k not in feats:
                    raise ValueError(
                        f"image key {k!r} not in dataset features {list(feats)}. "
                        f"Use one of the image features, e.g. 'image' / 'wrist_image'.")

        ed = cls(states=states, actions=actions, segments=segments, files=files,
                 image_keys=tuple(image_keys), state_key=state_key,
                 action_key=action_key, envs=tuple(envs))
        if verbose:
            print(f"[calvin] envs={envs} subset={subset}: {len(ed)} steps in "
                  f"{len(segments)} episodes, state_dim={ed.state_dim}, "
                  f"action_dim={ed.action_dim}, cameras={list(image_keys)}"
                  + ("  [cached]" if cached is not None else ""))
        return ed

    # -- properties --------------------------------------------------------
    def __len__(self) -> int:
        return self.states.shape[0]

    @property
    def state_dim(self) -> int:
        return self.states.shape[1]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    def _seg_index(self, row: int) -> int:
        if row < 0 or row >= len(self):
            raise IndexError(row)
        return int(np.searchsorted(self._seg_starts, row, side="right") - 1)

    def segment_of(self, row: int) -> tuple[int, int]:
        return self.segments[self._seg_index(row)]

    # -- action chunk window ----------------------------------------------
    def chunk(self, row: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Action chunk ``a[row : row+horizon]`` clipped to the row's episode.

        Returns ``(chunk [horizon, A], pad_mask [horizon] bool)`` where True marks
        padded (invalid) steps; the last valid action is repeated as padding.
        """
        _, seg_end = self.segment_of(row)
        end = min(row + horizon, seg_end)
        valid = end - row
        chunk = np.empty((horizon, self.action_dim), dtype=np.float32)
        chunk[:valid] = self.actions[row:end]
        if valid < horizon:
            chunk[valid:] = self.actions[end - 1]      # repeat last
        pad = np.zeros(horizon, dtype=bool)
        pad[valid:] = True
        return chunk, pad

    # -- images (lazy, LRU-cached encoded columns) ------------------------
    def _cache(self) -> "OrderedDict":
        if self._img_cache is None:
            self._img_cache = OrderedDict()
        return self._img_cache

    def _segment_image_columns(self, seg_idx: int) -> dict:
        cache = self._cache()
        if seg_idx in cache:
            cache.move_to_end(seg_idx)
            return cache[seg_idx]
        df = _read_columns(self.files[seg_idx],
                           [k for k in self.image_keys])
        cols = {k: df[k].to_list() for k in self.image_keys if k in df.columns}
        cache[seg_idx] = cols
        while len(cache) > self.cache_capacity:
            cache.popitem(last=False)
        return cols

    def load_images(self, row: int) -> dict[str, np.ndarray]:
        seg_idx = self._seg_index(row)
        local = row - self.segments[seg_idx][0]
        cols = self._segment_image_columns(seg_idx)
        out: dict[str, np.ndarray] = {}
        for k in self.image_keys:
            col = cols.get(k)
            if col is not None:
                out[k] = _decode_image(col[local])
        return out

    # -- pickling (drop the per-process image cache) ----------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_img_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# low-dim preloading + cache (module-level so they're picklable / testable)
# ---------------------------------------------------------------------------
def _episode_order(n_eps: int, subset: str, val_fraction: float, split_seed: int,
                   env: str) -> list[int]:
    order = list(range(n_eps))
    if val_fraction and 0.0 < val_fraction < 1.0 and subset in ("train", "val"):
        rng = np.random.default_rng(split_seed + (ord(env) if env else 0))
        perm = rng.permutation(n_eps)
        n_val = max(1, int(round(n_eps * val_fraction)))
        val_idx = set(int(i) for i in perm[:n_val])
        if subset == "val":
            order = [i for i in order if i in val_idx]
        else:                                          # "train"
            order = [i for i in order if i not in val_idx]
    return order


def _preload_lowdim(env_dirs, envs, state_key, action_key, max_episodes, max_steps,
                    subset, val_fraction, split_seed, verbose):
    states_list, actions_list, segments, files = [], [], [], []
    row = 0
    for d, e in zip(env_dirs, envs):
        info = load_info(d)
        eps = load_episodes_meta(d)
        order = _episode_order(len(eps), subset, val_fraction, split_seed, e)
        if max_episodes is not None:
            order = order[:max_episodes]
        if verbose:
            print(f"[calvin] reading env {e}: {len(order)} episodes from {d.name} ...")
        for oi in order:
            ep = eps[oi]
            ep_index = int(ep.get("episode_index", oi))
            path = episode_parquet_path(d, info, ep_index)
            if not path.exists():
                raise FileNotFoundError(f"episode parquet missing: {path}")
            df = _read_columns(path, [state_key, action_key])
            st = _stack_vectors(df[state_key].to_list())
            ac = _stack_vectors(df[action_key].to_list())
            n = len(st)
            if max_steps is not None and row + n > max_steps:
                n = max_steps - row
                st, ac = st[:n], ac[:n]
            if n <= 0:
                break
            states_list.append(st)
            actions_list.append(ac)
            files.append(path)
            segments.append((row, row + n))
            row += n
            if max_steps is not None and row >= max_steps:
                break
        if max_steps is not None and row >= max_steps:
            break

    if not segments:
        raise RuntimeError(
            f"no usable timesteps for envs={envs} (subset={subset}) under the given "
            f"data root. Check the split folders contain episode parquet files.")
    states = np.concatenate(states_list).astype(np.float32)
    actions = np.concatenate(actions_list).astype(np.float32)
    return states, actions, segments, files


def _save_lowdim_cache(cache_file: Path, states, actions, segments, files) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        seg = np.asarray(segments, dtype=np.int64)
        np.savez(cache_file, states=states, actions=actions, segments=seg,
                 files=np.asarray([str(p) for p in files]))
    except Exception as e:                              # caching is best-effort
        print(f"[calvin] (warning) could not write cache {cache_file.name}: {e}")


def _load_lowdim_cache(cache_file: Path):
    if not cache_file.exists():
        return None
    try:
        z = np.load(cache_file, allow_pickle=False)
        states = z["states"].astype(np.float32)
        actions = z["actions"].astype(np.float32)
        segments = [(int(s), int(t)) for s, t in z["segments"]]
        files = [Path(p) for p in z["files"].tolist()]
        return states, actions, segments, files
    except Exception:
        return None
