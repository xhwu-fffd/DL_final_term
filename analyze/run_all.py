# -*- coding: utf-8 -*-
"""Run the whole Task-2 analysis suite end to end.

  python run_all.py                 # full run (re-evaluates models, then all figs)
  python run_all.py --skip-evals    # reuse cached evals, just rebuild figures

Outputs land in analyze/figures/*.png and analyze/results/*.json|csv|md.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run(mod, argv=None):
    print(f"\n{'='*70}\n# {mod}\n{'='*70}")
    sys.argv = [mod] + (argv or [])
    runpy.run_path(str(HERE / f"{mod}.py"), run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-evals", action="store_true",
                    help="reuse analyze/results/arrays/*.npz from a previous run")
    ap.add_argument("--n-windows", type=int, default=3000)
    args = ap.parse_args()

    if not args.skip_evals:
        run("run_evals", ["--n-windows", str(args.n_windows)])
    run("fig_generalization")
    run("fig_chunking")
    run("fig_error_breakdown")
    run("fig_temporal")
    run("fig_training")
    run("fig_data_shift")
    print("\nAll analysis artifacts written to analyze/figures and analyze/results.")


if __name__ == "__main__":
    main()
