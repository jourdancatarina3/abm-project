"""Run the full experimental sweep.

Usage
-----
    python scripts/run_experiments.py [--reps 30]

Outputs
-------
    data/raw/tick_log.parquet           per-tick metrics for every run
    data/raw/final_states.parquet       per-user terminal state for every run
    data/processed/summary.csv          per-run summary metrics
    data/processed/network_sensitivity.csv
"""

import argparse
import sys
from pathlib import Path

# allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import run_network_sensitivity, run_sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=30,
                   help="replicates per scenario (default: 30)")
    p.add_argument("--net-reps", type=int, default=15,
                   help="replicates per network topology (default: 15)")
    p.add_argument("--jobs", type=int, default=-1,
                   help="parallel workers (-1 = all cores)")
    args = p.parse_args()

    print(f"[*] Main sweep: {args.reps} replicates x 6 scenarios")
    run_sweep(n_replicates=args.reps, n_jobs=args.jobs)

    print(f"[*] Network sensitivity: {args.net_reps} replicates x 3 topologies")
    run_network_sensitivity(n_replicates=args.net_reps, n_jobs=args.jobs)

    print("[+] Done. See data/processed/summary.csv")


if __name__ == "__main__":
    main()
