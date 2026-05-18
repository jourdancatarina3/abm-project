"""Experimental scenarios and the runner that produces every CSV consumed by
the analysis layer.

We define six scenarios.  The first three are direct controls for the three
parts of the alternative hypothesis ("virality emerges from the joint
interaction of user behaviour and the platform algorithm"):

    S0  No-Algorithm                -- pure word-of-mouth (true negative control)
    S1  Baseline                    -- moderate user engagement, NORMAL algorithm
    S2  Algorithm-Boost-Only        -- aggressive BOOSTING, low user engagement
    S3  High-Engagement-Only        -- high peer-share, algorithm stays PASSIVE
    S4  Combined Boost + Engagement -- both high (full interaction)
    S5  Influencer Seeding          -- like S1 but trend originates from hubs

By contrasting S2 vs S3 vs S4 we can isolate the *interaction* effect
required to either accept or reject the null.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import pandas as pd
from tqdm import tqdm

from .agents import AlgorithmState
from .model import ModelConfig, ViralityModel


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------


BASE = ModelConfig(
    n_users=1000,
    n_ticks=480,
    network_kind="ba",
    ba_m=3,
    intrinsic_appeal=0.55,
    novelty_half_life=96.0,
    peer_share_prob=0.30,
    algorithm_initial_state=AlgorithmState.NORMAL,
    boost_threshold=0.015,
    boost_inject_prob=0.05,
    normal_inject_prob=0.002,
    boost_multiplier=1.0,
    boost_duration_max=72,
    seed_strategy="random",
    n_seeds=3,
)


def scenario_configs() -> dict[str, ModelConfig]:
    return {
        "S0_NoAlgorithm": replace(
            BASE,
            algorithm_initial_state=AlgorithmState.PASSIVE,
            normal_inject_prob=0.0,
            boost_inject_prob=0.0,
            peer_share_prob=0.30,
        ),
        "S1_Baseline": replace(BASE),
        "S2_AlgoBoostOnly": replace(
            BASE,
            peer_share_prob=0.10,       # weak peer sharing
            boost_threshold=0.001,      # easy to trip
            boost_inject_prob=0.12,     # aggressive boost
            boost_multiplier=1.5,
        ),
        "S3_HighEngagementOnly": replace(
            BASE,
            algorithm_initial_state=AlgorithmState.PASSIVE,
            normal_inject_prob=0.0,
            boost_inject_prob=0.0,
            peer_share_prob=0.60,       # strong peer sharing
        ),
        "S4_BoostPlusEngagement": replace(
            BASE,
            peer_share_prob=0.60,
            boost_threshold=0.005,
            boost_inject_prob=0.12,
            boost_multiplier=1.5,
        ),
        "S5_InfluencerSeed": replace(
            BASE,
            seed_strategy="hub",
        ),
    }


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def _run_one(args):
    scenario_name, cfg = args
    m = ViralityModel(cfg)
    result = m.run()
    tick = result.tick_log.assign(scenario=scenario_name, seed=cfg.seed)
    summary = dict(result.summary)
    summary["scenario"] = scenario_name
    summary["seed"] = cfg.seed
    fin = result.final_states.assign(scenario=scenario_name, seed=cfg.seed)
    return tick, summary, fin


def run_sweep(
    n_replicates: int = 30,
    base_seed: int = 1000,
    n_jobs: int = -1,
    out_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Run every scenario `n_replicates` times in parallel.

    Returns a dict of three pandas DataFrames:
        tick_log  -- one row per (scenario, seed, tick)
        summary   -- one row per (scenario, seed)
        final     -- one row per (scenario, seed, user)
    """
    scenarios = scenario_configs()
    jobs = []
    for name, cfg in scenarios.items():
        for r in range(n_replicates):
            jobs.append((name, replace(cfg, seed=base_seed + r)))

    n_jobs = mp.cpu_count() if n_jobs < 0 else n_jobs
    tick_frames = []
    summaries = []
    final_frames = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_run_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sims"):
            tick, summary, fin = fut.result()
            tick_frames.append(tick)
            summaries.append(summary)
            final_frames.append(fin)

    tick_log = pd.concat(tick_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    final_df = pd.concat(final_frames, ignore_index=True)

    os.makedirs(f"{out_dir}/raw", exist_ok=True)
    os.makedirs(f"{out_dir}/processed", exist_ok=True)

    tick_log.to_csv(f"{out_dir}/raw/tick_log.csv.gz", index=False, compression="gzip")
    summary_df.to_csv(f"{out_dir}/processed/summary.csv", index=False)
    final_df.to_csv(f"{out_dir}/raw/final_states.csv.gz", index=False, compression="gzip")

    return dict(tick_log=tick_log, summary=summary_df, final=final_df)


# ---------------------------------------------------------------------------
# Network sensitivity sweep (smaller, for the discussion section)
# ---------------------------------------------------------------------------


def run_network_sensitivity(
    n_replicates: int = 15,
    base_seed: int = 5000,
    n_jobs: int = -1,
    out_dir: str = "data",
) -> pd.DataFrame:
    """Run the BASE scenario on three network topologies."""
    kinds = ["ba", "ws", "er"]
    jobs = []
    for kind in kinds:
        for r in range(n_replicates):
            cfg = replace(BASE, network_kind=kind, seed=base_seed + r)
            jobs.append((f"NET_{kind.upper()}", cfg))

    n_jobs = mp.cpu_count() if n_jobs < 0 else n_jobs
    summaries = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_run_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="nets"):
            _, summary, _ = fut.result()
            summaries.append(summary)

    df = pd.DataFrame(summaries)
    df.to_csv(f"{out_dir}/processed/network_sensitivity.csv", index=False)
    return df
