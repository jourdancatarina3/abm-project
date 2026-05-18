"""Publication-quality figures.

Every figure is saved twice -- once into figures/ for inspection and once
into paper/figs/ for direct \\includegraphics use.

Figures produced
----------------
    fig01_lifecycle_curves           per-tick engagement, 95% CI band, 6 scenarios
    fig02_outcome_boxplots           4 boxplots (reach, peak, ttv, lifetime)
    fig03_2x2_interaction            2x2 interaction plot (peak_engaged)
    fig04_2x2_interaction_reach      2x2 interaction plot (viral_reach)
    fig05_cohen_heatmap              Cohen's d matrix on peak_engaged
    fig06_network_sensitivity        viral_reach across BA / WS / ER
    fig07_state_proportions          UNAWARE/AWARE/ENGAGED/FATIGUED area chart
    fig08_exposure_distribution      per-user exposure histograms (log-y)
    fig09_correlation_heatmap        Pearson r between metrics (per scenario)
    fig10_algo_state_timeline        algorithm-state ribbon per scenario
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .agents import AlgorithmState

# ---------------------------------------------------------------------------
# Style + paths
# ---------------------------------------------------------------------------

SCENARIO_ORDER = [
    "S0_NoAlgorithm",
    "S1_Baseline",
    "S2_AlgoBoostOnly",
    "S3_HighEngagementOnly",
    "S4_BoostPlusEngagement",
    "S5_InfluencerSeed",
]
SCENARIO_LABELS = {
    "S0_NoAlgorithm":          "S0  No-Algo",
    "S1_Baseline":             "S1  Baseline",
    "S2_AlgoBoostOnly":        "S2  Algo-Only",
    "S3_HighEngagementOnly":   "S3  Engage-Only",
    "S4_BoostPlusEngagement":  "S4  Combined",
    "S5_InfluencerSeed":       "S5  Influencer",
}
PALETTE = {
    "S0_NoAlgorithm":          "#9b9b9b",
    "S1_Baseline":             "#4c72b0",
    "S2_AlgoBoostOnly":        "#dd8452",
    "S3_HighEngagementOnly":   "#55a868",
    "S4_BoostPlusEngagement":  "#c44e52",
    "S5_InfluencerSeed":       "#8172b3",
}


def _setup_style():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def _save(fig, name: str):
    for d in ("figures", "paper/figs"):
        os.makedirs(d, exist_ok=True)
        fig.savefig(f"{d}/{name}.png", bbox_inches="tight")
        fig.savefig(f"{d}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- per-tick engagement curves
# ---------------------------------------------------------------------------


def fig_lifecycle_curves(tick_log: pd.DataFrame, name: str = "fig01_lifecycle_curves"):
    """Two-panel figure:
        left  -- new engagements per tick (the lifecycle "wave")
        right -- cumulative reach (fraction of population engaged at least once)
    The x-axis is clipped to the active window where almost all dynamics occur,
    because the long tail is information-free.
    """
    def _ci(df, m):
        out = df.groupby(["scenario", "tick"])[m].agg(["mean", "std", "count"]).reset_index()
        out["ci"] = 1.96 * out["std"] / np.sqrt(out["count"])
        return out

    new_eng = _ci(tick_log, "n_new_engaged")
    cum = _ci(tick_log.assign(cum_frac=tick_log["cumulative_engaged"] / 1000), "cum_frac")

    # active window: last tick at which median new_engaged is non-trivial,
    # padded by 20 ticks.  We use a global cutoff so all scenarios are visible.
    active_cutoff = 150
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for sc in SCENARIO_ORDER:
        sub = new_eng[new_eng.scenario == sc]
        ax1.plot(sub["tick"], sub["mean"], label=SCENARIO_LABELS[sc],
                 color=PALETTE[sc], lw=1.8)
        ax1.fill_between(sub["tick"], sub["mean"] - sub["ci"], sub["mean"] + sub["ci"],
                         color=PALETTE[sc], alpha=0.15)
        sub2 = cum[cum.scenario == sc]
        ax2.plot(sub2["tick"], sub2["mean"], color=PALETTE[sc], lw=1.8,
                 label=SCENARIO_LABELS[sc])
        ax2.fill_between(sub2["tick"], sub2["mean"] - sub2["ci"], sub2["mean"] + sub2["ci"],
                         color=PALETTE[sc], alpha=0.15)
    ax1.set_xlim(0, active_cutoff)
    ax2.set_xlim(0, active_cutoff)
    ax1.set_xlabel("Time step (≈ hours)")
    ax1.set_ylabel("New engagements per tick")
    ax1.set_title("(a) Per-tick engagement wave")
    ax2.set_xlabel("Time step (≈ hours)")
    ax2.set_ylabel("Cumulative reach (fraction)")
    ax2.set_title("(b) Cumulative viral reach")
    ax1.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Trend lifecycle across scenarios (mean ± 95% CI, 30 reps each)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 2 -- outcome boxplots
# ---------------------------------------------------------------------------


def fig_outcome_boxplots(summary: pd.DataFrame, name: str = "fig02_outcome_boxplots"):
    metrics = [
        ("viral_reach", "Cumulative viral reach (fraction)"),
        ("peak_engaged", "Peak simultaneous engaged (count)"),
        ("time_to_viral", "Time-to-viral (ticks)"),
        ("lifetime", "Trend lifetime (ticks)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    for ax, (m, lab) in zip(axes, metrics):
        sub = summary.copy()
        sub["scenario_short"] = sub["scenario"].map(SCENARIO_LABELS)
        order = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
        sns.boxplot(data=sub, x="scenario_short", y=m, order=order,
                    palette=[PALETTE[s] for s in SCENARIO_ORDER], ax=ax,
                    width=0.55, fliersize=2)
        sns.stripplot(data=sub, x="scenario_short", y=m, order=order,
                      color="black", size=2.2, alpha=0.5, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(lab)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment("right")
    fig.suptitle("Outcome metrics across scenarios (30 replicates each)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 3/4 -- 2x2 interaction plots
# ---------------------------------------------------------------------------


def _interaction_plot(summary: pd.DataFrame, metric: str, ylabel: str, name: str):
    from src.analysis import FACTOR_MAP
    keep = ["S0_NoAlgorithm", "S2_AlgoBoostOnly",
            "S3_HighEngagementOnly", "S4_BoostPlusEngagement"]
    df = summary[summary.scenario.isin(keep)].copy()
    df["engagement"] = df["scenario"].map(lambda s: FACTOR_MAP[s]["engagement"])
    df["algo"] = df["scenario"].map(lambda s: FACTOR_MAP[s]["algo"])

    grouped = df.groupby(["engagement", "algo"])[metric].agg(["mean", "std", "count"]).reset_index()
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci"] = 1.96 * grouped["sem"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = {"low": "#4c72b0", "high": "#c44e52"}
    for eng, sub in grouped.groupby("engagement"):
        sub = sub.set_index("algo").reindex(["off", "boost"]).reset_index()
        ax.errorbar(sub["algo"], sub["mean"], yerr=sub["ci"],
                    marker="o", lw=2, capsize=4, color=colors[eng],
                    label=f"User engagement = {eng}")
    ax.set_xlabel("Algorithm boost")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Two-way interaction: {ylabel}")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, name)


def fig_interaction_peak(summary: pd.DataFrame):
    _interaction_plot(summary, "peak_engaged",
                      "Peak simultaneous engaged",
                      "fig03_2x2_interaction_peak")


def fig_interaction_reach(summary: pd.DataFrame):
    _interaction_plot(summary, "viral_reach",
                      "Cumulative viral reach",
                      "fig04_2x2_interaction_reach")


# ---------------------------------------------------------------------------
# Figure 5 -- Cohen's d heatmap
# ---------------------------------------------------------------------------


def fig_cohen_heatmap(summary: pd.DataFrame, metric: str = "peak_engaged",
                      name: str = "fig05_cohen_heatmap"):
    from src.analysis import cohen_d_matrix
    mat = cohen_d_matrix(summary, metric)
    mat = mat.loc[SCENARIO_ORDER, SCENARIO_ORDER]
    short = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=short, yticklabels=short, ax=ax,
                cbar_kws={"label": "Cohen's d"})
    ax.set_title(f"Effect-size matrix (Cohen's d) for {metric}")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 6 -- network sensitivity
# ---------------------------------------------------------------------------


def fig_network_sensitivity(net_df: pd.DataFrame, name: str = "fig06_network_sensitivity"):
    label_map = {"NET_BA": "Barabási–Albert", "NET_WS": "Watts–Strogatz", "NET_ER": "Erdős–Rényi"}
    sub = net_df.copy()
    sub["topology"] = sub["scenario"].map(label_map)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, m, lab in zip(axes,
                          ["viral_reach", "peak_engaged", "lifetime"],
                          ["Cumulative viral reach",
                           "Peak simultaneous engaged",
                           "Lifetime (ticks)"]):
        sns.boxplot(data=sub, x="topology", y=m, ax=ax,
                    palette=["#4c72b0", "#dd8452", "#55a868"], width=0.55)
        sns.stripplot(data=sub, x="topology", y=m, ax=ax,
                      color="black", size=2.5, alpha=0.6)
        ax.set_xlabel("")
        ax.set_ylabel(lab)
    fig.suptitle("Sensitivity to network topology (Baseline scenario, 15 reps each)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 7 -- state proportion stack
# ---------------------------------------------------------------------------


def fig_state_proportions(tick_log: pd.DataFrame, scenario: str = "S4_BoostPlusEngagement",
                          name: str = "fig07_state_proportions"):
    df = tick_log[tick_log.scenario == scenario].copy()
    agg = df.groupby("tick")[["n_unaware", "n_aware", "n_engaged", "n_fatigued"]].mean()
    agg = agg.div(agg.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stackplot(agg.index,
                 agg["n_unaware"], agg["n_aware"], agg["n_engaged"], agg["n_fatigued"],
                 labels=["UNAWARE", "AWARE", "ENGAGED", "FATIGUED"],
                 colors=["#cccccc", "#f6c14a", "#55a868", "#c44e52"],
                 alpha=0.85)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Population fraction")
    ax.set_xlim(0, agg.index.max())
    ax.set_ylim(0, 1)
    ax.set_title(f"User-state composition over time -- {SCENARIO_LABELS[scenario]}")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 8 -- per-user exposure distributions
# ---------------------------------------------------------------------------


def fig_exposure_distribution(final_states: pd.DataFrame, name: str = "fig08_exposure_distribution"):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for sc in SCENARIO_ORDER:
        x = final_states.loc[(final_states.scenario == sc) & (final_states.exposure_count > 0),
                             "exposure_count"].to_numpy()
        if len(x) == 0:
            continue
        bins = np.arange(0.5, x.max() + 1.5, 1)
        h, edges = np.histogram(x, bins=bins, density=True)
        c = 0.5 * (edges[:-1] + edges[1:])
        # smooth pmf -> step plot
        ax.plot(c, h, drawstyle="steps-mid",
                color=PALETTE[sc], lw=1.6, label=SCENARIO_LABELS[sc])
    ax.set_yscale("log")
    ax.set_xlabel("Per-user exposure count")
    ax.set_ylabel("Probability density (log scale)")
    ax.set_title("Heavy-tailed per-user exposure distributions")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 9 -- correlation heatmap (for the Combined scenario, where dynamics
# are richest)
# ---------------------------------------------------------------------------


def fig_correlation_heatmap(summary: pd.DataFrame, scenario: str = "S4_BoostPlusEngagement",
                            name: str = "fig09_correlation_heatmap"):
    from src.analysis import METRICS
    sub = summary[summary.scenario == scenario][METRICS].copy()
    corr = sub.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"label": "Pearson r"})
    ax.set_title(f"Cross-metric correlations -- {SCENARIO_LABELS[scenario]}")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 10 -- algorithm state timeline ribbon
# ---------------------------------------------------------------------------


def fig_algo_state_timeline(tick_log: pd.DataFrame, name: str = "fig10_algo_state_timeline"):
    # for each scenario compute the modal algorithm state per tick
    state_names = {int(AlgorithmState.PASSIVE): "PASSIVE",
                   int(AlgorithmState.NORMAL): "NORMAL",
                   int(AlgorithmState.BOOSTING): "BOOSTING",
                   int(AlgorithmState.SUPPRESSING): "SUPPRESSING"}
    state_color = {
        "PASSIVE":     "#bdbdbd",
        "NORMAL":      "#4c72b0",
        "BOOSTING":    "#dd8452",
        "SUPPRESSING": "#c44e52",
    }
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(SCENARIO_ORDER) + 0.8))
    for i, sc in enumerate(SCENARIO_ORDER):
        sub = tick_log[tick_log.scenario == sc]
        modal = sub.groupby("tick")["algorithm_state"].apply(
            lambda s: s.value_counts().idxmax())
        modal = modal.reset_index(name="state")
        modal["state_name"] = modal["state"].map(state_names)
        # draw colored segments
        prev = None
        start = 0
        for _, row in modal.iterrows():
            if prev is None:
                prev = row["state_name"]; start = row["tick"]
            elif row["state_name"] != prev:
                ax.barh(i, row["tick"] - start, left=start, height=0.7,
                        color=state_color[prev], edgecolor="none")
                prev = row["state_name"]; start = row["tick"]
        # final segment
        ax.barh(i, modal["tick"].iloc[-1] - start + 1, left=start, height=0.7,
                color=state_color[prev], edgecolor="none")
    ax.set_yticks(range(len(SCENARIO_ORDER)))
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_xlabel("Time step")
    ax.set_title("Modal algorithm state over time, per scenario")
    legend = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in state_color.items()]
    ax.legend(handles=legend, ncol=4, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, -0.18), fontsize=9)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def make_all_figures(data_dir: str = "data"):
    _setup_style()
    summary = pd.read_csv(f"{data_dir}/processed/summary.csv")
    tick_log = pd.read_csv(f"{data_dir}/raw/tick_log.csv.gz")
    final_states = pd.read_csv(f"{data_dir}/raw/final_states.csv.gz")
    net = pd.read_csv(f"{data_dir}/processed/network_sensitivity.csv")

    fig_lifecycle_curves(tick_log)
    fig_outcome_boxplots(summary)
    fig_interaction_peak(summary)
    fig_interaction_reach(summary)
    fig_cohen_heatmap(summary, "peak_engaged")
    fig_network_sensitivity(net)
    fig_state_proportions(tick_log, "S4_BoostPlusEngagement")
    fig_exposure_distribution(final_states)
    fig_correlation_heatmap(summary, "S4_BoostPlusEngagement")
    fig_algo_state_timeline(tick_log)


if __name__ == "__main__":
    make_all_figures()
    print("[+] All figures saved to figures/ and paper/figs/")
