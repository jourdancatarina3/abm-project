"""Generate two additional figures used in the presentation:

    fig11_model_architecture.png   Agent state-machine diagram
    fig12_network_snapshots.png    3 scenarios x 3 time points
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import AlgorithmState, UserState
from src.experiments import scenario_configs
from src.model import ViralityModel


# Colours match the Streamlit UI / paper.
STATE_COLOR = {
    int(UserState.UNAWARE):  "#a8adb5",
    int(UserState.AWARE):    "#c89211",
    int(UserState.ENGAGED):  "#2f7a44",
    int(UserState.FATIGUED): "#a02c2c",
}
STATE_LABEL = {
    int(UserState.UNAWARE):  "Unaware",
    int(UserState.AWARE):    "Aware",
    int(UserState.ENGAGED):  "Engaged",
    int(UserState.FATIGUED): "Fatigued",
}
ALGO_COLOR = {
    int(AlgorithmState.PASSIVE):     "#8b929b",
    int(AlgorithmState.NORMAL):      "#1f4e79",
    int(AlgorithmState.BOOSTING):    "#c66a1a",
    int(AlgorithmState.SUPPRESSING): "#7c1f1f",
}
ALGO_LABEL = {
    int(AlgorithmState.PASSIVE):     "Passive",
    int(AlgorithmState.NORMAL):      "Normal",
    int(AlgorithmState.BOOSTING):    "Boosting",
    int(AlgorithmState.SUPPRESSING): "Suppressing",
}

COLOR_PRIMARY = "#1f4e79"
COLOR_INK     = "#1c1c1c"


# ---------------------------------------------------------------------------
# Figure 11 -- Agent state diagrams
# ---------------------------------------------------------------------------


def _draw_node(ax, x, y, label, color, w=1.4, h=0.7):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=color, edgecolor=COLOR_INK, lw=1.2, zorder=2,
    ))
    ax.text(x, y, label, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", zorder=3)


def _draw_arrow(ax, x1, y1, x2, y2, label="", curve=0.0, color=COLOR_INK,
                label_offset=(0.0, 0.18)):
    style = f"arc3,rad={curve}" if curve != 0 else "arc3,rad=0"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                        connectionstyle=style, mutation_scale=16),
        zorder=1,
    )
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9, color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.22",
                          fc="white", ec="#cfd3d8", alpha=0.95, lw=0.5))


def figure_state_machines(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5),
                             gridspec_kw=dict(wspace=0.12))

    # --- User state machine ----------------------------------------------
    ax = axes[0]
    ax.set_xlim(-0.8, 9.6); ax.set_ylim(-3.0, 2.2)
    ax.axis("off")
    ax.set_title("User agent state machine", color=COLOR_PRIMARY,
                 fontsize=14, fontweight="bold", pad=14)
    pos = {
        int(UserState.UNAWARE):  (0.8, 1.0),
        int(UserState.AWARE):    (3.6, 1.0),
        int(UserState.ENGAGED):  (6.4, 1.0),
        int(UserState.FATIGUED): (6.4, -1.8),
    }
    for sv, (x, y) in pos.items():
        _draw_node(ax, x, y, STATE_LABEL[sv], STATE_COLOR[sv])

    _draw_arrow(ax, 1.55, 1.0, 2.85, 1.0,
                "exposed by neighbour\nor algorithm", label_offset=(0, 0.55))
    _draw_arrow(ax, 4.35, 1.0, 5.65, 1.0,
                "appeal x sensitivity", label_offset=(0, 0.4))
    _draw_arrow(ax, 6.4, 0.6, 6.4, -1.4,
                "exposure ≥ fatigue\nthreshold", label_offset=(1.4, 0))
    _draw_arrow(ax, 3.6, 0.65, 6.0, -1.55,
                "exposure ≥ threshold + 2", curve=-0.35,
                label_offset=(-0.5, -0.7))

    # --- Algorithm state machine ------------------------------------------
    ax = axes[1]
    ax.set_xlim(-0.8, 9.6); ax.set_ylim(-3.0, 2.2)
    ax.axis("off")
    ax.set_title("Platform-algorithm state machine", color=COLOR_PRIMARY,
                 fontsize=14, fontweight="bold", pad=14)
    apos = {
        int(AlgorithmState.PASSIVE):     (0.9, -1.8),
        int(AlgorithmState.NORMAL):      (2.7, 1.0),
        int(AlgorithmState.BOOSTING):    (5.2, 1.0),
        int(AlgorithmState.SUPPRESSING): (7.1, -1.8),
    }
    # use a slightly wider box for the longest label
    for sv, (x, y) in apos.items():
        w = 1.85 if sv == int(AlgorithmState.SUPPRESSING) else 1.5
        _draw_node(ax, x, y, ALGO_LABEL[sv], ALGO_COLOR[sv], w=w)

    _draw_arrow(ax, 3.45, 1.0, 4.45, 1.0,
                "engagement rate\n> boost_threshold",
                label_offset=(0, 0.55))
    _draw_arrow(ax, 5.35, 0.6, 6.7, -1.55,
                "fatigue ratio crossed\nor duration cap",
                label_offset=(0.9, 0.4))
    # Annotate PASSIVE
    ax.text(0.9, -2.55,
            "Passive: the strict\n'no-algorithm' control\nused in scenarios S0 / S3",
            ha="center", va="top", fontsize=9, color="#5b6470",
            style="italic")

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12 -- Network snapshots, 3 scenarios x 3 stages
# ---------------------------------------------------------------------------


def _run_snapshots(name: str, cfg, capture_ticks: list[int]) -> list[dict]:
    """Run a single replicate, capturing snapshots at the requested ticks."""
    model = ViralityModel(cfg)
    model.setup_run()
    snaps = []
    captured = set()
    while not model.finished:
        if model.tick in capture_ticks and model.tick not in captured:
            snaps.append(dict(
                tick=model.tick,
                states=[int(u.state) for u in model.users],
            ))
            captured.add(model.tick)
        model.step()
    # capture final
    if "final" in [str(t) for t in capture_ticks] or max(capture_ticks) == -1:
        snaps.append(dict(tick=model.tick,
                          states=[int(u.state) for u in model.users]))
    return snaps, model


def figure_network_snapshots(out: Path):
    presets = scenario_configs()

    # Pick two strongly contrasting scenarios so the figure stays wide
    # enough to fit a 16:9 slide.
    rows = [
        ("S0  No-Algorithm",        "S0_NoAlgorithm"),
        ("S4  Combined",            "S4_BoostPlusEngagement"),
    ]
    # 3 capture ticks: early seed, mid-burst (~10), late (~80)
    capture_ticks = [2, 10, 80]

    # Run each scenario at 300 users for speed and clarity
    runs = []
    for label, key in rows:
        cfg = replace(presets[key], n_users=300, n_ticks=200, seed=99)
        model = ViralityModel(cfg)
        model.setup_run()
        # Same layout for every panel of the same scenario
        pos = nx.spring_layout(model.graph, seed=99, iterations=60)
        snaps = []
        captured = set()
        while not model.finished:
            if model.tick in capture_ticks and model.tick not in captured:
                snaps.append(dict(
                    tick=model.tick,
                    states=[int(u.state) for u in model.users],
                    n_engaged=sum(1 for u in model.users
                                  if int(u.state) == int(UserState.ENGAGED)),
                    cum_engaged=len(model.ever_engaged),
                ))
                captured.add(model.tick)
            model.step()
        runs.append(dict(label=label, model=model, pos=pos, snaps=snaps))

    fig, axes = plt.subplots(len(rows), len(capture_ticks),
                             figsize=(13.5, 3.6 * len(rows)),
                             squeeze=False)
    for r, run in enumerate(runs):
        graph = run["model"].graph
        pos = run["pos"]
        degrees = np.array([graph.degree(i)
                            for i in range(run["model"].config.n_users)])
        sizes = 5 + 25 * (degrees / max(degrees.max(), 1)) ** 0.6
        xs = np.array([pos[i][0] for i in range(run["model"].config.n_users)])
        ys = np.array([pos[i][1] for i in range(run["model"].config.n_users)])
        for c, snap in enumerate(run["snaps"]):
            ax = axes[r][c]
            # Edges (subsampled if needed)
            edges = list(graph.edges())
            if len(edges) > 2500:
                idx = np.random.default_rng(0).choice(
                    len(edges), 2500, replace=False)
                edges = [edges[i] for i in idx]
            for a, b in edges:
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                        color="#cfd3d8", lw=0.4, zorder=1)
            colors = [STATE_COLOR[s] for s in snap["states"]]
            ax.scatter(xs, ys, s=sizes, c=colors,
                       edgecolors="black", linewidths=0.25, zorder=2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_color("#cfd3d8")
            ax.set_title(f"Tick {snap['tick']}  |  Engaged now: {snap['n_engaged']}",
                         fontsize=10, color=COLOR_INK)
        axes[r][0].set_ylabel(run["label"], color=COLOR_PRIMARY,
                              fontsize=11, fontweight="bold", labelpad=10)

    # Single legend
    handles = [mpatches.Patch(color=STATE_COLOR[sv], label=STATE_LABEL[sv])
               for sv in [int(UserState.UNAWARE), int(UserState.AWARE),
                          int(UserState.ENGAGED), int(UserState.FATIGUED)]]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)
    fig.suptitle("Network evolution: No-Algorithm versus Combined (300 users, identical seed)",
                 color=COLOR_PRIMARY, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    fig_dir = Path("figures")
    paper_fig = Path("paper/figs")
    fig_dir.mkdir(exist_ok=True)
    paper_fig.mkdir(exist_ok=True, parents=True)

    state_path = fig_dir / "fig11_model_architecture.png"
    snap_path  = fig_dir / "fig12_network_snapshots.png"

    print(f"[*] Building {state_path} ...")
    figure_state_machines(state_path)
    figure_state_machines(paper_fig / state_path.name)
    figure_state_machines(paper_fig / "fig11_model_architecture.pdf")

    print(f"[*] Building {snap_path} ...")
    figure_network_snapshots(snap_path)
    figure_network_snapshots(paper_fig / snap_path.name)
    figure_network_snapshots(paper_fig / "fig12_network_snapshots.pdf")

    print("[+] Done.")


if __name__ == "__main__":
    main()
