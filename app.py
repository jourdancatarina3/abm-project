"""Interactive simulator for the Social-Media Virality ABM.

Run with:

    streamlit run app.py

Provides a NetLogo-equivalent dashboard: every model parameter is exposed
as a control on the sidebar; the main pane shows a live network view, a
time-series of new engagements and cumulative reach, the population state
composition over time, and a timeline of the recommendation algorithm's
state.  Scenario presets reproduce the six experimental conditions from
the accompanying paper.
"""

from __future__ import annotations

import time
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agents import AlgorithmState, UserState, UserType
from src.experiments import scenario_configs
from src.model import ModelConfig, ViralityModel


# ---------------------------------------------------------------------------
# Page setup and palette
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Viral Trend Lifecycle ABM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Formal, light-theme palette.  All foreground colours are selected for
# >= 4.5:1 contrast on a white background.
COLOR_PRIMARY    = "#1f4e79"   # deep academic blue
COLOR_ACCENT     = "#a02c2c"   # burgundy
COLOR_INK        = "#1c1c1c"
COLOR_MUTED      = "#5b6470"
COLOR_PANEL      = "#ffffff"
COLOR_PANEL_ALT  = "#f7f7f8"
COLOR_GRID       = "#e1e3e7"

STATE_COLORS = {
    int(UserState.UNAWARE):  "#a8adb5",
    int(UserState.AWARE):    "#c89211",
    int(UserState.ENGAGED):  "#2f7a44",
    int(UserState.FATIGUED): "#a02c2c",
}
STATE_NAMES = {
    int(UserState.UNAWARE):  "Unaware",
    int(UserState.AWARE):    "Aware",
    int(UserState.ENGAGED):  "Engaged",
    int(UserState.FATIGUED): "Fatigued",
}
ALGO_COLORS = {
    int(AlgorithmState.PASSIVE):     "#8b929b",
    int(AlgorithmState.NORMAL):      "#1f4e79",
    int(AlgorithmState.BOOSTING):    "#c66a1a",
    int(AlgorithmState.SUPPRESSING): "#7c1f1f",
}
ALGO_NAMES = {
    int(AlgorithmState.PASSIVE):     "Passive",
    int(AlgorithmState.NORMAL):      "Normal",
    int(AlgorithmState.BOOSTING):    "Boosting",
    int(AlgorithmState.SUPPRESSING): "Suppressing",
}
TYPE_NAMES = {
    int(UserType.INFLUENCER): "Influencer",
    int(UserType.REGULAR):    "Regular",
    int(UserType.LURKER):     "Lurker",
    int(UserType.SKEPTIC):    "Skeptic",
}


# Global CSS to enforce a formal, restrained appearance.
st.markdown(
    f"""
    <style>
      html, body, [class*="css"]  {{
        font-family: "Source Serif Pro", "Georgia", serif;
        color: {COLOR_INK};
      }}
      h1, h2, h3, h4 {{
        font-family: "Source Serif Pro", "Georgia", serif;
        color: {COLOR_INK};
        letter-spacing: -0.01em;
      }}
      .stButton button {{
        border-radius: 4px;
        border: 1px solid #c4c8ce;
        background: #ffffff;
        color: {COLOR_INK};
        font-weight: 500;
      }}
      .stButton button:hover {{
        border-color: {COLOR_PRIMARY};
        color: {COLOR_PRIMARY};
      }}
      .section-divider {{
        border-top: 1px solid {COLOR_GRID};
        margin: 1.0rem 0 0.6rem 0;
      }}
      .header-title {{
        color: {COLOR_PRIMARY};
        margin-bottom: 0.1rem;
        font-weight: 600;
      }}
      .header-subtitle {{
        color: {COLOR_MUTED};
        margin-top: 0;
        font-size: 0.95rem;
      }}
      .algo-pill {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        color: white;
      }}
      .legend-swatch {{
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
        border: 1px solid #d0d4d9;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_state():
    ss = st.session_state
    ss.setdefault("model", None)
    ss.setdefault("layout", None)
    ss.setdefault("running", False)
    ss.setdefault("show_network", True)


def _reset_model():
    st.session_state["model"] = None
    st.session_state["layout"] = None
    st.session_state["running"] = False


def _scenario_defaults(name: str) -> Dict:
    base = scenario_configs()
    if name not in base:
        return {}
    c = base[name]
    return dict(
        n_users=c.n_users,
        n_ticks=c.n_ticks,
        network_kind=c.network_kind,
        ba_m=c.ba_m,
        ws_k=c.ws_k,
        ws_p=c.ws_p,
        er_p=c.er_p,
        intrinsic_appeal=c.intrinsic_appeal,
        novelty_half_life=c.novelty_half_life,
        peer_share_prob=c.peer_share_prob,
        base_view_prob=c.base_view_prob,
        algorithm_initial_state=int(c.algorithm_initial_state),
        boost_threshold=c.boost_threshold,
        boost_inject_prob=c.boost_inject_prob,
        normal_inject_prob=c.normal_inject_prob,
        boost_multiplier=c.boost_multiplier,
        boost_duration_max=c.boost_duration_max,
        suppress_fatigue_ratio=c.suppress_fatigue_ratio,
        seed_strategy=c.seed_strategy,
        n_seeds=c.n_seeds,
        viral_reach_fraction=c.viral_reach_fraction,
    )


PRESETS = {
    "Custom (use the sliders below)": None,
    "S0  No-Algorithm":            "S0_NoAlgorithm",
    "S1  Baseline":                "S1_Baseline",
    "S2  Algorithm-Boost-Only":    "S2_AlgoBoostOnly",
    "S3  High-Engagement-Only":    "S3_HighEngagementOnly",
    "S4  Combined (interaction)":  "S4_BoostPlusEngagement",
    "S5  Influencer-Seed":         "S5_InfluencerSeed",
}


# ---------------------------------------------------------------------------
# Sidebar (controls)
# ---------------------------------------------------------------------------


def _sidebar() -> tuple[ModelConfig, dict]:
    st.sidebar.markdown(f"<h3 style='color:{COLOR_PRIMARY}; margin-top:0;'>"
                        f"Simulation Controls</h3>", unsafe_allow_html=True)

    preset_label = st.sidebar.selectbox(
        "Scenario preset",
        list(PRESETS.keys()),
        index=0,
        help="Select a preset to auto-populate the sliders, or choose Custom.",
    )
    preset_key = PRESETS[preset_label]
    if preset_key is not None and st.sidebar.button(
        "Load preset into sliders", use_container_width=True
    ):
        st.session_state["preset_defaults"] = _scenario_defaults(preset_key)
        _reset_model()
        st.rerun()

    defaults = st.session_state.get("preset_defaults", _scenario_defaults("S1_Baseline"))

    st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Population and network**")
    n_users = st.sidebar.slider("Number of users", 100, 2000,
                                 int(defaults.get("n_users", 500)), step=50,
                                 help="How many people are on the simulated platform.")
    network_kind_label = st.sidebar.radio(
        "Network topology",
        ["Barabasi-Albert (scale-free)",
         "Watts-Strogatz (small-world)",
         "Erdos-Renyi (random)"],
        index={"ba": 0, "ws": 1, "er": 2}[defaults.get("network_kind", "ba")],
        help="How people are connected. Scale-free: a few big influencers and "
             "many ordinary users (like real social media). Small-world: tight "
             "friend clusters linked by a few long-distance shortcuts. "
             "Random: everyone connected by pure chance, no hubs.",
    )
    network_kind = {"Barabasi-Albert (scale-free)": "ba",
                    "Watts-Strogatz (small-world)": "ws",
                    "Erdos-Renyi (random)": "er"}[network_kind_label]
    ba_m = st.sidebar.slider("BA: edges per new node (m)", 1, 10,
                              int(defaults.get("ba_m", 3)),
                              help="For scale-free networks: how many people each "
                                   "new user follows when joining. Higher = a denser, "
                                   "more tightly connected network.")
    if network_kind == "ws":
        ws_k = st.sidebar.slider("WS: nearest neighbours (k)", 2, 12,
                                  int(defaults.get("ws_k", 6)), step=2,
                                  help="For small-world networks: how many neighbours "
                                       "each person starts connected to (the size of "
                                       "their immediate friend group).")
        ws_p = st.sidebar.slider("WS: rewiring probability", 0.0, 1.0,
                                  float(defaults.get("ws_p", 0.1)), step=0.01,
                                  help="For small-world networks: what fraction of local "
                                       "links get swapped for random long-distance "
                                       "shortcuts. Higher = more of a 'small world'.")
    else:
        ws_k = int(defaults.get("ws_k", 6))
        ws_p = float(defaults.get("ws_p", 0.1))
    if network_kind == "er":
        er_p = st.sidebar.slider("ER: edge probability", 0.001, 0.05,
                                  float(defaults.get("er_p", 0.006)),
                                  step=0.001, format="%.3f",
                                  help="For random networks: the chance that any two "
                                       "people are connected. Higher = a denser network.")
    else:
        er_p = float(defaults.get("er_p", 0.006))

    st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Content properties**")
    intrinsic_appeal = st.sidebar.slider(
        "Intrinsic appeal", 0.05, 1.0,
        float(defaults.get("intrinsic_appeal", 0.55)), step=0.05,
        help="How naturally interesting or catchy the content is. Higher = people "
             "engage with it easily on their own.",
    )
    novelty_half_life = st.sidebar.slider(
        "Novelty half-life (ticks)", 1, 480,
        int(defaults.get("novelty_half_life", 96)),
        help="How fast the content gets 'old'. Short = it goes stale quickly; "
             "long = it stays fresh and appealing for longer.",
    )

    st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**User behaviour**")
    peer_share_prob = st.sidebar.slider(
        "Peer-share probability",
        0.0, 1.0, float(defaults.get("peer_share_prob", 0.30)), step=0.05,
        help="Probability that a follower of a sharer is exposed to the content.",
    )
    base_view_prob = st.sidebar.slider(
        "Ambient discovery probability",
        0.0, 0.01, float(defaults.get("base_view_prob", 0.0008)),
        step=0.0002, format="%.4f",
        help="The tiny chance a random person stumbles onto the content on their "
             "own, with no sharing or algorithm involved (background luck).",
    )

    st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Platform algorithm**")
    algo_state_idx = st.sidebar.radio(
        "Initial algorithm state",
        [int(AlgorithmState.PASSIVE), int(AlgorithmState.NORMAL)],
        format_func=lambda i: ALGO_NAMES[i],
        index=0 if defaults.get("algorithm_initial_state",
                                int(AlgorithmState.NORMAL)) == int(AlgorithmState.PASSIVE)
              else 1,
        horizontal=True,
        help="Whether the recommender starts switched off (Passive) or mildly "
             "active (Normal) at the beginning of the simulation.",
    )
    boost_threshold = st.sidebar.slider(
        "Boost-trigger threshold (engagement rate)",
        0.0001, 0.1,
        float(defaults.get("boost_threshold", 0.015)), step=0.0005, format="%.4f",
        help="How much buzz is needed before the algorithm decides 'this is "
             "trending!' and starts pushing it hard. Lower = trips more easily.",
    )
    boost_inject_prob = st.sidebar.slider(
        "Boost-state injection probability",
        0.0, 0.5,
        float(defaults.get("boost_inject_prob", 0.05)), step=0.005, format="%.3f",
        help="While boosting, how aggressively the algorithm shoves the content "
             "into people's feeds. Higher = wider, faster reach.",
    )
    normal_inject_prob = st.sidebar.slider(
        "Normal-state injection probability",
        0.0, 0.05,
        float(defaults.get("normal_inject_prob", 0.002)), step=0.0005, format="%.4f",
        help="The gentle everyday rate at which the algorithm shows the content "
             "before any trending boost kicks in.",
    )
    boost_multiplier = st.sidebar.slider(
        "Boost appeal multiplier",
        0.0, 3.0,
        float(defaults.get("boost_multiplier", 1.0)), step=0.1,
        help="How much more attractive the algorithm makes boosted content look. "
             "Higher = stronger amplification of the trend.",
    )
    boost_duration_max = st.sidebar.slider(
        "Maximum boost duration (ticks)",
        6, 240,
        int(defaults.get("boost_duration_max", 72)),
        help="How long the algorithm will keep boosting a trend before it stops.",
    )
    suppress_fatigue_ratio = st.sidebar.slider(
        "Suppression fatigue ratio",
        0.05, 0.95,
        float(defaults.get("suppress_fatigue_ratio", 0.35)), step=0.05,
        help="When the fraction of touched users that are Fatigued exceeds this value, "
             "the algorithm switches to the Suppressing state.",
    )

    st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Simulation horizon**")
    n_ticks = st.sidebar.slider("Total ticks", 60, 720,
                                 int(defaults.get("n_ticks", 240)), step=20,
                                 help="How long the simulation runs. Each tick is one "
                                      "step of time; more ticks = a longer story.")
    seed_strategy = st.sidebar.radio(
        "Seed origin",
        ["random", "hub"],
        index=0 if defaults.get("seed_strategy", "random") == "random" else 1,
        horizontal=True,
        help="random: a random user starts the trend. hub: the highest-degree user does.",
    )
    n_seeds = st.sidebar.slider("Number of initial seeds", 1, 20,
                                 int(defaults.get("n_seeds", 3)),
                                 help="How many people kick off the trend at the very "
                                      "start of the simulation.")
    seed = st.sidebar.number_input("Random seed", 0, 99_999, 42, step=1,
                                   help="A number that locks in the randomness. Reusing "
                                        "the same seed with the same settings reproduces "
                                        "the exact same run.")

    st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("**Visualisation**")
    speed = st.sidebar.slider("Animation speed (ticks per second)", 1, 60, 12)
    show_network = st.sidebar.checkbox(
        "Show network plot",
        value=st.session_state.get("show_network", True),
        help="Disable for very large populations to keep the page responsive.",
    )
    st.session_state["show_network"] = show_network

    cfg = ModelConfig(
        n_users=n_users,
        n_ticks=n_ticks,
        network_kind=network_kind,
        ba_m=ba_m, ws_k=ws_k, ws_p=ws_p, er_p=er_p,
        intrinsic_appeal=intrinsic_appeal,
        novelty_half_life=novelty_half_life,
        peer_share_prob=peer_share_prob,
        base_view_prob=base_view_prob,
        algorithm_initial_state=AlgorithmState(algo_state_idx),
        boost_threshold=boost_threshold,
        boost_inject_prob=boost_inject_prob,
        normal_inject_prob=normal_inject_prob,
        boost_multiplier=boost_multiplier,
        boost_duration_max=boost_duration_max,
        suppress_fatigue_ratio=suppress_fatigue_ratio,
        seed_strategy=seed_strategy,
        n_seeds=n_seeds,
        seed=int(seed),
    )
    settings = dict(speed=speed)
    return cfg, settings


# ---------------------------------------------------------------------------
# Plotting (Plotly, white background)
# ---------------------------------------------------------------------------


def _common_layout(height: int) -> dict:
    return dict(
        margin=dict(l=10, r=10, t=24, b=10),
        plot_bgcolor=COLOR_PANEL,
        paper_bgcolor=COLOR_PANEL,
        font=dict(color=COLOR_INK, family="Source Serif Pro, Georgia, serif"),
        height=height,
        xaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
                   linecolor=COLOR_GRID),
        yaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
                   linecolor=COLOR_GRID),
    )


def _compute_layout(graph: nx.Graph, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    n = graph.number_of_nodes()
    iters = 80 if n <= 400 else (40 if n <= 1000 else 25)
    pos = nx.spring_layout(graph, seed=seed, iterations=iters)
    xs = np.array([pos[i][0] for i in range(n)])
    ys = np.array([pos[i][1] for i in range(n)])
    return xs, ys


def _network_figure(model: ViralityModel, xs: np.ndarray, ys: np.ndarray) -> go.Figure:
    n = model.config.n_users
    edges = list(model.graph.edges())
    if len(edges) > 4000:
        rng = np.random.default_rng(0)
        edges = [edges[i] for i in rng.choice(len(edges), 4000, replace=False)]
    ex, ey = [], []
    for a, b in edges:
        ex += [xs[a], xs[b], None]
        ey += [ys[a], ys[b], None]

    states = np.array([int(u.state) for u in model.users])
    degrees = np.array([model.graph.degree(i) for i in range(n)])
    sizes = 4 + 14 * (degrees / max(degrees.max(), 1)) ** 0.6
    colors = [STATE_COLORS[s] for s in states]
    user_types = [TYPE_NAMES[int(u.user_type)] for u in model.users]
    hover = [
        f"User {i}<br>"
        f"Type: {user_types[i]}<br>"
        f"Degree: {degrees[i]}<br>"
        f"State: {STATE_NAMES[states[i]]}<br>"
        f"Sensitivity: {model.users[i].sensitivity:.2f}"
        for i in range(n)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.5, color="rgba(80, 90, 110, 0.18)"),
        hoverinfo="none", showlegend=False,
    ))
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors,
                    line=dict(width=0.5, color="rgba(20,20,20,0.45)")),
        text=hover, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    layout = _common_layout(height=540)
    layout["margin"] = dict(l=4, r=4, t=4, b=4)
    layout["xaxis"] = dict(visible=False)
    layout["yaxis"] = dict(visible=False)
    fig.update_layout(**layout)
    return fig


def _timeseries_figure(rows: list[dict]) -> go.Figure:
    if not rows:
        df = pd.DataFrame(dict(tick=[0], n_new_engaged=[0], cumulative_engaged=[0]))
    else:
        df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["tick"], y=df["n_new_engaged"],
        name="New engagements per tick",
        line=dict(color=COLOR_PRIMARY, width=2.4), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=df["tick"], y=df["cumulative_engaged"],
        name="Cumulative engaged",
        line=dict(color=COLOR_ACCENT, width=2.4, dash="dot"), yaxis="y2",
    ))
    layout = _common_layout(height=270)
    layout["yaxis"] = dict(title="New engagements per tick",
                           color=COLOR_PRIMARY, gridcolor=COLOR_GRID,
                           zerolinecolor=COLOR_GRID, linecolor=COLOR_GRID)
    layout["yaxis2"] = dict(title="Cumulative engaged", overlaying="y",
                            side="right", color=COLOR_ACCENT)
    layout["xaxis"] = dict(title="Tick", gridcolor=COLOR_GRID,
                           zerolinecolor=COLOR_GRID, linecolor=COLOR_GRID)
    layout["legend"] = dict(orientation="h", y=1.18,
                            bgcolor="rgba(255,255,255,0.95)")
    fig.update_layout(**layout, showlegend=True)
    return fig


def _state_area_figure(rows: list[dict], n_users: int) -> go.Figure:
    if not rows:
        rows = [dict(tick=0, n_unaware=n_users, n_aware=0, n_engaged=0, n_fatigued=0)]
    df = pd.DataFrame(rows)

    fig = go.Figure()
    cumulative = np.zeros(len(df))
    for col, label, sval in [
        ("n_unaware", "Unaware",  int(UserState.UNAWARE)),
        ("n_aware",   "Aware",    int(UserState.AWARE)),
        ("n_engaged", "Engaged",  int(UserState.ENGAGED)),
        ("n_fatigued","Fatigued", int(UserState.FATIGUED)),
    ]:
        cumulative = cumulative + df[col].to_numpy()
        fig.add_trace(go.Scatter(
            x=df["tick"], y=cumulative,
            name=label, mode="lines",
            line=dict(width=0.3, color=STATE_COLORS[sval]),
            fill="tonexty" if label != "Unaware" else "tozeroy",
            fillcolor=STATE_COLORS[sval],
            opacity=0.92,
        ))
    layout = _common_layout(height=270)
    layout["yaxis"] = dict(title="Population count", range=[0, n_users],
                           gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
                           linecolor=COLOR_GRID)
    layout["xaxis"] = dict(title="Tick", gridcolor=COLOR_GRID,
                           zerolinecolor=COLOR_GRID, linecolor=COLOR_GRID)
    layout["legend"] = dict(orientation="h", y=1.18,
                            bgcolor="rgba(255,255,255,0.95)")
    fig.update_layout(**layout)
    return fig


def _algorithm_strip(rows: list[dict]) -> go.Figure:
    if not rows:
        rows = [dict(tick=0, algorithm_state=int(AlgorithmState.NORMAL))]
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for state_val, label in ALGO_NAMES.items():
        mask = df["algorithm_state"] == state_val
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "tick"], y=[0] * int(mask.sum()),
            mode="markers",
            marker=dict(size=10, color=ALGO_COLORS[state_val], symbol="square"),
            name=label, hovertemplate=label + " at tick %{x}<extra></extra>",
        ))
    layout = _common_layout(height=90)
    layout["yaxis"] = dict(visible=False, range=[-0.5, 0.5])
    layout["xaxis"] = dict(title="Algorithm state over time",
                           gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
                           linecolor=COLOR_GRID)
    layout["legend"] = dict(orientation="h", y=-0.55,
                            bgcolor="rgba(255,255,255,0.95)")
    fig.update_layout(**layout, showlegend=True)
    return fig


# ---------------------------------------------------------------------------
# Plain-language interpretation of setup and results
# ---------------------------------------------------------------------------


_NETWORK_NAME = {
    "ba": "Barabasi-Albert scale-free graph (a few highly-connected influencers, many ordinary users)",
    "ws": "Watts-Strogatz small-world graph (everyone has a tight local circle plus occasional long-range ties)",
    "er": "Erdos-Renyi random graph (every pair of users is equally likely to be connected)",
}


def _classify_setup(cfg: ModelConfig) -> str:
    """Roughly identify which experimental regime the current config falls into."""
    high_engagement = cfg.peer_share_prob >= 0.45
    boost_active = (
        cfg.algorithm_initial_state != AlgorithmState.PASSIVE
        and cfg.boost_inject_prob >= 0.08
        and cfg.boost_threshold <= 0.005
    )
    algo_off = cfg.algorithm_initial_state == AlgorithmState.PASSIVE
    if high_engagement and boost_active:
        return ("Combined regime (high peer-sharing and aggressive algorithm boosting). "
                "This is the configuration where the model predicts the largest peak "
                "intensity due to the interaction between the two mechanisms.")
    if high_engagement and algo_off:
        return ("Organic-only regime. The algorithm is switched off, so the trend can "
                "spread only through user-to-user sharing. The model predicts wide "
                "reach but a moderate peak.")
    if boost_active and not high_engagement:
        return ("Algorithm-only regime. Peer-sharing is weak, so the algorithm has to "
                "do all the work. The model predicts a sharp burst followed by an "
                "early collapse once users fatigue.")
    if algo_off and not high_engagement:
        return ("No-algorithm baseline. Pure word-of-mouth with moderate sharing. The "
                "model predicts a slow burn with limited reach.")
    return ("Standard platform regime. The algorithm acts in its Normal state and "
            "users share at a moderate rate. The trend grows steadily and decays "
            "gradually.")


def _interpret_setup(cfg: ModelConfig) -> str:
    """Plain-language description of the current configuration."""
    network = _NETWORK_NAME.get(cfg.network_kind, cfg.network_kind)
    algo_initial = ALGO_NAMES[int(cfg.algorithm_initial_state)]
    seed_phrase = ("a small number of random users" if cfg.seed_strategy == "random"
                   else "the most-followed (hub) users")
    days = cfg.n_ticks / 24.0

    parts = [
        f"**Population.** {cfg.n_users} users connected as a {network}.",
        f"**Content.** A single piece of content with intrinsic appeal "
        f"{cfg.intrinsic_appeal:.2f} on a 0-to-1 scale. Its novelty halves "
        f"every {int(cfg.novelty_half_life)} ticks "
        f"(~{cfg.novelty_half_life / 24:.1f} days), simulating how a fresh "
        f"trend gradually feels old.",
        f"**Sharing behaviour.** When a user shares, each of their followers "
        f"has a {int(cfg.peer_share_prob * 100)}% chance of being exposed.",
        f"**Algorithm.** The recommender starts in the {algo_initial} state. "
        f"It will trip into Boosting once the engagement rate exceeds "
        f"{cfg.boost_threshold:.4f}; during Boosting it injects the content "
        f"into the feed of each Unaware user with probability "
        f"{cfg.boost_inject_prob:.3f} per tick.",
        f"**Seed.** The trend starts from {cfg.n_seeds} initial poster(s) "
        f"chosen as {seed_phrase}.",
        f"**Time horizon.** {cfg.n_ticks} ticks of simulated platform time "
        f"(~{days:.1f} days at one tick per hour).",
        f"**Regime.** {_classify_setup(cfg)}",
    ]
    return "\n\n".join(parts)


def _interpret_results(snap: dict, cfg: ModelConfig) -> str:
    """Plain-language summary of what just happened."""
    n = cfg.n_users
    tick = snap["tick"]
    cum_reach = snap["cumulative_engaged"]
    reach_pct = cum_reach / n * 100.0
    peak = snap["peak_engaged"]
    peak_tick = snap["peak_tick"]
    ttv = snap["time_to_viral"]
    fatigued = snap["n_fatigued"]
    unaware = snap["n_unaware"]
    engaged_now = snap["n_engaged"]
    algo = snap["algorithm_state"]

    finished = snap["finished"]
    rows = snap["tick_rows"]
    new_eng = [r["n_new_engaged"] for r in rows] if rows else []

    # Verdict on virality
    if reach_pct >= 30:
        viral_verdict = ("**The trend went strongly viral.** "
                         f"About {reach_pct:.1f}% of the population "
                         f"({cum_reach} out of {n} users) engaged with it.")
    elif reach_pct >= 5:
        viral_verdict = (f"**The trend crossed the viral threshold** (5% reach), "
                         f"finishing at {reach_pct:.1f}% cumulative engagement.")
    else:
        viral_verdict = (f"**The trend did not go viral.** Only "
                         f"{reach_pct:.1f}% of users ever engaged. Most of "
                         f"the network was never exposed to it.")

    # Speed
    if ttv >= 0:
        speed_phrase = (
            f"It reached viral status in **{ttv} ticks "
            f"({ttv / 24:.1f} simulated days)**, which is "
            + ("very fast." if ttv <= 6 else
               "moderately fast." if ttv <= 20 else
               "slow.")
        )
    else:
        speed_phrase = "It never reached the 5% viral threshold."

    # Peak
    if peak > 0:
        peak_phrase = (
            f"At its peak, **{peak} users were simultaneously engaged** in "
            f"the same tick (tick {peak_tick}). This is the trend's "
            "intensity, distinct from its eventual breadth."
        )
    else:
        peak_phrase = "The peak engagement was effectively zero."

    # Decay reason
    if finished or tick > 0:
        unaware_pct = unaware / n * 100.0
        fatigued_pct = fatigued / n * 100.0
        if fatigued_pct > 0.5 * (fatigued_pct + unaware_pct):
            decay_phrase = (
                f"**Why it died.** The trend was eventually killed by "
                f"audience fatigue: {fatigued_pct:.0f}% of users are now in "
                f"the Fatigued state and actively dismiss the content. "
                f"Roughly {unaware_pct:.0f}% of users on the network "
                f"periphery never even saw it."
            )
        else:
            decay_phrase = (
                f"**Why it died.** The trend ran out of new audience: "
                f"{unaware_pct:.0f}% of users remain Unaware and are no "
                f"longer being reached, while {fatigued_pct:.0f}% have "
                f"become Fatigued."
            )
    else:
        decay_phrase = ""

    # Algorithm role
    if cfg.algorithm_initial_state == AlgorithmState.PASSIVE:
        algo_phrase = ("**Role of the algorithm.** Switched off for this run. "
                       "Everything you see is the result of peer-to-peer "
                       "sharing alone.")
    else:
        if algo == int(AlgorithmState.SUPPRESSING):
            algo_phrase = ("**Role of the algorithm.** It boosted the trend "
                           "early on, then switched to Suppressing once too "
                           "many users became Fatigued. This is the same "
                           "pattern that real platforms use to stop a stale "
                           "trend from clogging feeds.")
        elif algo == int(AlgorithmState.BOOSTING):
            algo_phrase = ("**Role of the algorithm.** Currently boosting the "
                           "content; the engagement rate stayed high enough "
                           "that the algorithm kept amplifying it.")
        else:
            algo_phrase = ("**Role of the algorithm.** It remained in its "
                           "Normal state. The engagement rate never crossed "
                           "the threshold required to trigger an active boost.")

    # Lifecycle shape
    if peak > 0 and len(new_eng) > peak_tick + 6:
        post_peak = new_eng[peak_tick + 1:]
        half_peak = peak / 2.0
        ticks_to_half = next(
            (i + 1 for i, v in enumerate(post_peak) if v <= half_peak),
            len(post_peak),
        )
        if ticks_to_half <= 3:
            shape_phrase = ("**Shape.** A sharp spike followed by a near-"
                            "vertical collapse — typical of an algorithm-"
                            "amplified burst that exhausted its audience.")
        elif ticks_to_half <= 12:
            shape_phrase = ("**Shape.** A clean wave: rapid rise, brief plateau, "
                            "and a clear decay over a few simulated days.")
        else:
            shape_phrase = ("**Shape.** A long, gentle slope — consistent with "
                            "organic word-of-mouth diffusion without strong "
                            "algorithmic amplification.")
    else:
        shape_phrase = ""

    parts = [viral_verdict, speed_phrase, peak_phrase, decay_phrase,
             algo_phrase, shape_phrase]
    if not finished and tick > 0:
        parts.append(
            "*The simulation is still in progress. The summary above reflects "
            "the state at tick " f"{tick} of {cfg.n_ticks}.*"
        )
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Rendering loop
# ---------------------------------------------------------------------------


def _render(model: ViralityModel, settings: dict, slots: dict):
    snap = model.snapshot()
    n = model.config.n_users
    cum_reach = snap["cumulative_engaged"]

    with slots["metrics"].container():
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Tick", f"{snap['tick']} / {model.config.n_ticks}")
        c2.metric("Cumulative reach",
                  f"{cum_reach / n * 100:.1f}%",
                  delta=f"{cum_reach} users", delta_color="off")
        c3.metric("Currently engaged", snap["n_engaged"])
        c4.metric("Fatigued", snap["n_fatigued"])
        c5.metric("Peak so far",
                  f"{snap['peak_engaged']} at t={snap['peak_tick']}")
        c6.metric("Time-to-viral",
                  f"{snap['time_to_viral']}" if snap['time_to_viral'] >= 0 else "—")

    algo = snap["algorithm_state"]
    slots["algo_state"].markdown(
        f"<div class='algo-pill' style='background:{ALGO_COLORS[algo]};'>"
        f"Algorithm state: {ALGO_NAMES[algo]}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state["show_network"] and st.session_state["layout"] is not None:
        xs, ys = st.session_state["layout"]
        slots["network"].plotly_chart(
            _network_figure(model, xs, ys),
            use_container_width=True,
            key=f"net_{snap['tick']}",
        )

    slots["timeseries"].plotly_chart(
        _timeseries_figure(snap["tick_rows"]),
        use_container_width=True,
        key=f"ts_{snap['tick']}",
    )
    slots["statearea"].plotly_chart(
        _state_area_figure(snap["tick_rows"], n),
        use_container_width=True,
        key=f"sa_{snap['tick']}",
    )
    slots["algostrip"].plotly_chart(
        _algorithm_strip(snap["tick_rows"]),
        use_container_width=True,
        key=f"as_{snap['tick']}",
    )

    # Plain-language interpretation slot
    if "interpretation" in slots:
        with slots["interpretation"].container():
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(
                    f"<h4 style='color:{COLOR_PRIMARY}; margin-top:0;'>"
                    f"Current setup, in plain language</h4>",
                    unsafe_allow_html=True,
                )
                st.markdown(_interpret_setup(model.config))
            with col_right:
                heading = ("What just happened" if snap["finished"]
                           else "What is happening so far")
                st.markdown(
                    f"<h4 style='color:{COLOR_PRIMARY}; margin-top:0;'>"
                    f"{heading}</h4>",
                    unsafe_allow_html=True,
                )
                st.markdown(_interpret_results(snap, model.config))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _init_state()
    cfg, settings = _sidebar()

    # Header
    st.markdown(
        f"<h2 class='header-title'>Social-Media Virality and Content Lifecycle</h2>"
        f"<p class='header-subtitle'>Interactive Agent-Based Simulator &nbsp;|&nbsp; "
        f"CMSC 176 Final Project, Team 8</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Control row
    ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns([1, 1, 1, 1, 4])
    setup_clicked = ctl1.button("Setup", use_container_width=True)
    run_clicked   = ctl2.button("Run",   use_container_width=True,
                                disabled=st.session_state["model"] is None)
    step_clicked  = ctl3.button("Step",  use_container_width=True,
                                disabled=st.session_state["model"] is None)
    reset_clicked = ctl4.button("Reset", use_container_width=True)
    ctl5.markdown(
        f"<div style='padding-top:6px; color:{COLOR_MUTED}; font-size:0.92rem;'>"
        f"Click <b>Setup</b> after changing parameters. <b>Run</b> animates the simulation. "
        f"<b>Step</b> advances one tick. <b>Reset</b> clears the current run."
        f"</div>", unsafe_allow_html=True,
    )

    if reset_clicked:
        _reset_model()
        st.rerun()

    if setup_clicked:
        with st.spinner("Generating network and population..."):
            model = ViralityModel(cfg)
            model.setup_run()
            xs, ys = _compute_layout(model.graph, seed=cfg.seed)
        st.session_state["model"] = model
        st.session_state["layout"] = (xs, ys)
        st.session_state["running"] = False
        st.rerun()

    # Slot layout
    slots = {}
    slots["metrics"] = st.empty()
    slots["algo_state"] = st.empty()
    if st.session_state["show_network"]:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Social network**  &nbsp;<span style='color:" + COLOR_MUTED +
                        ";font-size:0.85rem;'>(nodes coloured by state, sized by degree)</span>",
                        unsafe_allow_html=True)
            slots["network"] = st.empty()
        with right:
            st.markdown("**Engagement and cumulative reach**")
            slots["timeseries"] = st.empty()
            st.markdown("**State composition over time**")
            slots["statearea"] = st.empty()
    else:
        st.markdown("**Engagement and cumulative reach**")
        slots["timeseries"] = st.empty()
        st.markdown("**State composition over time**")
        slots["statearea"] = st.empty()
    st.markdown("**Algorithm-state timeline**")
    slots["algostrip"] = st.empty()

    # Legend / quick reference
    with st.expander("Legend and model quick-reference"):
        cols = st.columns(4)
        for i, (sv, name) in enumerate(STATE_NAMES.items()):
            cols[i].markdown(
                f"<span class='legend-swatch' "
                f"style='background:{STATE_COLORS[sv]};'></span>"
                f"<b>{name}</b>",
                unsafe_allow_html=True,
            )
        st.write("")
        cols = st.columns(4)
        for i, (sv, name) in enumerate(ALGO_NAMES.items()):
            cols[i].markdown(
                f"<span class='legend-swatch' "
                f"style='background:{ALGO_COLORS[sv]};'></span>"
                f"<b>{name}</b>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "**User state transitions.** "
            "Users move through Unaware -> Aware -> Engaged -> Fatigued. A user "
            "becomes Aware when a neighbour shares the content or the algorithm "
            "injects it into their feed. Engagement is probabilistic in personal "
            "sensitivity multiplied by current content appeal. Repeated exposures "
            "drive the user into the Fatigued absorbing state, which acts as a "
            "natural trend dampener.\n\n"
            "**Algorithm state transitions.** "
            "Normal -> Boosting when the windowed engagement rate crosses the "
            "boost threshold. Boosting -> Suppressing when the share of touched "
            "users that are Fatigued exceeds the suppression ratio, or after the "
            "maximum boost duration elapses. Passive is a strict no-injection "
            "control used to isolate the contribution of algorithmic distribution."
        )

    # Summary and interpretation panel
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 style='color:{COLOR_PRIMARY}; margin-bottom:0.1rem;'>"
        f"Summary and interpretation</h3>"
        f"<p style='color:{COLOR_MUTED}; margin-top:0; font-size:0.9rem;'>"
        f"A plain-language description of the current configuration and what "
        f"the simulation produced, intended for a non-technical audience."
        f"</p>",
        unsafe_allow_html=True,
    )
    slots["interpretation"] = st.empty()

    # Initial empty state
    if st.session_state["model"] is None:
        slots["metrics"].info(
            "Configure the parameters in the sidebar, then click Setup."
        )
        slots["timeseries"].plotly_chart(_timeseries_figure([]),
                                          use_container_width=True, key="ts_init")
        slots["statearea"].plotly_chart(_state_area_figure([], cfg.n_users),
                                         use_container_width=True, key="sa_init")
        slots["algostrip"].plotly_chart(_algorithm_strip([]),
                                         use_container_width=True, key="as_init")
        with slots["interpretation"].container():
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(
                    f"<h4 style='color:{COLOR_PRIMARY}; margin-top:0;'>"
                    f"Current setup, in plain language</h4>",
                    unsafe_allow_html=True,
                )
                st.markdown(_interpret_setup(cfg))
            with col_right:
                st.markdown(
                    f"<h4 style='color:{COLOR_PRIMARY}; margin-top:0;'>"
                    f"What just happened</h4>",
                    unsafe_allow_html=True,
                )
                st.info("Click Setup, then Run, to generate a results summary.")
        return

    model = st.session_state["model"]
    _render(model, settings, slots)

    if step_clicked and not model.finished:
        model.step()
        _render(model, settings, slots)

    if run_clicked and not model.finished:
        delay = 1.0 / max(settings["speed"], 1)
        max_burst = max(1, int(round(settings["speed"] / 8)))
        while not model.finished:
            for _ in range(max_burst):
                if model.finished:
                    break
                model.step()
            _render(model, settings, slots)
            time.sleep(delay)
        st.success(
            "Simulation complete. Inspect the curves and metrics above. "
            "Adjust the sliders and click Setup to begin a new run."
        )


if __name__ == "__main__":
    main()
