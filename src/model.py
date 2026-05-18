"""Core ABM driver.

Two entry points are exposed:

    ViralityModel.run()
        Run a complete simulation from setup to the final tick and return a
        RunResult.  This is what experiments.py uses for batch sweeps.

    ViralityModel.setup_run()  +  ViralityModel.step()  +  ViralityModel.snapshot()
        Stateful, tick-by-tick interface.  The interactive Streamlit app
        uses this so it can render the network and time-series live.

Each tick (~1 hour of platform time):
    1.  content novelty decays
    2.  algorithm updates its state, optionally injecting the content
        into the feeds of unaware users
    3.  users transition between (UNAWARE -> AWARE -> ENGAGED ->
        FATIGUED) based on personal sensitivity, content appeal, and
        peer-share signals from neighbours
    4.  aggregate metrics are recorded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .agents import (
    AlgorithmState,
    PlatformAlgorithm,
    UserAgent,
    UserState,
    UserType,
    build_user_population,
)
from .content import Content
from .network import build_network, top_k_hubs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    # population / network
    n_users: int = 1000
    network_kind: str = "ba"          # "ba", "ws", "er"
    ba_m: int = 3
    ws_k: int = 6
    ws_p: float = 0.1
    er_p: float = 0.006

    # content
    intrinsic_appeal: float = 0.55
    novelty_half_life: float = 96.0   # ticks (~4 days)
    seed_strategy: str = "random"     # "random" or "hub"
    n_seeds: int = 3

    # user behaviour
    peer_share_prob: float = 0.40
    base_view_prob: float = 0.0008

    # algorithm
    algorithm_initial_state: AlgorithmState = AlgorithmState.NORMAL
    boost_threshold: float = 0.015
    boost_inject_prob: float = 0.05
    normal_inject_prob: float = 0.002
    boost_multiplier: float = 1.0
    boost_duration_max: int = 72
    suppress_fatigue_ratio: float = 0.35

    # simulation
    n_ticks: int = 480
    viral_reach_fraction: float = 0.05
    seed: int = 0


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    config: ModelConfig
    tick_log: pd.DataFrame
    summary: dict
    final_states: pd.DataFrame


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ViralityModel:
    """An agent-based model of social-media trend lifecycle."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.graph: Optional[nx.Graph] = None
        self.users: List[UserAgent] = []
        self.neighbors: List[np.ndarray] = []
        self.algorithm = PlatformAlgorithm(
            boost_threshold=config.boost_threshold,
            boost_inject_prob=config.boost_inject_prob,
            normal_inject_prob=config.normal_inject_prob,
            boost_multiplier=config.boost_multiplier,
            boost_duration_max=config.boost_duration_max,
            suppress_fatigue_ratio=config.suppress_fatigue_ratio,
            initial_state=config.algorithm_initial_state,
        )
        self.content = Content(
            intrinsic_appeal=config.intrinsic_appeal,
            novelty_half_life=config.novelty_half_life,
        )

        # tick-by-tick state (populated by setup_run / step)
        self.tick: int = 0
        self.unaware: set[int] = set()
        self.aware: set[int] = set()
        self.engaged: set[int] = set()
        self.fatigued: set[int] = set()
        self.ever_engaged: set[int] = set()
        self.shared_last_tick: set[int] = set()
        self.tick_rows: list[dict] = []
        self.finished: bool = False

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        c = self.config
        self.graph = build_network(
            c.network_kind, c.n_users, self.rng,
            m=c.ba_m, ws_k=c.ws_k, ws_p=c.ws_p, er_p=c.er_p,
        )
        degrees = [self.graph.degree(i) for i in range(c.n_users)]
        self.users = build_user_population(degrees, self.rng)
        self.neighbors = [
            np.array(list(self.graph.neighbors(i)), dtype=np.int64)
            for i in range(c.n_users)
        ]
        self.algorithm.reset()
        self.content = Content(
            intrinsic_appeal=c.intrinsic_appeal,
            novelty_half_life=c.novelty_half_life,
        )

    def _select_seeds(self) -> List[int]:
        c = self.config
        if c.seed_strategy == "hub":
            return top_k_hubs(self.graph, c.n_seeds)
        return list(self.rng.choice(c.n_users, size=c.n_seeds, replace=False))

    # ----------------------------------------------------- tick-by-tick API
    def setup_run(self) -> None:
        """Prepare a fresh simulation in tick-by-tick mode."""
        self.setup()
        c = self.config
        self.tick = 0
        self.unaware = set(range(c.n_users))
        self.aware = set()
        self.engaged = set()
        self.fatigued = set()
        self.ever_engaged = set()
        self.tick_rows = []
        self.finished = False

        for u in self.users:
            u.reset()

        # seed
        seeds = self._select_seeds()
        for sid in seeds:
            self.users[sid].state = UserState.ENGAGED
            self.users[sid].became_aware_at = 0
            self.users[sid].became_engaged_at = 0
            self.users[sid].has_shared = True
            self.unaware.discard(sid)
            self.engaged.add(sid)
            self.ever_engaged.add(sid)
        self.shared_last_tick = set(seeds)

    def step(self) -> dict:
        """Run a single tick and return its row of metrics."""
        if self.finished:
            return self.tick_rows[-1] if self.tick_rows else {}
        c = self.config
        t = self.tick
        all_n = c.n_users

        appeal = self.content.current_appeal(t) * self.algorithm.appeal_multiplier()

        inject_p = self.algorithm.inject_prob()
        algo_pushed: set[int] = set()
        if inject_p > 0 and self.unaware:
            ua = np.fromiter(self.unaware, dtype=np.int64)
            mask = self.rng.random(len(ua)) < inject_p
            algo_pushed = set(ua[mask].tolist())

        peer_exposed: set[int] = set()
        if self.shared_last_tick:
            for sid in self.shared_last_tick:
                nbrs = self.neighbors[sid]
                if nbrs.size == 0:
                    continue
                mask = self.rng.random(nbrs.size) < c.peer_share_prob
                peer_exposed.update(nbrs[mask].tolist())
            peer_exposed -= self.engaged
            peer_exposed -= self.fatigued

        if c.base_view_prob > 0 and self.unaware:
            ua = np.fromiter(self.unaware, dtype=np.int64)
            mask = self.rng.random(len(ua)) < c.base_view_prob
            peer_exposed.update(ua[mask].tolist())

        newly_exposed = (algo_pushed | peer_exposed) & self.unaware

        new_aware: set[int] = set()
        for uid in newly_exposed:
            u = self.users[uid]
            u.state = UserState.AWARE
            u.became_aware_at = t
            u.exposure_count += 1
            new_aware.add(uid)
        self.unaware -= new_aware
        self.aware |= new_aware

        aware_arr = np.fromiter(self.aware, dtype=np.int64)
        self.rng.shuffle(aware_arr)
        new_engaged_set: set[int] = set()
        for uid in aware_arr:
            u = self.users[uid]
            p_engage = float(np.clip(appeal * u.sensitivity, 0.0, 1.0))
            if self.rng.random() < p_engage:
                u.state = UserState.ENGAGED
                u.became_engaged_at = t
                new_engaged_set.add(uid)
        self.aware -= new_engaged_set
        self.engaged |= new_engaged_set
        self.ever_engaged |= new_engaged_set

        shared_this_tick: set[int] = set()
        new_fatigued: set[int] = set()
        for uid in list(self.engaged):
            u = self.users[uid]
            if not u.has_shared and self.rng.random() < u.share_propensity:
                u.has_shared = True
                shared_this_tick.add(uid)
            u.exposure_count += 1
            if u.exposure_count >= u.fatigue_threshold:
                if self.rng.random() < 0.5:
                    u.state = UserState.FATIGUED
                    u.became_fatigued_at = t
                    new_fatigued.add(uid)
        self.engaged -= new_fatigued
        self.fatigued |= new_fatigued

        extra_fat: set[int] = set()
        for uid in list(self.aware):
            u = self.users[uid]
            u.exposure_count += 1
            if u.exposure_count >= u.fatigue_threshold + 2:
                u.state = UserState.FATIGUED
                u.became_fatigued_at = t
                extra_fat.add(uid)
        self.aware -= extra_fat
        self.fatigued |= extra_fat

        self.algorithm.step(
            new_engaged=len(new_engaged_set),
            n_aware=len(self.aware),
            n_fatigued=len(self.fatigued),
            n_total=all_n,
        )

        cum_engaged = len(self.ever_engaged)
        if self.content.time_to_viral < 0 and cum_engaged >= c.viral_reach_fraction * all_n:
            self.content.time_to_viral = t
        if len(new_engaged_set) > self.content.peak_engaged:
            self.content.peak_engaged = len(new_engaged_set)
            self.content.peak_tick = t

        row = dict(
            tick=t,
            appeal=appeal,
            algorithm_state=int(self.algorithm.state),
            inject_prob=inject_p,
            n_unaware=len(self.unaware),
            n_aware=len(self.aware),
            n_engaged=len(self.engaged),
            n_fatigued=len(self.fatigued),
            n_new_engaged=len(new_engaged_set),
            n_new_aware=len(new_aware),
            n_new_fatigued=len(new_fatigued) + len(extra_fat),
            cumulative_engaged=cum_engaged,
            cumulative_reach=cum_engaged + len(self.aware) + len(self.fatigued),
            shared_this_tick=len(shared_this_tick),
        )
        self.tick_rows.append(row)

        self.shared_last_tick = shared_this_tick
        self.tick += 1
        if self.tick >= c.n_ticks:
            self.finished = True
        return row

    def snapshot(self) -> dict:
        """Return a compact dict the UI can use to render the current state."""
        states = np.array([int(u.state) for u in self.users], dtype=np.int32)
        return dict(
            tick=self.tick,
            algorithm_state=int(self.algorithm.state),
            states=states,
            n_unaware=int((states == int(UserState.UNAWARE)).sum()),
            n_aware=int((states == int(UserState.AWARE)).sum()),
            n_engaged=int((states == int(UserState.ENGAGED)).sum()),
            n_fatigued=int((states == int(UserState.FATIGUED)).sum()),
            cumulative_engaged=len(self.ever_engaged),
            peak_engaged=int(self.content.peak_engaged),
            peak_tick=int(self.content.peak_tick),
            time_to_viral=int(self.content.time_to_viral),
            tick_rows=list(self.tick_rows),
            finished=self.finished,
        )

    # ------------------------------------------------------------------- run
    def run(self) -> RunResult:
        self.setup_run()
        while not self.finished:
            self.step()
        tick_log = pd.DataFrame(self.tick_rows)
        summary = self._summarise(tick_log, self.ever_engaged)
        final_states = self._final_state_df()
        return RunResult(self.config, tick_log, summary, final_states)

    # ------------------------------------------------------------ summarise
    def _summarise(self, tick_log: pd.DataFrame, ever_engaged: set[int]) -> dict:
        c = self.config
        new_eng = tick_log["n_new_engaged"].to_numpy()
        viral_reach = len(ever_engaged) / c.n_users
        peak_tick = int(np.argmax(new_eng)) if new_eng.max() > 0 else -1
        peak_engaged = int(new_eng.max())

        window = 6
        if peak_engaged > 0:
            smooth = pd.Series(new_eng).rolling(window, min_periods=1).mean().to_numpy()
            active = smooth > max(1.0, 0.01 * peak_engaged)
            first = int(np.argmax(active))
            last = int(len(active) - 1 - np.argmax(active[::-1]))
            lifetime = max(0, last - first)
        else:
            lifetime = 0

        decay_rate = 0.0
        if peak_tick >= 0 and peak_tick < len(new_eng) - 5:
            tail = pd.Series(new_eng[peak_tick:]).rolling(window, min_periods=1).mean()
            tail = tail[tail > 0]
            if len(tail) >= 5:
                x = np.arange(len(tail))
                y = np.log(tail.to_numpy())
                decay_rate = float(np.polyfit(x, y, 1)[0])

        auc_engaged = float(tick_log["n_engaged"].sum())

        return dict(
            viral_reach=viral_reach,
            peak_engaged=peak_engaged,
            peak_tick=peak_tick,
            time_to_viral=self.content.time_to_viral,
            lifetime=lifetime,
            decay_rate=decay_rate,
            auc_engaged=auc_engaged,
            final_fatigued=int(tick_log["n_fatigued"].iloc[-1]),
        )

    def _final_state_df(self) -> pd.DataFrame:
        rows = []
        for u in self.users:
            rows.append(dict(
                uid=u.uid,
                user_type=int(u.user_type),
                follower_count=u.follower_count,
                sensitivity=u.sensitivity,
                share_propensity=u.share_propensity,
                social_influence=u.social_influence,
                state=int(u.state),
                exposure_count=u.exposure_count,
                became_aware_at=u.became_aware_at,
                became_engaged_at=u.became_engaged_at,
                became_fatigued_at=u.became_fatigued_at,
                has_shared=u.has_shared,
            ))
        return pd.DataFrame(rows)
