"""Agent definitions for the social-media virality ABM.

Two agent classes are defined:

    UserAgent          -- one per node in the social graph
    PlatformAlgorithm  -- a single, global recommender-system agent

The original proposal mentioned only a single homogeneous "user" agent and a
single algorithm agent.  We extend the proposal by introducing four user
sub-types (influencer, regular, lurker, skeptic).  This heterogeneity is
required to reproduce the long-tailed engagement distribution observed in
real social platforms (see e.g. Goel et al., 2016, "The structural virality
of online diffusion") and lets us separate the contribution of network
position from intrinsic user behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# State / type enumerations
# ---------------------------------------------------------------------------


class UserState(IntEnum):
    UNAWARE = 0
    AWARE = 1
    ENGAGED = 2
    FATIGUED = 3


class UserType(IntEnum):
    INFLUENCER = 0
    REGULAR = 1
    LURKER = 2
    SKEPTIC = 3


class AlgorithmState(IntEnum):
    PASSIVE = 0     # no algorithmic intervention (control)
    NORMAL = 1      # mild ranking boost
    BOOSTING = 2    # aggressive amplification (explore / trending tab)
    SUPPRESSING = 3 # active demotion (saturation / spam / safety)


# Mean behavioural parameters per user type.  These are calibrated so that the
# resulting aggregate engagement distribution is heavy-tailed, matching
# stylised facts in social-network research.
TYPE_PARAMS = {
    UserType.INFLUENCER: dict(
        sensitivity_mean=0.55, sensitivity_sd=0.10,
        share_propensity=0.70, fatigue_threshold=6,
    ),
    UserType.REGULAR: dict(
        sensitivity_mean=0.40, sensitivity_sd=0.15,
        share_propensity=0.30, fatigue_threshold=4,
    ),
    UserType.LURKER: dict(
        sensitivity_mean=0.45, sensitivity_sd=0.15,
        share_propensity=0.05, fatigue_threshold=3,
    ),
    UserType.SKEPTIC: dict(
        sensitivity_mean=0.15, sensitivity_sd=0.08,
        share_propensity=0.10, fatigue_threshold=2,
    ),
}

# Population mixture used to assign types.  Roughly matches reported
# active-user distributions on large platforms (5% creators / heavy posters,
# 75% engaged consumers, 15% pure lurkers, 5% contrarians).
DEFAULT_TYPE_MIX = {
    UserType.INFLUENCER: 0.05,
    UserType.REGULAR: 0.75,
    UserType.LURKER: 0.15,
    UserType.SKEPTIC: 0.05,
}


# ---------------------------------------------------------------------------
# User agent
# ---------------------------------------------------------------------------


@dataclass
class UserAgent:
    """An individual social-media user embedded in a follower graph."""

    uid: int
    user_type: UserType
    follower_count: int                  # degree in the graph
    sensitivity: float                   # P(engage | aware & appealing)
    share_propensity: float              # P(share | engaged)
    fatigue_threshold: int               # # exposures before fatigue
    social_influence: float              # normalised log-degree, in [0, 1]
    state: UserState = UserState.UNAWARE
    exposure_count: int = 0              # times the trend hit the feed
    became_aware_at: int = -1
    became_engaged_at: int = -1
    became_fatigued_at: int = -1
    has_shared: bool = False

    # ------------------------------------------------------------------ utils
    def reset(self) -> None:
        """Reset state for a new simulation replicate (keeps static attrs)."""
        self.state = UserState.UNAWARE
        self.exposure_count = 0
        self.became_aware_at = -1
        self.became_engaged_at = -1
        self.became_fatigued_at = -1
        self.has_shared = False


def build_user_population(
    degrees: List[int],
    rng: np.random.Generator,
    type_mix: dict[UserType, float] = DEFAULT_TYPE_MIX,
) -> List[UserAgent]:
    """Create one UserAgent per node in the supplied degree sequence.

    The top-`type_mix[INFLUENCER]` fraction by degree is forced to be of
    type INFLUENCER (network hubs are influencers by definition).  The
    remaining users are assigned to REGULAR / LURKER / SKEPTIC according
    to the residual mixture probabilities.
    """
    n = len(degrees)
    order = np.argsort(degrees)[::-1]    # high degree first
    n_inf = max(1, int(round(type_mix[UserType.INFLUENCER] * n)))
    is_influencer = np.zeros(n, dtype=bool)
    is_influencer[order[:n_inf]] = True

    # residual probabilities for non-influencers
    residual = {
        t: type_mix[t] for t in (UserType.REGULAR, UserType.LURKER, UserType.SKEPTIC)
    }
    z = sum(residual.values())
    types_list = list(residual.keys())
    probs = np.array([residual[t] / z for t in types_list])

    # normalise degree -> social influence score in [0, 1]
    log_deg = np.log1p(np.array(degrees, dtype=float))
    if log_deg.max() > 0:
        influence = log_deg / log_deg.max()
    else:
        influence = log_deg

    users: List[UserAgent] = []
    for i in range(n):
        if is_influencer[i]:
            utype = UserType.INFLUENCER
        else:
            utype = UserType(rng.choice(types_list, p=probs))
        params = TYPE_PARAMS[utype]
        sens = float(np.clip(
            rng.normal(params["sensitivity_mean"], params["sensitivity_sd"]),
            0.01, 0.99,
        ))
        users.append(
            UserAgent(
                uid=i,
                user_type=utype,
                follower_count=int(degrees[i]),
                sensitivity=sens,
                share_propensity=float(params["share_propensity"]),
                fatigue_threshold=int(params["fatigue_threshold"]),
                social_influence=float(influence[i]),
            )
        )
    return users


# ---------------------------------------------------------------------------
# Platform-algorithm agent
# ---------------------------------------------------------------------------


@dataclass
class PlatformAlgorithm:
    """Singleton recommender / distribution agent.

    The algorithm keeps a sliding window of new-engagement signals.  If the
    average rate over the window exceeds `boost_threshold`, it enters the
    BOOSTING state and starts injecting the trending content into the feeds
    of UNAWARE users (modelling the "for-you" / explore page).  After
    `boost_duration_max` ticks or when the rate of FATIGUED users grows
    rapidly, the algorithm transitions to SUPPRESSING -- modelling content
    being demoted once it is judged stale or spammy.
    """

    boost_threshold: float = 0.015        # engagement rate per tick to trigger boost
    suppress_fatigue_ratio: float = 0.35  # fraction of touched users that are FATIGUED
    boost_inject_prob: float = 0.05       # per-tick prob of injecting into an UNAWARE feed
    normal_inject_prob: float = 0.002     # background recommendation rate
    boost_multiplier: float = 1.0         # extra appeal boost during BOOSTING
    boost_duration_max: int = 72          # ticks
    window: int = 24                      # sliding-window size (ticks)
    initial_state: AlgorithmState = AlgorithmState.NORMAL

    state: AlgorithmState = field(init=False)
    _engage_history: List[int] = field(default_factory=list, init=False)
    _time_in_boost: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.state = self.initial_state

    def reset(self) -> None:
        self.state = self.initial_state
        self._engage_history.clear()
        self._time_in_boost = 0

    # ------------------------------------------------------------------ step
    def step(
        self,
        new_engaged: int,
        n_aware: int,
        n_fatigued: int,
        n_total: int,
    ) -> None:
        """Update the algorithm state given the latest tick statistics."""
        # Always record the engagement signal so the window is full of signal
        # whatever state we are in.  This matters for the PASSIVE control
        # condition because it lets us assert that the engagement window is
        # never large enough to trip a boost (we keep PASSIVE locked).
        self._engage_history.append(new_engaged)
        if len(self._engage_history) > self.window:
            self._engage_history.pop(0)

        if self.state == AlgorithmState.PASSIVE:
            return     # control condition: no state change ever

        rate = (sum(self._engage_history) / self.window) / max(1, n_total)
        touched = n_aware + n_fatigued
        fatigued_ratio = (n_fatigued / touched) if touched else 0.0

        if self.state == AlgorithmState.BOOSTING:
            self._time_in_boost += 1
            if (
                self._time_in_boost >= self.boost_duration_max
                or fatigued_ratio >= self.suppress_fatigue_ratio
            ):
                self.state = AlgorithmState.SUPPRESSING
                self._time_in_boost = 0
        elif self.state == AlgorithmState.SUPPRESSING:
            # Stay suppressed for the remainder of the run -- once a platform
            # decides a piece of content is stale, it rarely revives it.
            pass
        else:  # NORMAL
            if rate >= self.boost_threshold:
                self.state = AlgorithmState.BOOSTING
                self._time_in_boost = 0

    # ----------------------------------------------------- injection probability
    def inject_prob(self) -> float:
        """Probability per tick that the algorithm pushes the trend into an
        UNAWARE user's feed."""
        if self.state == AlgorithmState.BOOSTING:
            return self.boost_inject_prob
        if self.state == AlgorithmState.NORMAL:
            return self.normal_inject_prob
        return 0.0   # PASSIVE or SUPPRESSING -> no injection

    def appeal_multiplier(self) -> float:
        """Multiplier on content appeal seen by aware users."""
        if self.state == AlgorithmState.BOOSTING:
            return 1.0 + self.boost_multiplier
        if self.state == AlgorithmState.SUPPRESSING:
            return 0.5
        return 1.0
