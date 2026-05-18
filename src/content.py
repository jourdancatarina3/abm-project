"""The single trending content piece used in each simulation run.

We deliberately model one trend per run (rather than many concurrent trends)
so that the lifecycle metrics -- time-to-viral, peak engagement, lifetime --
are easy to attribute.  Multiple-content competition is left to future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Content:
    intrinsic_appeal: float          # base "quality" of the content, in [0, 1]
    novelty_half_life: float         # ticks until appeal halves due to novelty loss
    seed_tick: int = 0

    # bookkeeping
    peak_engaged: int = 0
    peak_tick: int = -1
    time_to_viral: int = -1          # first tick at which cumulative engaged >= 5% of N

    def current_appeal(self, tick: int) -> float:
        """Exponential novelty decay.

        appeal(t) = intrinsic_appeal * 0.5^((t - seed_tick) / half_life)
        """
        age = max(0, tick - self.seed_tick)
        if self.novelty_half_life <= 0:
            return self.intrinsic_appeal
        return self.intrinsic_appeal * (0.5 ** (age / self.novelty_half_life))
