"""Statistical analysis of simulation output.

Tests performed (all reported in the paper):

    - Levene's test for variance homogeneity (assumption check for ANOVA)
    - One-way ANOVA across the six scenarios for every outcome metric
    - Kruskal--Wallis as a non-parametric robustness check
    - Tukey HSD post-hoc tests for pairwise scenario contrasts
    - Cohen's d effect sizes for the key contrast (S4 vs S1)
    - 2x2 interaction analysis (Engagement x AlgorithmBoost) -> Two-way
      factorial ANOVA on viral_reach and peak_engaged.  This is what
      formally tests the alternative hypothesis ("virality requires BOTH
      user engagement AND algorithm boosting").
    - Pearson + Spearman correlation between summary metrics
    - Kolmogorov--Smirnov test that the per-user exposure-count
      distribution is heavy-tailed (consistent with empirical virality)
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


METRICS = [
    "viral_reach",
    "peak_engaged",
    "time_to_viral",
    "lifetime",
    "decay_rate",
    "auc_engaged",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for two independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    s2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    s = np.sqrt(s2) if s2 > 0 else np.nan
    if not np.isfinite(s) or s == 0:
        return np.nan
    return float((a.mean() - b.mean()) / s)


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------


def descriptive_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario mean ± std for every outcome metric."""
    rows = []
    for sc, g in summary.groupby("scenario"):
        row = {"scenario": sc, "n": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Omnibus ANOVA + non-parametric backup
# ---------------------------------------------------------------------------


def omnibus_tests(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = sorted(summary["scenario"].unique())
    groups = {sc: summary.loc[summary.scenario == sc] for sc in scenarios}
    for m in METRICS:
        samples = [groups[sc][m].to_numpy() for sc in scenarios]
        # variance homogeneity
        try:
            lev_stat, lev_p = stats.levene(*samples)
        except Exception:
            lev_stat, lev_p = (np.nan, np.nan)
        # one-way ANOVA
        f_stat, f_p = stats.f_oneway(*samples)
        # kruskal-wallis as a robustness check
        h_stat, h_p = stats.kruskal(*samples)
        # eta-squared effect size for ANOVA
        all_vals = np.concatenate(samples)
        ss_between = sum(len(s) * (s.mean() - all_vals.mean()) ** 2 for s in samples)
        ss_total = ((all_vals - all_vals.mean()) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan
        rows.append(dict(
            metric=m,
            levene_stat=lev_stat, levene_p=lev_p,
            anova_F=f_stat, anova_p=f_p,
            eta_squared=eta2,
            kruskal_H=h_stat, kruskal_p=h_p,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tukey HSD post-hoc per metric
# ---------------------------------------------------------------------------


def tukey_tables(summary: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for m in METRICS:
        tk = pairwise_tukeyhsd(summary[m].to_numpy(), summary["scenario"].to_numpy())
        df = pd.DataFrame(data=tk._results_table.data[1:],
                          columns=tk._results_table.data[0])
        df["metric"] = m
        out[m] = df
    return out


# ---------------------------------------------------------------------------
# Cohen's d matrix for one chosen metric
# ---------------------------------------------------------------------------


def cohen_d_matrix(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    scs = sorted(summary["scenario"].unique())
    mat = pd.DataFrame(index=scs, columns=scs, dtype=float)
    for a in scs:
        va = summary.loc[summary.scenario == a, metric].to_numpy()
        for b in scs:
            vb = summary.loc[summary.scenario == b, metric].to_numpy()
            mat.loc[a, b] = cohen_d(va, vb)
    return mat


# ---------------------------------------------------------------------------
# 2x2 interaction analysis (Engagement x AlgorithmBoost) on viral_reach,
# peak_engaged, lifetime.
# ---------------------------------------------------------------------------


# Map each scenario to a (engagement_level, algo_level) factor pair.
# Only S0/S1/S2/S3/S4 form a clean 2x2 cell layout; S5 isolates seeding.
FACTOR_MAP = {
    "S0_NoAlgorithm":          dict(engagement="low",  algo="off"),
    "S1_Baseline":             dict(engagement="low",  algo="normal"),
    "S2_AlgoBoostOnly":        dict(engagement="low",  algo="boost"),
    "S3_HighEngagementOnly":   dict(engagement="high", algo="off"),
    "S4_BoostPlusEngagement":  dict(engagement="high", algo="boost"),
}


def two_way_anova(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Two-way factorial ANOVA: metric ~ Engagement * AlgoBoost.

    Uses only the four corner cells (S0, S2, S3, S4) so the design is balanced
    -- this is the rigorous test of the alternative hypothesis.
    """
    keep = ["S0_NoAlgorithm", "S2_AlgoBoostOnly",
            "S3_HighEngagementOnly", "S4_BoostPlusEngagement"]
    df = summary[summary.scenario.isin(keep)].copy()
    df["engagement"] = df["scenario"].map(lambda s: FACTOR_MAP[s]["engagement"])
    df["algo"] = df["scenario"].map(lambda s: FACTOR_MAP[s]["algo"])
    # collapse algo: "off" stays off, "boost" becomes on (no NORMAL cell)
    model = ols(f"{metric} ~ C(engagement) * C(algo)", data=df).fit()
    table = anova_lm(model, typ=2)
    table["metric"] = metric
    return table.reset_index().rename(columns={"index": "source"})


# ---------------------------------------------------------------------------
# Correlations across metrics within a scenario
# ---------------------------------------------------------------------------


def metric_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc, g in summary.groupby("scenario"):
        for i, m1 in enumerate(METRICS):
            for m2 in METRICS[i + 1:]:
                # filter out infinities/missing
                a = g[m1].to_numpy()
                b = g[m2].to_numpy()
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 5:
                    continue
                r_p, p_p = stats.pearsonr(a[ok], b[ok])
                r_s, p_s = stats.spearmanr(a[ok], b[ok])
                rows.append(dict(
                    scenario=sc, m1=m1, m2=m2,
                    pearson_r=r_p, pearson_p=p_p,
                    spearman_r=r_s, spearman_p=p_s,
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distribution-shape tests on per-user exposure counts
# ---------------------------------------------------------------------------


def exposure_distribution_tests(final_states: pd.DataFrame) -> pd.DataFrame:
    """Quantify how heavy-tailed the per-user exposure distribution is in
    each scenario.  A heavy tail (high skew, KS away from normal) is a key
    empirical fingerprint of viral spread."""
    rows = []
    for sc, g in final_states.groupby("scenario"):
        x = g["exposure_count"].to_numpy()
        x = x[x > 0]
        if len(x) < 10:
            continue
        # KS test vs. normal (null: same as normal -> rejection means non-normal)
        z = (x - x.mean()) / (x.std(ddof=1) + 1e-9)
        ks_stat, ks_p = stats.kstest(z, "norm")
        # Gini coefficient
        xs = np.sort(x)
        n = len(xs)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum()))
        rows.append(dict(
            scenario=sc,
            mean_exposures=float(x.mean()),
            std_exposures=float(x.std(ddof=1)),
            skew=float(stats.skew(x)),
            kurtosis=float(stats.kurtosis(x)),
            ks_vs_normal=ks_stat,
            ks_p=ks_p,
            gini=gini,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all(data_dir: str = "data", out_dir: str = "data/processed") -> dict:
    summary = pd.read_csv(f"{data_dir}/processed/summary.csv")
    final_states = pd.read_csv(f"{data_dir}/raw/final_states.csv.gz")

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    results["descriptive"] = descriptive_table(summary)
    results["omnibus"] = omnibus_tests(summary)
    tukey = tukey_tables(summary)
    results["tukey"] = pd.concat(tukey.values(), ignore_index=True)
    results["cohen_d_viral_reach"] = cohen_d_matrix(summary, "viral_reach")
    results["cohen_d_peak_engaged"] = cohen_d_matrix(summary, "peak_engaged")
    results["two_way_viral_reach"] = two_way_anova(summary, "viral_reach")
    results["two_way_peak_engaged"] = two_way_anova(summary, "peak_engaged")
    results["two_way_lifetime"] = two_way_anova(summary, "lifetime")
    results["correlations"] = metric_correlations(summary)
    results["exposure_distribution"] = exposure_distribution_tests(final_states)

    # save everything
    for name, df in results.items():
        df.to_csv(f"{out_dir}/stats_{name}.csv", index=True)

    # Hypothesis decision: based on omnibus p-values and 2x2 interaction term
    omni = results["omnibus"]
    twa = results["two_way_viral_reach"]
    inter_p = twa.loc[twa["source"] == "C(engagement):C(algo)", "PR(>F)"].iloc[0]
    decision = {
        "reject_H0_omnibus":     bool((omni["anova_p"] < 0.05).all()),
        "interaction_p_value":   float(inter_p),
        "reject_H0_interaction": bool(inter_p < 0.05),
    }
    pd.Series(decision).to_csv(f"{out_dir}/stats_hypothesis_decision.csv")
    results["hypothesis_decision"] = decision

    return results


if __name__ == "__main__":
    out = run_all()
    print("=== Descriptive table ===")
    print(out["descriptive"].round(3).to_string(index=False))
    print("\n=== Omnibus ANOVA / Kruskal--Wallis ===")
    print(out["omnibus"].round(4).to_string(index=False))
    print("\n=== Two-way ANOVA on viral_reach ===")
    print(out["two_way_viral_reach"].round(4).to_string(index=False))
    print("\n=== Two-way ANOVA on peak_engaged ===")
    print(out["two_way_peak_engaged"].round(4).to_string(index=False))
    print("\n=== Hypothesis decision ===")
    for k, v in out["hypothesis_decision"].items():
        print(f"  {k}: {v}")
