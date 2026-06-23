# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Macro Regime Detection — Robustness Companion
#
# **Chapter 1 · §1.4 Market Regimes: Change Is the Constant**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Point-in-time (PIT) and out-of-sample counterpart to `macro_regimes.py`. The
# baseline notebook fits regimes in-sample on today's *revised* macro data — which
# quietly uses numbers that were not knowable at the time. This companion loads the
# **first-release** panel alongside the revised one and (in later sections) compares
# in-sample vs walk-forward regime assignment.
#
# ## Learning Objectives
#
# - Load the revised and first-release (PIT) macro panels through the same loader.
# - Process both panels identically so differences come only from data revisions.
# - Set up the base for the revision-impact demo and walk-forward regime fitting.
#
# ## Book Reference
#
# Section 1.4 of Chapter 1. Robustness follow-up to `macro_regimes.py`; the baseline
# notebook and Figure 1.6 are left untouched (this notebook writes to its own
# `output/macro_regimes_robust/`).
#
# ## Structure
#
# 1. **Dual-panel load** — revised + first-release macro panels (this scaffold).
# 2. **Revision-impact demo** — same in-sample fit on each panel (later).
# 3. **Walk-forward regimes** — expanding vs rolling vs in-sample (later).

# %% [markdown]
# ## Imports

# %%
"""Macro Regime Detection — robustness companion (point-in-time + out-of-sample)."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from data import load_macro, load_sp500_index
from utils.paths import get_output_dir
from utils.regimes import walk_forward_regimes
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
# Production defaults (Papermill overrides for testing)
SEED = 42

# %% [markdown]
# ## Configuration

# %%
OUTPUT_DIR = get_output_dir(1, "macro_regimes_robust")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_global_seeds(SEED)

DATE_COL = "timestamp"

# Core 4 indicators (case-insensitive matching), mirroring macro_regimes.py.
CORE_INDICATORS = ["unrate", "dff", "t10y2y", "cpiaucsl"]

# %% [markdown]
# ## Shared Data Prep
#
# `prepare_core_panel` reproduces the core-4 monthly-resample + CPI-YoY + standardize
# pipeline from `macro_regimes.py`. It runs on BOTH panels so any difference between
# the revised and first-release regimes comes from data revisions, not preprocessing.


# %%
def prepare_core_panel(
    macro_raw: pl.DataFrame, extra_yoy: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample core macro indicators to monthly, derive CPI YoY, and z-score.

    Returns (macro_df, macro_scaled): a pandas DataFrame indexed by month-end with
    columns [unrate, dff, t10y2y, cpi_yoy], and its StandardScaler-z-scored array.
    Empty inputs yield an empty DataFrame and array.

    ``extra_yoy`` names additional level series (e.g. ``"payems"``) to append as
    12-month %-change features (``<name>_yoy``), matching the CPI treatment. This
    lets the revision-impact demo bring in a heavily-revised series without
    disturbing the baseline core-4 feature set used elsewhere.
    """
    # group_by_dynamic requires Datetime, not Date
    if macro_raw["timestamp"].dtype == pl.Date:
        macro_raw = macro_raw.with_columns(pl.col("timestamp").cast(pl.Datetime))

    requested = list(CORE_INDICATORS) + list(extra_yoy)
    core_cols: list[str] = []
    col_name_map: dict[str, str] = {}
    for indicator in requested:
        for col in macro_raw.columns:
            if col.lower() == indicator:
                core_cols.append(col)
                col_name_map[col] = indicator
                break

    missing = [ind for ind in requested if ind not in col_name_map.values()]
    if missing:
        raise ValueError(
            f"prepare_core_panel: required indicators missing from panel: {missing}. "
            "Re-run data/macro/download.py to materialize the full series set."
        )

    macro_monthly = (
        macro_raw.select([DATE_COL] + core_cols)
        .sort(DATE_COL)
        .group_by_dynamic(DATE_COL, every="1mo", label="right")
        .agg([pl.col(c).last() for c in core_cols])
        # Forward-fill carries the last observation through release lags. Unlike the
        # baseline we DROP leading nulls rather than backward-filling them: a
        # backward-fill would leak a future observation into an earlier month, which
        # defeats this companion's point-in-time premise. Applied to both panels.
        .fill_null(strategy="forward")
        .drop_nulls(subset=core_cols)
    )

    if macro_monthly.height > 0:
        min_date = macro_monthly[DATE_COL].min()
        if min_date is not None and str(min_date) < "2002-01-01":
            macro_monthly = macro_monthly.filter(pl.col(DATE_COL) >= pl.datetime(2002, 1, 1))

    if macro_monthly.height == 0:
        return pd.DataFrame(), np.array([])

    macro_df = macro_monthly.select(core_cols).to_pandas()
    macro_df.columns = [col_name_map.get(c, c) for c in macro_df.columns]
    macro_df.index = macro_monthly[DATE_COL].to_pandas()

    # CPI level -> YoY %: clustering on the trending level would capture
    # "early vs recent" instead of "inflationary vs non-inflationary".
    macro_df["cpi_yoy"] = macro_df["cpiaucsl"].pct_change(12) * 100
    macro_df = macro_df.drop(columns=["cpiaucsl"])

    # Extra level series -> YoY %, same rationale as CPI (a level like total
    # payrolls trends monotonically; its growth rate is the regime-relevant signal).
    for name in extra_yoy:
        macro_df[f"{name}_yoy"] = macro_df[name].pct_change(12) * 100
        macro_df = macro_df.drop(columns=[name])

    macro_df = macro_df.dropna()
    macro_scaled = StandardScaler().fit_transform(macro_df)
    return macro_df, macro_scaled


# %% [markdown]
# ## Load Both Panels
#
# `load_macro(release="first")` reads the first-release panel materialized by
# `data/macro/download.py`. If it is missing, the loader raises `DataNotFoundError`
# with the download command — run the downloader to materialize it.

# %%
macro_revised = load_macro(release="revised")
macro_first = load_macro(release="first")

revised_df, revised_scaled = prepare_core_panel(macro_revised)
first_df, first_scaled = prepare_core_panel(macro_first)

print(f"Revised panel:       {revised_df.shape[0]} months x {revised_df.shape[1]} features")
print(f"First-release panel: {first_df.shape[0]} months x {first_df.shape[1]} features")
if not revised_df.empty:
    print(f"Revised range:       {revised_df.index.min():%Y-%m} .. {revised_df.index.max():%Y-%m}")
if not first_df.empty:
    print(f"First-release range: {first_df.index.min():%Y-%m} .. {first_df.index.max():%Y-%m}")

# %% [markdown]
# ## 2 · Revision-Impact Demo: Lookahead from Revised Data
#
# The baseline notebook fits regimes on today's *revised* macro panel. But the
# values it clusters on were not all knowable at the time: agencies publish a
# first estimate and revise it for months or years. Fitting on the revised series
# is a form of **lookahead bias** — the model effectively "sees" the final,
# benchmark-revised number while standing in a month when only a noisier first
# print existed.
#
# This section runs the *identical* in-sample GMM on the revised panel and on the
# first-release (point-in-time) panel and asks: **does the regime timeline change?**
# We first measure how much each series revises, then compare regime assignments.

# %% [markdown]
# ### 2.1 · How much does each series revise?
#
# Market and policy observations (Treasury yields, the fed funds rate, VIX) are
# never revised — they were knowable in real time. Survey-based macro aggregates
# (payrolls, GDP, the unemployment rate) are revised, sometimes heavily, and the
# revisions are largest near turning points — exactly where regime boundaries sit.


# %%
def revision_magnitude(
    revised: pl.DataFrame, first: pl.DataFrame, series: list[str]
) -> pd.DataFrame:
    """Monthly mean/max absolute revision (first-release vs revised) per series.

    Resamples both panels to month-end last values, inner-joins on month, and
    reports the absolute difference. ``pct_months_changed`` is the share of
    *observed* overlapping months whose first print differed from the current
    value — the denominator excludes NaN months so lower-frequency series (e.g.
    quarterly ``gdp``, NaN in ~2 of 3 months) share the same basis as the
    mean/max statistics.
    """
    missing = [s for s in series if s not in revised.columns or s not in first.columns]
    if missing:
        raise ValueError(
            f"revision_magnitude: series missing from a panel: {missing}. "
            "Re-run data/macro/download.py to materialize the full series set."
        )

    def monthly(panel: pl.DataFrame) -> pd.DataFrame:
        pdf = panel.to_pandas().set_index("timestamp").sort_index()
        return pdf[series].resample("ME").last()

    rev_m, first_m = monthly(revised), monthly(first)
    common = rev_m.index.intersection(first_m.index)
    diff = (first_m.loc[common] - rev_m.loc[common]).abs()
    out = pd.DataFrame(
        {
            "mean_abs_revision": diff.mean(),
            "max_abs_revision": diff.max(),
            "pct_months_changed": (diff > 1e-9).sum() / diff.notna().sum() * 100,
        }
    )
    return out.sort_values("max_abs_revision", ascending=False)


REVISION_SERIES = ["payems", "gdp", "unrate", "cpiaucsl", "dff", "dgs10", "t10y2y", "vixcls"]
revision_summary = revision_magnitude(macro_revised, macro_first, REVISION_SERIES)
print("Absolute revision, first-release vs revised (monthly):")
print(revision_summary.round(2).to_string())

# %% [markdown]
# `payems` (total nonfarm payrolls) and `gdp` revise by hundreds of thousands of
# jobs / billions of dollars; `unrate` moves by up to ~0.2pp; `cpiaucsl` revises
# slightly. The market/policy series (`dff`, `dgs10`, `t10y2y`, `vixcls`) do not
# revise at all. So of the baseline core-4, only `unrate` carries revision risk —
# and only modestly. We exploit both facts below.

# %% [markdown]
# ### 2.2 · Regime-comparison machinery
#
# Two independent GMM fits assign arbitrary integer ids, so before comparing
# timelines we relabel the first-release clusters to match the revised ones by
# nearest centroid (a minimum-cost assignment in the revised panel's feature
# scale). After alignment, "Regime *k*" means the same thing on both timelines,
# and we borrow the baseline's economic labels so a moved boundary is legible.

# %%
N_REGIMES = 4

# Columns create_regime_labels reads; the regime vocabulary is defined on these
# baseline features regardless of any extra_yoy series added to the fit.
LABEL_FEATURES = ["unrate", "dff", "t10y2y", "cpi_yoy"]


def fit_regimes(scaled: np.ndarray, seed: int = SEED, n_regimes: int = N_REGIMES) -> np.ndarray:
    """Fit the baseline-pinned 4-component GMM and return hard regime labels."""
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        random_state=seed,
        n_init=10,
        reg_covar=1e-6,
    ).fit(scaled)
    return gmm.predict(scaled)


def _scaled_centroids(
    df: pd.DataFrame, labels: np.ndarray, scaler: StandardScaler, n_regimes: int
) -> np.ndarray:
    """Per-cluster mean in a shared (reference) feature scale; empty clusters -> NaN."""
    z = scaler.transform(df.values)
    means = pd.DataFrame(z, index=labels).groupby(level=0).mean().reindex(range(n_regimes))
    return means.values


def align_labels(
    ref_df: pd.DataFrame,
    ref_labels: np.ndarray,
    other_df: pd.DataFrame,
    other_labels: np.ndarray,
    n_regimes: int = N_REGIMES,
) -> np.ndarray:
    """Relabel ``other``'s clusters to match ``ref``'s by nearest centroid.

    Both panels are z-scored in the reference panel's scale so centroids are
    comparable, then a Hungarian assignment pairs each first-release cluster with
    a revised one. Returns ``other_labels`` remapped into the reference id space.
    """
    scaler = StandardScaler().fit(ref_df.values)
    ref_c = _scaled_centroids(ref_df, ref_labels, scaler, n_regimes)
    oth_c = _scaled_centroids(other_df, other_labels, scaler, n_regimes)
    cost = np.linalg.norm(ref_c[:, None, :] - oth_c[None, :, :], axis=2)
    cost = np.nan_to_num(cost, nan=1e6)  # an empty cluster is never a cheap match
    ref_idx, oth_idx = linear_sum_assignment(cost)
    # Identity default: the square cost matrix guarantees a full permutation, but
    # this keeps remap well-defined even if a future caller passes a partial match.
    remap = np.arange(n_regimes)
    for r, o in zip(ref_idx, oth_idx):
        remap[o] = r
    return remap[other_labels]


def create_regime_labels(chars: pd.DataFrame) -> dict[int, str]:
    """Short economic labels from cluster means (verbatim from ``macro_regimes.py``)."""
    labels = {}
    for regime in chars.index:
        c = chars.loc[regime]
        if c["unrate"] > 10:
            labels[regime] = "Crisis"
        elif c["unrate"] > 6 and c["dff"] < 0.5:
            labels[regime] = "Recovery"
        elif c["dff"] > 3 and c["t10y2y"] < 0.5:
            labels[regime] = "Tightening"
        elif c["cpi_yoy"] > 4:
            labels[regime] = "Inflation"
        elif c["unrate"] < 5 and c["dff"] < 2:
            labels[regime] = "Expansion"
        else:
            labels[regime] = "Transition"

    seen = {}
    for r, label in list(labels.items()):
        if label in seen:
            if chars.loc[r, "unrate"] > chars.loc[seen[label], "unrate"]:
                labels[r] = f"{label} (High Unemp.)"
            else:
                labels[seen[label]] = f"{label} (High Unemp.)"
        seen[label] = r
    return labels


def regime_transitions(
    labels: np.ndarray, index: pd.DatetimeIndex, label_map: dict[int, str]
) -> pd.DataFrame:
    """Months where the regime changes, with the regime's economic label."""
    series = pd.Series(labels, index=index)
    rows, prev = [], None
    for date, value in series.items():
        if value != prev:
            rows.append(
                {
                    "date": date.strftime("%Y-%m"),
                    "regime": int(value),
                    "label": label_map.get(int(value), str(value)),
                }
            )
            prev = value
    return pd.DataFrame(rows)


def compare_regimes(
    rev_df: pd.DataFrame, first_df_: pd.DataFrame, title: str, output_name: str
) -> dict[str, object]:
    """Fit, align, plot, and summarize revised-vs-first-release regimes.

    Returns the aligned label arrays, the shared label map, the per-panel
    transition tables, and the months whose regime label differs.
    """
    rev_scaled = StandardScaler().fit_transform(rev_df.values)
    first_scaled_ = StandardScaler().fit_transform(first_df_.values)
    rev_labels = fit_regimes(rev_scaled)
    first_labels = align_labels(rev_df, rev_labels, first_df_, fit_regimes(first_scaled_))

    # Labels are derived from the baseline core-4 means only; any extra_yoy
    # feature (e.g. payems_yoy) is excluded so create_regime_labels' thresholds
    # see exactly the columns they were written for.
    chars = rev_df[LABEL_FEATURES].copy()
    chars["regime"] = rev_labels
    label_map = create_regime_labels(chars.groupby("regime").mean())

    rev_tx = regime_transitions(rev_labels, rev_df.index, label_map)
    first_tx = regime_transitions(first_labels, first_df_.index, label_map)

    common = rev_df.index.intersection(first_df_.index)
    rev_c = pd.Series(rev_labels, index=rev_df.index).reindex(common)
    first_c = pd.Series(first_labels, index=first_df_.index).reindex(common)
    diff_months = common[rev_c.values != first_c.values]

    _plot_regime_comparison(
        rev_c.values,
        first_c.values,
        common,
        label_map,
        diff_months,
        title,
        OUTPUT_DIR / output_name,
    )
    return {
        "rev_labels": rev_labels,
        "first_labels": first_labels,
        "label_map": label_map,
        "rev_tx": rev_tx,
        "first_tx": first_tx,
        "diff_months": diff_months,
        "common": common,
    }


def _plot_regime_comparison(
    rev_labels, first_labels, dates, label_map, diff_months, title, output_path
) -> None:
    """Two-row regime heatmap (revised vs first-release) with a disagreement strip."""
    cmap = plt.get_cmap("tab10", N_REGIMES)
    data = np.vstack([rev_labels, first_labels])

    fig, ax = plt.subplots(figsize=(14, 3.2), layout="tight")
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=N_REGIMES - 1, interpolation="nearest")

    diff_set = set(diff_months)
    for j, date in enumerate(dates):
        if date in diff_set:
            ax.axvline(j, color="black", lw=0.6, ymin=-0.02, ymax=0.0, clip_on=False)

    years = [pd.Timestamp(ts).year for ts in dates]
    ticks = [j for j, y in enumerate(years) if j == 0 or years[j - 1] != y]
    ax.set_xticks(ticks[::2])
    ax.set_xticklabels([str(years[j]) for j in ticks[::2]], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Revised", "First-release"])
    ax.set_xlabel("Year")
    ax.set_title(f"{title}  ({len(diff_months)}/{len(dates)} months reclassified)", fontsize=12)

    handles = [
        Patch(facecolor=cmap(k), label=f"{k}: {label_map.get(k, k)}") for k in range(N_REGIMES)
    ]
    ax.legend(
        handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Regime"
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ### 2.3 · Same fit, baseline core-4 (`unrate, dff, t10y2y, cpi_yoy`)
#
# In the baseline feature set the only revising input is `unrate`. It revises
# modestly (~0.2pp), but those revisions cluster around turning points — and the
# GMM boundaries live there too — so even this small revision moves part of the
# timeline. The reclassified count printed below quantifies it for the current
# data vintage; the figure shows where the two timelines diverge.

# %%
core4 = compare_regimes(revised_df, first_df, "Macro regimes: core-4", "regime_compare_core4.png")
print("Revised transitions:")
print(core4["rev_tx"].to_string(index=False))
print("\nFirst-release transitions:")
print(core4["first_tx"].to_string(index=False))
print(f"\nMonths reclassified: {len(core4['diff_months'])} / {len(core4['common'])}")

# %% [markdown]
# The timelines agree through calm periods and diverge around turning points
# (the 2008 and 2020–21 transitions), where `unrate`'s revisions are largest and a
# 0.2pp move is enough to flip the regime call — including a multi-month shift in
# when the post-COVID Tightening regime begins. Even so, `unrate` is the *only*
# revising core feature, so this still understates the full exposure. To see it we
# add the most heavily-revised macro series.

# %% [markdown]
# ### 2.4 · Add PAYEMS — a heavily-revised series
#
# `payems` is revised every month, by up to ~1M jobs near turning points. Added as
# a YoY %-change feature (the regime-relevant transform, matching CPI), it widens
# the gap further: a larger share of the timeline is reclassified than under
# core-4 when we restrict the model to what was actually knowable.

# %%
revised_aug, _ = prepare_core_panel(macro_revised, extra_yoy=("payems",))
first_aug, _ = prepare_core_panel(macro_first, extra_yoy=("payems",))
aug = compare_regimes(
    revised_aug, first_aug, "Macro regimes: core-4 + PAYEMS YoY", "regime_compare_payems.png"
)
print("Revised transitions:")
print(aug["rev_tx"].to_string(index=False))
print("\nFirst-release transitions:")
print(aug["first_tx"].to_string(index=False))
print(f"\nMonths reclassified: {len(aug['diff_months'])} / {len(aug['common'])}")

# %% [markdown]
# ### 2.5 · Why this is lookahead bias
#
# The revised and first-release panels differ **only in data vintage** — identical
# preprocessing, identical GMM, identical seed. Yet the regime timeline shifts:
# already with the baseline core-4 (where `unrate` is the only revising input),
# and more so once `payems` — the most heavily revised macro series — enters the
# feature set. The divergence concentrates around recession turning points, where
# revisions are largest, and a meaningful share of months get a *different* regime
# call (see the reclassified counts above).
#
# This is lookahead bias in its purest form. A model fit on revised data assigns
# 2009 or 2020 to the regime justified by the **benchmark-revised** payroll and
# unemployment figures — numbers that did not exist until months or years later.
# A strategy that conditioned on those regimes in a backtest would have "known"
# the eventual revision in real time, inflating its measured edge. The
# point-in-time panel removes that knowledge: every regime label here is one the
# model could have produced from data available *that month*. The lesson is the
# chapter's — change is the constant — sharpened: so is revision, and a regime
# model is only as honest as the vintage of the data it is fit on.

# %% [markdown]
# ## 3 · Walk-Forward (Out-of-Sample) Regimes
#
# The baseline fits ONE mixture on the whole 2003–2026 panel and reads regime
# labels back over the same span — so every boundary was placed with knowledge of
# the entire history, including the future. That is fine for *describing* the past
# but dishonest as a signal: in production you only ever have the past.
#
# This section re-fits the regime model **out-of-sample** with `utils.regimes`,
# assigning each month using only data available *before* it, under two windowing
# schemes, and stacks them against the in-sample timeline:
#
# 1. **In-sample** — the baseline whole-panel fit (uses the future; the benchmark).
# 2. **Expanding** — train on all months before *t*, predict *t*.
# 3. **Rolling** — train on the last `rolling_window` months before *t*, predict *t*.
#
# A fresh `StandardScaler` is fit per step on in-window data only (no scaling
# look-ahead), and per-step clusters are centroid-matched so regime ids stay
# stable. The first `min_train_samples` months have no out-of-sample assignment.

# %% [markdown]
# ### 3.1 · Fit the three schemes
#
# Each scheme's integer clusters are turned into the baseline's economic labels via
# `create_regime_labels`, applied to that scheme's own cluster means — so the three
# timelines are comparable by *meaning*, not by arbitrary cluster id.


# %%
def semantic_labels(labels: np.ndarray, panel: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Map per-month integer regimes to economic labels via their cluster means.

    Months marked ``-1`` (the walk-forward min-window guard) become ``None``.
    Returns ``(semantic_array, {int: label})`` where the label map is derived from
    this scheme's own assigned-month means.
    """
    assigned = labels != -1
    chars = panel.loc[assigned, LABEL_FEATURES].copy()
    chars["regime"] = labels[assigned]
    label_map = create_regime_labels(chars.groupby("regime").mean())
    semantic = np.array([label_map.get(int(x)) if x != -1 else None for x in labels], dtype=object)
    return semantic, label_map


insample_labels = fit_regimes(revised_scaled)
expanding = walk_forward_regimes(revised_df, scheme="expanding", random_state=SEED)
rolling = walk_forward_regimes(revised_df, scheme="rolling", rolling_window=60, random_state=SEED)

scheme_labels = {
    "In-sample": insample_labels,
    "Expanding": expanding.labels,
    "Rolling": rolling.labels,
}
scheme_semantic = {name: semantic_labels(lab, revised_df)[0] for name, lab in scheme_labels.items()}

for name, sem in scheme_semantic.items():
    n_assigned = int(sum(s is not None for s in sem))
    n_boundaries = int(
        sum(
            sem[i] != sem[i - 1]
            for i in range(1, len(sem))
            if sem[i] is not None and sem[i - 1] is not None
        )
    )
    print(
        f"{name:10}: {n_assigned:3d}/{len(sem)} months assigned, {n_boundaries:2d} regime changes"
    )

# %% [markdown]
# ### 3.2 · Three stacked timelines
#
# Rows share a colour key by *base* regime (the `(High Unemp.)` de-dup suffix is
# folded into its base). Blank cells are months with no out-of-sample assignment.

# %%
# Vol-ordered canonical regimes give every timeline a shared, stable colour key.
CANONICAL_REGIMES = ["Expansion", "Recovery", "Tightening", "Inflation", "Transition", "Crisis"]


def _base_regime(label: str | None) -> str | None:
    """Strip the ``(High Unemp.)`` de-dup suffix to the economic base regime."""
    return None if label is None else label.split(" (")[0]


def plot_three_timelines(scheme_sems: dict[str, np.ndarray], dates, output_path) -> None:
    """Stack the schemes' regime timelines, coloured by shared base regime."""
    cmap = plt.get_cmap("tab10", len(CANONICAL_REGIMES))
    cmap.set_bad("white")
    present: set[int] = set()

    fig, axes = plt.subplots(len(scheme_sems), 1, figsize=(14, 5.5), sharex=True, layout="tight")
    for ax, (name, sem) in zip(axes, scheme_sems.items()):
        row = np.full(len(sem), np.nan)
        for j, label in enumerate(sem):
            base = _base_regime(label)
            if base is not None:
                if base not in CANONICAL_REGIMES:
                    raise ValueError(
                        f"base regime {base!r} is not in CANONICAL_REGIMES "
                        f"{CANONICAL_REGIMES} — add it to keep the colour key in sync"
                    )
                idx = CANONICAL_REGIMES.index(base)
                row[j] = idx
                present.add(idx)
        ax.imshow(
            np.ma.masked_invalid(row[None, :]),
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=len(CANONICAL_REGIMES) - 1,
            interpolation="nearest",
        )
        ax.set_yticks([0])
        ax.set_yticklabels([name])

    years = [pd.Timestamp(ts).year for ts in dates]
    ticks = [j for j, y in enumerate(years) if j == 0 or years[j - 1] != y]
    axes[-1].set_xticks(ticks[::2])
    axes[-1].set_xticklabels([str(years[j]) for j in ticks[::2]], fontsize=9)
    axes[-1].set_xlabel("Year")
    axes[0].set_title("Macro regimes: in-sample vs out-of-sample (expanding, rolling)", fontsize=12)

    handles = [Patch(facecolor=cmap(k), label=CANONICAL_REGIMES[k]) for k in sorted(present)]
    axes[0].legend(
        handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Base regime"
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show()


plot_three_timelines(scheme_semantic, revised_df.index, OUTPUT_DIR / "regime_three_timelines.png")

# %% [markdown]
# ### 3.3 · S&P 500 validation per scheme
#
# The baseline's test of a regime model is whether its regimes separate equity
# volatility and drawdown. We compute the S&P 500 monthly volatility (trailing
# 12-month, annualized) and running drawdown once on the macro calendar, then group
# them by each scheme's regime labels. The S&P series starts 2006, so the
# in-sample timeline's early-2000s months have no equity validation; the
# walk-forward schemes only begin emitting labels around then anyway.

# %%
sp500_raw = load_sp500_index().to_pandas()
sp500_raw["timestamp"] = pd.to_datetime(sp500_raw["timestamp"])
sp500_raw = sp500_raw.set_index("timestamp")
sp500_monthly = sp500_raw["close"].resample("ME").last().to_frame()
sp500_monthly["returns"] = sp500_monthly["close"].pct_change()

sp500_macro = sp500_monthly.reindex(
    revised_df.index, method="nearest", tolerance=pd.Timedelta("5D")
)
sp500_macro["peak"] = sp500_macro["close"].cummax()
sp500_macro["drawdown"] = (sp500_macro["close"] - sp500_macro["peak"]) / sp500_macro["peak"]
sp500_macro["volatility"] = sp500_macro["returns"].rolling(12).std() * np.sqrt(12)


def regime_sp500_stats(semantic: np.ndarray) -> pd.DataFrame:
    """Per-regime S&P 500 vol/drawdown stats, vol-sorted (mirrors the baseline)."""
    frame = sp500_macro.copy()
    frame["regime_label"] = semantic
    frame = frame[frame["regime_label"].notna()]
    stats = frame.groupby("regime_label").agg(
        {"returns": ["mean", "std", "count"], "drawdown": "min", "volatility": "mean"}
    )
    stats.columns = ["mean_ret", "std_ret", "months", "max_dd", "avg_vol"]
    stats["annual_vol_pct"] = stats["std_ret"] * np.sqrt(12) * 100
    stats["max_dd_pct"] = -stats["max_dd"] * 100
    return stats.sort_values("annual_vol_pct")[["months", "annual_vol_pct", "max_dd_pct"]]


scheme_stats = {name: regime_sp500_stats(sem) for name, sem in scheme_semantic.items()}
for name, stats in scheme_stats.items():
    print(f"\n=== {name} — S&P 500 by regime (vol-sorted) ===")
    print(stats.round(1).to_string())

# %% [markdown]
# ### 3.4 · Boundary stability — what we learn
#
# **In-sample** produces the cleanest timeline: four well-separated regimes
# (Expansion, Recovery, Tightening, Crisis) with decisive boundaries, including a
# clean single-month COVID Crisis spike. That tidiness is exactly the look-ahead
# the companion critiques — the model placed every boundary already knowing how
# each episode resolved.
#
# **Expanding** tracks the in-sample story but pays the honesty tax: it can only
# react *after* seeing the data, so boundaries arrive late and flicker far more,
# and early clusters are not yet separated enough to earn the clean economic names
# — several months fall into the catch-all "Transition" regime. As its window
# grows it stabilizes and its later boundaries approach the in-sample ones.
#
# **Rolling** is the least stable: a fixed window forgets older regimes, so cluster
# definitions drift and rare regimes can vanish entirely — note it never recovers a
# distinct "Crisis" or "Expansion" label. Boundary instability here is a *finding*,
# not a bug: it is what a strictly-windowed, memory-limited estimator actually does.
#
# The S&P validation shows what survives. In-sample keeps the textbook ordering —
# Expansion the lowest-volatility, shallowest-drawdown regime; Crisis/Tightening the
# highest. Out-of-sample, the labels blur but the *association* largely holds: the
# deepest drawdowns still attach to the high-unemployment clusters, and volatility
# still rises with regime severity, even if the spread compresses. The robust
# read: macro regimes track equity volatility more than returns, and that signal
# survives honest out-of-sample fitting — but the crisp four-regime taxonomy does
# not. It is partly an artifact of fitting on the whole history at once.
