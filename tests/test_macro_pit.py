"""Point-in-time (first-release) macro panel tests.

Pins the lookahead-free guarantees the robust-regime notebook relies on when it
fits regimes on ``load_macro(release="first")`` instead of today's fully-revised
values:

- First-release differs from revised on a heavily-revised series (PAYEMS, GDP):
  the panel does NOT silently store revised numbers, so no future revision is
  baked into a historical row.
- A never-revised daily series (DGS10, DFF, ...) is bit-identical between the two
  releases: the first-release pipeline perturbs ONLY data that actually revises,
  which is the complement of a leak.
- The loader is causal under an ``end_date`` cutoff: truncating the panel never
  alters the first-release values of the rows that remain — a past row cannot
  depend on observations that arrive after it.
- The walk-forward regime engine is reproducible under a fixed seed when run on
  the real first-release panel (real-data companion to the synthetic determinism
  test in ``test_regimes.py``).

The first-release panel (``fred_macro_firstrelease.parquet``) is produced by
``data/macro/download.py`` and is absent in a bare checkout / CI without the
data, so the whole module skips cleanly when it is missing.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from data import load_macro
from utils import ML4T_DATA_PATH

# Resolve the first-release parquet from the data path bound at import time
# (the same symbol every macro loader uses) and skip the module when it is
# absent — the file is built by data/macro/download.py and is not committed.
_FIRSTRELEASE = ML4T_DATA_PATH / "macro" / "fred_macro_firstrelease.parquet"

pytestmark = pytest.mark.skipif(
    not _FIRSTRELEASE.exists(),
    reason=(
        "fred_macro_firstrelease.parquet not present "
        "(run data/macro/download.py to materialize the first-release panel)"
    ),
)

# Heavily-revised series (level/flow econ data): first print != current revised.
REVISED_SERIES = ["payems", "gdp"]
# Daily rate/market series FRED never revises: first release IS the final value,
# so first-release == revised must hold bit-for-bit.
NEVER_REVISED_SERIES = ["dgs10", "dff", "vixcls", "t10y2y"]


def _joined(series: list[str]) -> pl.DataFrame:
    """Inner-join the revised and first-release panels on ``timestamp``.

    First-release columns are suffixed ``_fr``.
    """
    revised = load_macro(series=series, release="revised")
    first = load_macro(series=series, release="first")
    joined = revised.join(first, on="timestamp", how="inner", suffix="_fr").sort("timestamp")
    # _read_macro_panel silently drops a requested series absent from the panel,
    # so on a partial-panel checkout the join would lack these columns. Skip with
    # a clear reason rather than letting a later lookup raise an opaque KeyError.
    for col in series:
        if col not in joined.columns:
            pytest.skip(f"series {col!r} not present in the macro panel")
    return joined


def _aligned_pairs(joined: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(revised, first_release)`` value arrays for ``col``, dropping any
    row where either side is null so the comparison is well defined."""
    revised = joined[col].to_numpy()
    first = joined[f"{col}_fr"].to_numpy()
    mask = ~(np.isnan(revised) | np.isnan(first))
    return revised[mask], first[mask]


# -----------------------------------------------------------------------------
# Difference: first-release is NOT the revised panel on revised series
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("series", REVISED_SERIES)
def test_first_release_differs_from_revised(series: str) -> None:
    """A heavily-revised series must diverge between releases — proof that future
    revisions are excluded from the first-release panel (no leak)."""
    joined = _joined([series])
    revised, first = _aligned_pairs(joined, series)
    assert revised.size > 0, f"no overlapping {series} observations to compare"

    # Collapse the daily forward-fill plateaus to the distinct monthly/quarterly
    # release levels, then require a substantial fraction to differ materially.
    # Stated as a loose lower bound: revisions only accumulate over time, never
    # vanish, so this holds as the revised panel drifts on each re-download.
    changes = np.abs(np.diff(revised)) > 1e-6
    revised_levels = np.concatenate(([True], changes))  # keep first + each change
    diff = np.abs(revised[revised_levels] - first[revised_levels])
    frac_revised = float((diff > 1.0).mean())
    assert frac_revised >= 0.5, (
        f"{series}: only {frac_revised:.0%} of distinct levels differ between "
        "first-release and revised — first-release may be storing revised values"
    )


# -----------------------------------------------------------------------------
# Faithfulness: unrevised series are untouched (complement of a leak)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("series", NEVER_REVISED_SERIES)
def test_never_revised_series_identical_across_releases(series: str) -> None:
    """A series FRED never revises must be bit-identical between releases. This is
    the specificity control: it proves the difference seen on PAYEMS/GDP is a real
    revision signal, not a pipeline artifact that perturbs every column."""
    joined = _joined([series])
    revised, first = _aligned_pairs(joined, series)
    assert revised.size > 0, f"no overlapping {series} observations to compare"
    np.testing.assert_array_equal(revised, first)


# -----------------------------------------------------------------------------
# No future leak: the first-release load is causal under a date cutoff
# -----------------------------------------------------------------------------


def test_first_release_load_is_causal_under_cutoff() -> None:
    """No future revision leaks into a past row: the first-release value attached
    to a date ``t`` must not depend on observations after ``t``.

    Loading with an ``end_date`` cutoff must return exactly the rows ``<= cutoff``
    from the full-history load, unchanged. A pipeline that re-normalized or
    back-filled using the whole window (a lookahead bug) would fail this.
    """
    series = REVISED_SERIES[0]
    full = load_macro(series=[series], release="first").sort("timestamp")
    cutoff = full["timestamp"][len(full) // 2]

    # load_macro does not sort, so sort before comparing — pl.equals is row-order
    # sensitive and we are testing values, not the parquet's stored order.
    truncated = load_macro(series=[series], release="first", end_date=str(cutoff)).sort("timestamp")
    expected = full.filter(pl.col("timestamp") <= cutoff)

    assert truncated.equals(expected), (
        "first-release values for retained dates changed when later dates were "
        "excluded — the load is not point-in-time causal"
    )


# -----------------------------------------------------------------------------
# Determinism: walk-forward regimes reproducible on the real first-release panel
# -----------------------------------------------------------------------------


def test_walk_forward_deterministic_on_first_release_panel() -> None:
    """Walk-forward labels are reproducible under a fixed seed when fit on the real
    first-release macro panel — a real-data companion to the synthetic determinism
    test in ``test_regimes.py``."""
    walk_forward_regimes = pytest.importorskip("utils.regimes").walk_forward_regimes

    cols = ["unrate", "payems", "indpro", "cpiaucsl"]
    panel = load_macro(series=cols, release="first").sort("timestamp")
    # Collapse the daily forward-fill to one row per month, then drop any months
    # missing a column so the feature matrix is dense.
    monthly = (
        panel.group_by_dynamic("timestamp", every="1mo")
        .agg([pl.col(c).last() for c in cols])
        .drop_nulls()
    )
    features = monthly.select(cols).to_numpy()
    assert features.shape[0] > 40, "not enough monthly observations to walk forward"

    kw = dict(scheme="expanding", min_train_samples=40, n_components=4, random_state=42)
    first_run = walk_forward_regimes(features, **kw)
    second_run = walk_forward_regimes(features, **kw)

    np.testing.assert_array_equal(first_run.labels, second_run.labels)
    np.testing.assert_allclose(first_run.probabilities, second_run.probabilities, equal_nan=True)
    np.testing.assert_array_equal(first_run.flagged, second_run.flagged)
    assert np.any(~first_run.flagged), "expected at least some months to be assigned"
