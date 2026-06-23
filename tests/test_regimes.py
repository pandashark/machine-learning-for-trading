"""Tests for utils/regimes.py — walk-forward (out-of-sample) regime fitting.

Pins the correctness properties the robust-regime notebook depends on:

- ``match_clusters_to_reference`` recovers the true permutation between a
  reference centroid set and a relabeled copy (stable regime identity).
- ``walk_forward_regimes`` is deterministic under a fixed ``random_state``.
- The min-window guard skips early months (label ``-1``, ``flagged``, NaN probs)
  and assigns later ones.
- Regime identity is stable across months: a given underlying blob keeps the
  same emitted label even though raw GMM indices reshuffle per re-fit.
- There is no scaling/look-ahead leakage: perturbing a FUTURE month cannot
  change an earlier month's assignment under the expanding scheme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.regimes import match_clusters_to_reference, walk_forward_regimes


def _make_blobs(
    n_months: int, centers: np.ndarray, noise: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cycle through ``centers`` month by month with small Gaussian noise.

    Returns ``(X, true_blob)`` where ``true_blob[i] = i % len(centers)``.
    """
    rng = np.random.default_rng(seed)
    true_blob = np.arange(n_months) % len(centers)
    X = centers[true_blob] + rng.normal(0.0, noise, size=(n_months, centers.shape[1]))
    return X, true_blob


CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])


# -----------------------------------------------------------------------------
# Pure: match_clusters_to_reference
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("perm", [[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 1, 0]])
def test_match_recovers_permutation(perm: list[int]) -> None:
    reference = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])
    perm_arr = np.array(perm)
    centroids = reference[perm_arr]  # component i sits on reference[perm[i]]
    mapping = match_clusters_to_reference(centroids, reference)
    np.testing.assert_array_equal(mapping, perm_arr)


def test_match_is_a_permutation_even_with_noise() -> None:
    reference = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])
    centroids = reference[[2, 0, 1]] + np.array([[0.1, -0.1], [0.0, 0.2], [-0.1, 0.0]])
    mapping = match_clusters_to_reference(centroids, reference)
    assert sorted(mapping.tolist()) == [0, 1, 2]  # collision-free
    np.testing.assert_array_equal(mapping, [2, 0, 1])


def test_match_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        match_clusters_to_reference(np.zeros((3, 2)), np.zeros((4, 2)))


# -----------------------------------------------------------------------------
# walk_forward_regimes — determinism
# -----------------------------------------------------------------------------


def test_deterministic_under_fixed_seed() -> None:
    X, _ = _make_blobs(90, CENTERS, seed=1)
    kw = dict(scheme="expanding", min_train_samples=12, n_components=3, random_state=42)
    a = walk_forward_regimes(X, **kw)
    b = walk_forward_regimes(X, **kw)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.probabilities, b.probabilities, equal_nan=True)
    np.testing.assert_array_equal(a.flagged, b.flagged)


# -----------------------------------------------------------------------------
# walk_forward_regimes — min-window guard
# -----------------------------------------------------------------------------


def test_min_window_guard_skips_early_months() -> None:
    X, _ = _make_blobs(60, CENTERS, seed=2)
    min_train = 40
    res = walk_forward_regimes(
        X, scheme="expanding", min_train_samples=min_train, n_components=3, random_state=42
    )
    # Expanding: month t trains on rows [0, t); fewer than min_train -> skipped.
    early = np.arange(min_train)
    late = np.arange(min_train, 60)
    assert np.all(res.labels[early] == -1)
    assert np.all(res.flagged[early])
    assert np.all(np.isnan(res.probabilities[early]))
    assert np.all(res.labels[late] >= 0)
    assert not np.any(res.flagged[late])
    # assigned probability rows are proper distributions
    np.testing.assert_allclose(res.probabilities[late].sum(axis=1), 1.0, atol=1e-6)


def test_all_skipped_when_panel_too_short() -> None:
    X, _ = _make_blobs(10, CENTERS, seed=3)
    res = walk_forward_regimes(X, min_train_samples=40, n_components=3, random_state=42)
    assert np.all(res.labels == -1)
    assert np.all(res.flagged)
    assert res.reference_centroids.shape == (3, 2)
    assert np.all(np.isnan(res.reference_centroids))


# -----------------------------------------------------------------------------
# walk_forward_regimes — stable regime identity
# -----------------------------------------------------------------------------


def test_regime_identity_stable_across_months() -> None:
    X, true_blob = _make_blobs(96, CENTERS, noise=0.2, seed=4)
    res = walk_forward_regimes(
        X, scheme="expanding", min_train_samples=12, n_components=3, random_state=42
    )
    assigned = ~res.flagged
    # Each underlying blob must map to exactly one emitted label across all
    # assigned months, and the three blobs must occupy three distinct labels.
    blob_to_labels: dict[int, set[int]] = {}
    for blob in range(len(CENTERS)):
        mask = assigned & (true_blob == blob)
        blob_to_labels[blob] = set(res.labels[mask].tolist())
        assert len(blob_to_labels[blob]) == 1, f"blob {blob} got labels {blob_to_labels[blob]}"
    distinct = {next(iter(s)) for s in blob_to_labels.values()}
    assert len(distinct) == 3


# -----------------------------------------------------------------------------
# walk_forward_regimes — no look-ahead in scaling
# -----------------------------------------------------------------------------


def test_outlier_changes_only_windows_that_contain_it() -> None:
    # Pin the causal boundary: perturbing row k must leave months whose training
    # window ends at/before k untouched, while changing a later month whose
    # expanding window [0, t) includes k. A whole-panel scaler/fit (look-ahead)
    # would corrupt the earlier month too, failing the first assertion.
    X, _ = _make_blobs(96, CENTERS, noise=0.2, seed=5)
    kw = dict(scheme="expanding", min_train_samples=12, n_components=3, random_state=42)
    base = walk_forward_regimes(X, **kw)

    k = 70
    poisoned = X.copy()
    poisoned[k] = [1e6, 1e6]  # extreme value in an interior month
    after = walk_forward_regimes(poisoned, **kw)

    before = 65  # window [0, 65) excludes row k -> must be identical
    assert not base.flagged[before]
    assert base.labels[before] == after.labels[before]
    np.testing.assert_allclose(base.probabilities[before], after.probabilities[before])

    # A later month whose window [0, t) contains k is corrupted by the outlier;
    # at least one such month's assignment must change (proves the window is used).
    later = slice(k + 1, 96)
    assert np.any(base.labels[later] != after.labels[later])


# -----------------------------------------------------------------------------
# walk_forward_regimes — schemes / wiring
# -----------------------------------------------------------------------------


def test_rolling_scheme_runs_and_uses_datetime_index() -> None:
    X, _ = _make_blobs(96, CENTERS, noise=0.2, seed=6)
    dates = pd.date_range("2003-01-31", periods=96, freq="ME")
    df = pd.DataFrame(X, index=dates, columns=["f0", "f1"])
    res = walk_forward_regimes(
        df, scheme="rolling", rolling_window=24, min_train_samples=24, n_components=3
    )
    assert res.scheme == "rolling"
    assert res.dates.equals(dates)
    # rolling window 24 with min 24 -> first 24 months skipped, rest assigned
    assert np.all(res.labels[:24] == -1)
    assert np.all(res.labels[24:] >= 0)


def test_invalid_scheme_raises() -> None:
    X, _ = _make_blobs(50, CENTERS, seed=7)
    with pytest.raises(ValueError, match="scheme must be"):
        walk_forward_regimes(X, scheme="bogus")
