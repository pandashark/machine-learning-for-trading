"""Out-of-sample (walk-forward) regime fitting for macro/factor panels.

The in-sample regime notebooks fit ONE Gaussian mixture on the whole history and
read labels back over the same span — descriptive, but it quietly uses the
future to label the past. This module re-fits month by month on only the data
available *before* each month and assigns that month **out-of-sample**, under two
windowing schemes (expanding/anchored and rolling fixed-length).

Two correctness properties ride along and are the point of the module:

- **No scaling lookahead.** A fresh ``StandardScaler`` is fit on the in-window
  training rows ONLY at each step; the out-of-sample month is transformed with
  that same scaler. Standardizing the whole panel once would leak future
  mean/variance into past months.
- **Stable regime identity.** GaussianMixture component indices are arbitrary and
  reshuffle on every re-fit. Each step's centroids are matched to a single fixed
  reference set (Hungarian assignment) so "Regime 2" means the same thing across
  the whole timeline.

The engine is notebook-agnostic and unit-testable in isolation: it takes a 2-D
array (or DataFrame) plus a date index and returns plain arrays. Determinism
comes from the explicit ``random_state`` passed into every mixture, not a global
seed.

Usage:
    from utils.regimes import walk_forward_regimes

    result = walk_forward_regimes(macro_df, scheme="expanding", random_state=42)
    # result.labels[t] is month t's regime, assigned using only data before t;
    # -1 (with result.flagged[t] == True) marks months whose training window was
    # below min_train_samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

__all__ = ["WalkForwardRegimeResult", "walk_forward_regimes", "match_clusters_to_reference"]


@dataclass
class WalkForwardRegimeResult:
    """Per-month out-of-sample regime assignments and metadata.

    Attributes
    ----------
    labels : np.ndarray
        Shape ``(n_months,)``, int. Reference-space regime id for each month;
        ``-1`` where the min-window guard skipped assignment.
    probabilities : np.ndarray
        Shape ``(n_months, n_components)``, float. Soft posteriors in reference
        order; ``np.nan`` rows where the month was skipped.
    flagged : np.ndarray
        Shape ``(n_months,)``, bool. ``True`` where the min-window guard fired
        (equivalently, where ``labels == -1``).
    dates : pd.Index
        The out-of-sample month for each row, aligned to ``labels``.
    scheme : str
        ``"expanding"`` or ``"rolling"``.
    reference_centroids : np.ndarray
        Shape ``(n_components, n_features)`` in raw feature space — the fixed
        reference all per-step centroids are matched against. All-``nan`` if no
        window ever reached ``min_train_samples``.

        Caveat: the reference is anchored to the *first* qualifying window, i.e.
        the smallest (``min_train_samples``-row) and so noisiest centroid
        estimate. Label identity stability therefore degrades when early regimes
        are poorly separated: if two regime centroids later drift past each other
        relative to this early anchor, the Hungarian match can flip them.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    flagged: np.ndarray
    dates: pd.Index
    scheme: str
    reference_centroids: np.ndarray


def match_clusters_to_reference(centroids: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map mixture component indices onto a fixed reference set.

    Solves a minimum-cost assignment (Hungarian) on pairwise Euclidean distance
    between ``centroids`` and ``reference``, both in the same feature space, so
    the mapping is a collision-free permutation: ``mapping[i]`` is the reference
    index that component ``i`` corresponds to.

    Parameters
    ----------
    centroids, reference : np.ndarray
        Shape ``(n_components, n_features)``. Must have the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(n_components,)``, int. ``mapping[i]`` = reference id for
        component ``i``.
    """
    centroids = np.asarray(centroids, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if centroids.shape != reference.shape:
        raise ValueError(f"centroids {centroids.shape} and reference {reference.shape} must match")
    cost = np.linalg.norm(centroids[:, None, :] - reference[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = np.empty(centroids.shape[0], dtype=int)
    mapping[row_ind] = col_ind
    return mapping


def _fit_step(
    train_x: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    n_init: int,
    reg_covar: float,
    random_state: int,
) -> tuple[GaussianMixture, StandardScaler, np.ndarray]:
    """Fit a fresh scaler + GMM on one in-window training block.

    Returns the fitted mixture, the scaler (fit on ``train_x`` only), and the
    component centroids inverse-transformed back to raw feature space so they are
    comparable across steps that each use their own scaler.
    """
    scaler = StandardScaler().fit(train_x)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state,
    ).fit(scaler.transform(train_x))
    centroids_raw = scaler.inverse_transform(gmm.means_)
    return gmm, scaler, centroids_raw


def _train_slice(t: int, scheme: str, rolling_window: int) -> slice:
    """Training-row slice for predicting month ``t`` (strictly causal: rows < t)."""
    if scheme == "expanding":
        return slice(0, t)
    if scheme == "rolling":
        return slice(max(0, t - rolling_window), t)
    raise ValueError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}")


def walk_forward_regimes(
    X: pd.DataFrame | np.ndarray,
    *,
    dates: pd.Index | None = None,
    scheme: str = "expanding",
    rolling_window: int = 60,
    min_train_samples: int = 40,
    n_components: int = 4,
    covariance_type: str = "full",
    n_init: int = 10,
    reg_covar: float = 1e-6,
    random_state: int = 42,
) -> WalkForwardRegimeResult:
    """Assign each month a regime out-of-sample under a walk-forward scheme.

    For each month ``t`` the model trains on rows **strictly before** ``t``
    (expanding: ``[0, t)``; rolling: ``[t - rolling_window, t)``), fits a fresh
    ``StandardScaler`` + ``GaussianMixture`` on that window, and predicts month
    ``t`` only. Per-step component ids are remapped onto a fixed reference set
    (established at the first window that meets ``min_train_samples``) so labels
    are comparable across the timeline.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Shape ``(n_months, n_features)``, sorted ascending by date. If a
        DataFrame with a ``DatetimeIndex``, that index is used for ``dates``.
    dates : pd.Index, optional
        Out-of-sample month per row. Falls back to ``X``'s index (DataFrame) or a
        ``RangeIndex``.
    scheme : str
        ``"expanding"`` (anchored) or ``"rolling"`` (last ``rolling_window`` rows).
    rolling_window : int
        Training length for ``scheme="rolling"``.
    min_train_samples : int
        Months whose training window has fewer rows are skipped (``label -1``,
        ``flagged True``, ``nan`` probabilities) rather than fitting a degenerate
        mixture.
    n_components, covariance_type, n_init, reg_covar, random_state
        ``GaussianMixture`` configuration. Defaults reproduce the baseline macro
        regime model. ``random_state`` is the sole source of determinism.

    Returns
    -------
    WalkForwardRegimeResult
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}")
    if scheme == "rolling" and rolling_window < min_train_samples:
        # Every rolling window is capped at rolling_window rows, so it could never
        # reach min_train_samples — the guard would fire for all months and the
        # result would be silently all-skipped. Fail loudly on the misconfig.
        raise ValueError(
            f"rolling_window ({rolling_window}) must be >= min_train_samples "
            f"({min_train_samples}) or every month is skipped"
        )

    values = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be 2-D (n_months, n_features), got shape {values.shape}")
    n_months, n_features = values.shape

    if dates is None:
        dates = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(n_months)
    if len(dates) != n_months:
        raise ValueError(f"dates length {len(dates)} != n_months {n_months}")

    labels = np.full(n_months, -1, dtype=int)
    probabilities = np.full((n_months, n_components), np.nan, dtype=float)
    flagged = np.ones(n_months, dtype=bool)
    reference: np.ndarray | None = None

    for t in range(n_months):
        train_x = values[_train_slice(t, scheme, rolling_window)]
        if train_x.shape[0] < min_train_samples:
            continue  # min-window guard: leave as skipped (-1 / nan / flagged)

        gmm, scaler, centroids_raw = _fit_step(
            train_x,
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
        )

        if reference is None:
            # Canonicalize the reference once via a full lexicographic sort over
            # all features (feature 0 primary), so regime ids are stable and
            # reproducible even when two centroids share a leading-feature value.
            order = np.lexsort(centroids_raw.T[::-1])
            reference = centroids_raw[order]

        mapping = match_clusters_to_reference(centroids_raw, reference)
        scaled_t = scaler.transform(values[t : t + 1])
        step_label = int(gmm.predict(scaled_t)[0])
        step_proba = gmm.predict_proba(scaled_t)[0]

        labels[t] = mapping[step_label]
        probabilities[t, mapping] = step_proba
        flagged[t] = False

    if reference is None:
        reference = np.full((n_components, n_features), np.nan, dtype=float)

    return WalkForwardRegimeResult(
        labels=labels,
        probabilities=probabilities,
        flagged=flagged,
        dates=dates,
        scheme=scheme,
        reference_centroids=reference,
    )
