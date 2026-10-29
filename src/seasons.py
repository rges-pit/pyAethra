"""Season splitting and per-season scan."""

import warnings
import numpy as np

from .detection import detect_bump, flatness_chi2
from .variability import is_non_flat_lightcurve

__all__ = ["split_into_seasons", "assign_seasons", "analyze_season_scan"]


def split_into_seasons(time, gap_days=100):
    time = np.asarray(time, dtype=float)
    if len(time) == 0:
        return np.array([], dtype=int)
    order    = np.argsort(time)
    t_sorted = time[order]
    labels_sorted = np.ones(len(t_sorted), dtype=int)
    season_id = 1
    for i in range(1, len(t_sorted)):
        if t_sorted[i] - t_sorted[i - 1] > gap_days:
            season_id += 1
        labels_sorted[i] = season_id
    labels = np.empty(len(time), dtype=int)
    labels[order] = labels_sorted
    return labels

def assign_seasons(time, gap_days=100, current_season=None, season_edges="auto"):
    """Fixed, survey-wide season number for every row.

    Call ONCE on the full input (all objects, all filters) so that a season
    number means the same stretch of time for every object, including one
    with no data in some season.

    season_edges
        "auto": boundaries are gaps > gap_days in the pooled timestamps.
            If at least `current_season` seasons are found (or current_season
            is None) they are numbered 1, 2, ... forwards. If fewer are found,
            the input is assumed to end in the current season and seasons are
            numbered backwards from current_season (with a warning).
        list of floats: start time of each season in the input time system,
            season 1 first. Rows before the first start get season 0.

    Rows with non-finite time get season 0.
    """
    time = np.asarray(time, dtype=float)
    labels = np.zeros(len(time), dtype=int)
    finite = np.isfinite(time)
    if not np.any(finite):
        return labels

    offset = 0
    if isinstance(season_edges, str):
        if season_edges != "auto":
            raise ValueError(
                f"season_edges must be 'auto' or a list of season start times, "
                f"got {season_edges!r}"
            )
        t = np.unique(time[finite])  # sorted
        starts = np.concatenate(([t[0]], t[1:][np.diff(t) > gap_days]))
        if current_season is not None and len(starts) < current_season:
            offset = current_season - len(starts)
            warnings.warn(
                f"Found {len(starts)} season(s) but current_season={current_season}; "
                f"assuming the data ends in the current season and numbering "
                f"seasons {offset + 1}..{current_season}."
            )
    else:
        starts = np.sort(np.asarray(season_edges, dtype=float))

    labels[finite] = np.searchsorted(starts, time[finite], side="right") + offset
    return labels


def analyze_season_scan(time, flux, flux_err, min_points=10):
    time    = np.asarray(time,    dtype=float)
    flux    = np.asarray(flux,    dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]

    empty = {
        "is_non_flat": False, "baseline_flux": np.nan, "peak_flux": np.nan,
        "bump_flag": False, "bump_snr": np.nan, "best_window": np.nan,
        "chi2_flat": np.nan, "dof_flat": np.nan, "chi2_red_flat": np.nan,
        "t0_guess_raw": np.nan,
    }

    if len(flux) < min_points or np.ptp(time) <= 0:
        return empty

    positive_err = flux_err[flux_err > 0]
    if len(positive_err) == 0:
        return empty

    # Relative floor only. The old `max(..., 1e-3)` was 1e-3 *magnitudes*;
    # in flux the same constant is either negligible or catastrophic depending
    # on the instrument's units, so the absolute term has to go.
    err_floor = np.median(positive_err) * 0.1
    flux_err = np.where(flux_err <= 0, err_floor, flux_err)

    chi2_flat, dof_flat, chi2_red_flat = flatness_chi2(flux, flux_err)

    outlier_flag = is_non_flat_lightcurve(flux, sigma_thresh=2.0, n_consecutive=2)
    is_non_flat = outlier_flag or (np.isfinite(chi2_red_flat) and chi2_red_flat > 1.5)

    bump_flag, best_window, bump_snr = detect_bump(
        time, flux, flux_err, window_days=(0.1, 2, 5, 10, 20, 50),
    )

    t0_guess_raw = time[np.argmax(flux)] if len(time) > 0 else np.nan

    return {
        "is_non_flat": is_non_flat,
        "baseline_flux": float(np.median(flux)),
        "peak_flux": float(np.max(flux)),
        "bump_flag": bump_flag,
        "bump_snr": bump_snr,
        "best_window": best_window,
        "chi2_flat": chi2_flat,
        "dof_flat": dof_flat,
        "chi2_red_flat": chi2_red_flat,
        "t0_guess_raw": t0_guess_raw,
    }
