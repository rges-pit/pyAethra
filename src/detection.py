"""Outlier, flatness, and bump-detection routines."""

import numpy as np
import pandas as pd

__all__ = [
    "has_consecutive_outliers",
    "flatness_chi2",
    "detect_bump",
    "recurrent_bump_veto",
]


def has_consecutive_outliers(threshold_mask, n_consecutive=3):
    count = 0
    for val in threshold_mask:
        if val:
            count += 1
            if count >= n_consecutive:
                return True
        else:
            count = 0
    return False


def flatness_chi2(flux, flux_err):
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    flux = flux[valid]
    flux_err = flux_err[valid]

    if len(flux) < 2:
        return np.nan, np.nan, np.nan

    w = 1.0 / flux_err**2
    wsum = np.sum(w)
    if not np.isfinite(wsum) or wsum <= 0:
        return np.nan, np.nan, np.nan

    c = np.sum(w * flux) / wsum
    chi2 = np.sum(w * (flux - c) ** 2)
    dof = len(flux) - 1
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return chi2, dof, chi2_red


def detect_bump(time, flux, flux_err, window_days=(1, 2, 5, 10, 20, 50),
                snr_threshold=3.0, n_consecutive=3, threshold_margin=1.0):
    time     = np.asarray(time, dtype=float)
    flux     = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]

    n = len(flux)
    if n < 5:
        return False, np.nan, np.nan

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return False, np.nan, np.nan
    cadence = np.median(dt)

    # Unbiased baseline. The old p30 percentile sat ~0.5σ below the mean, which
    # gave every point a positive offset that summed to a window bias growing
    # like 0.5·√w_pts — the source of the high-N phantom bumps. The median is
    # ≈ the mean for flat data and is barely moved by one localized bump.
    baseline_flux = np.median(flux)
    z = (flux - baseline_flux) / flux_err   # ~ N(0,1) per point under flat null

    best_sig = -np.inf
    best_window_days = np.nan
    n_trials = 0.0

    for w_days in window_days:
        w_pts = int(round(w_days / cadence))
        if w_pts < 2 or w_pts >= n:
            continue

        # Integrated, error-normalised excess over each length-w_pts window.
        # sum(z)/√w_pts is ~ N(0,1) under the flat null, so one threshold means
        # the same thing at any season length or cadence.
        rolling_sig = (
            pd.Series(z)
            .rolling(w_pts, center=True, min_periods=w_pts)
            .sum()
            .to_numpy()
            / np.sqrt(w_pts)
        )
        if not np.any(np.isfinite(rolling_sig)):
            continue

        n_trials += n / w_pts          # ≈ independent windows at this scale
        peak = np.nanmax(rolling_sig)
        if np.isfinite(peak) and peak > best_sig:
            best_sig = peak
            best_window_days = float(w_days)

    if not np.isfinite(best_sig) or n_trials < 1:
        return False, best_window_days, (best_sig if np.isfinite(best_sig) else np.nan)

    # Trials correction: the max of n_trials independent N(0,1) windows sits near
    # √(2 ln n_trials). Requiring best_sig to clear that by threshold_margin
    # holds the flat-null false-positive rate roughly constant across all N.
    threshold_eff = np.sqrt(2.0 * np.log(n_trials)) + threshold_margin
    bump_flag = bool(best_sig > threshold_eff)

    return bump_flag, best_window_days, float(best_sig)


def recurrent_bump_veto(season_rows, snr_floor=25.0, max_run_seasons=3):
    """
    Veto recurrent variability: bumps in more than one *excursion*.

    Reuses per-season scan results (season_rows) as the single source of truth
    for where bumps are. A 'bump season' is one the scan flagged with SNR >=
    snr_floor.

    Fires only on MULTIPLE EXCURSIONS — bump seasons that are not all
    contiguous (a bump, a return-to-baseline season, then another bump):
    rise-peak-fall / rise-peak-fall, which a single lens cannot produce.

    A single contiguous run of bump seasons is ONE excursion and does NOT veto —
    a genuine long-tE event whose single rise-peak-fall is split across a season
    gap looks exactly like that and must be protected.

    Returns: veto (bool), bump_seasons (list), n_excursions (int).
    """
    bump_seasons = sorted(
        r["season"] for r in season_rows
        if r["bump_flag"] and np.isfinite(r["bump_snr"]) and r["bump_snr"] >= snr_floor
    )

    runs = []
    prev = None
    for s in bump_seasons:
        if prev is None or s != prev + 1:
            runs.append(1)
        else:
            runs[-1] += 1
        prev = s

    n_excursions = len(runs)
    longest_run = max(runs) if runs else 0

    veto = bool(n_excursions >= 2 or longest_run > max_run_seasons)
    return veto, bump_seasons, n_excursions
