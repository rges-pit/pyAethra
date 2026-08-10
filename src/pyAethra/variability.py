"""Variability and Lomb-Scargle periodicity tests."""

import warnings

import numpy as np
from astropy.timeseries import LombScargle

from .detection import has_consecutive_outliers

__all__ = [
    "is_non_flat_lightcurve",
    "lomb_scargle_test",
    "is_periodic",
    "periodic_veto_from_other_seasons",
]


def is_non_flat_lightcurve(mag, sigma_thresh=3, use_robust=True, n_consecutive=3):
    mag = np.asarray(mag, dtype=float)
    if len(mag) == 0 or not np.all(np.isfinite(mag)):
        return False
    med = np.median(mag)
    if use_robust:
        mad = np.median(np.abs(mag - med))
        sigma = 1.4826 * mad
    else:
        sigma = np.std(mag)
    if not np.isfinite(sigma) or sigma <= 0:
        return False
    mask = np.abs(mag - med) > sigma_thresh * sigma
    return has_consecutive_outliers(mask, n_consecutive=n_consecutive)


def lomb_scargle_test(time, mag, mag_err=None, min_period=0.1, max_period=100, min_points=10):
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)

    if mag_err is not None:
        mag_err = np.asarray(mag_err, dtype=float)
        valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    else:
        valid = np.isfinite(time) & np.isfinite(mag)

    time = time[valid]
    mag = mag[valid]
    if mag_err is not None:
        mag_err = mag_err[valid]

    if len(mag) < min_points or np.ptp(time) <= 0 or np.ptp(mag) <= 0:
        return np.nan, np.nan, np.nan

    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period

    try:
        ls = LombScargle(time, mag, mag_err)
        frequency, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )
    except Exception:
        return np.nan, np.nan, np.nan

    finite = np.isfinite(frequency) & np.isfinite(power)
    frequency = frequency[finite]
    power = power[finite]

    if len(power) == 0:
        return np.nan, np.nan, np.nan

    idx = np.argmax(power)
    max_power = power[idx]
    best_frequency = frequency[idx]

    if not np.isfinite(max_power) or not np.isfinite(best_frequency) or best_frequency <= 0:
        return np.nan, np.nan, np.nan

    best_period = 1.0 / best_frequency

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fap = ls.false_alarm_probability(max_power)
    except Exception:
        fap = np.nan

    return best_period, fap, max_power


def is_periodic(time, mag, mag_err=None, fap_threshold=0.01, min_points=10,
                min_period=0.1, max_period=100):
    period, fap, max_power = lomb_scargle_test(
        time, mag, mag_err,
        min_period=min_period, max_period=max_period, min_points=min_points,
    )
    if not np.isfinite(period) or not np.isfinite(fap):
        return False, period, fap
    return (fap < fap_threshold), period, fap

def _sinusoid_semi_amplitude(time, flux, flux_err, period):
    """
    Weighted least-squares fit of a sinusoid at a FIXED period.

    Model: flux ≈ c + a·sin(2πt/P) + b·cos(2πt/P), weighted by 1/σ².
    Returns the semi-amplitude √(a²+b²) in the units of `flux`, or nan if
    the fit is not well posed. Closed-form 3×3 solve — no iteration.
    """
    t = np.asarray(time,     dtype=float)
    f = np.asarray(flux,     dtype=float)
    e = np.asarray(flux_err, dtype=float)

    valid = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    t, f, e = t[valid], f[valid], e[valid]

    if len(t) < 4 or not np.isfinite(period) or period <= 0:
        return np.nan

    # Reference phases to the first epoch. BJD values are ~2.4e6, so at short
    # periods the absolute phase carries needless float64 error; this also
    # conditions the normal equations better. Amplitude is phase-invariant.
    t = t - t[0]

    omega = 2.0 * np.pi / period
    X = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    w = 1.0 / e**2

    M = X.T @ (X * w[:, None])
    rhs = X.T @ (w * f)
    try:
        _, a, b = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return np.nan

    return float(np.hypot(a, b))

def periodic_veto_from_other_seasons(
    obj_df_primary, best_season,
    time_col="bjd", mag_col="mag", err_col="mag_err", season_col="season_id",
    min_points=40, fap_threshold=1e-6, min_period=0.1, max_period=50,
    min_cycles=4, ceiling_fraction=0.7,
    event_delta_flux=None, amp_frac=0.1,
):
    """
    Veto if any off-event season shows significant internal periodicity.
    Each season is tested independently so inter-season gaps never enter
    the Lomb-Scargle, avoiding survey-cadence aliases.
    """
    other_df = obj_df_primary[obj_df_primary[season_col] != best_season].copy()

    if len(other_df) < min_points:
        return False, np.nan, np.nan, "not enough off-event data"

    for season_id, season_df in other_df.groupby(season_col):
        if len(season_df) < min_points:
            continue

        time    = season_df[time_col].to_numpy(dtype=float)
        mag     = season_df[mag_col].to_numpy(dtype=float)
        mag_err = season_df[err_col].to_numpy(dtype=float)

        baseline = time.max() - time.min()
        season_max_period = min(max_period, ceiling_fraction * baseline)
        if season_max_period <= min_period:
            continue

        periodic_flag, period, fap = is_periodic(
            time, mag, mag_err,
            fap_threshold=fap_threshold,
            min_points=min_points,
            min_period=min_period,
            max_period=season_max_period,
        )

        if not periodic_flag:
            continue
        if not np.isfinite(period) or period <= 0:
            continue
        if period >= ceiling_fraction * season_max_period:
            continue

        n_cycles = baseline / period
        if n_cycles < min_cycles:
            continue

        # Amplitude gate. At 12-min cadence a season holds ~8000 points, so a
        # fixed FAP of 1e-6 corresponds to a normalised power of ~0.005 — a
        # ~2 mmag systematic clears it easily. Detector/window systematics sit
        # ~3 orders of magnitude below a real event's Δflux; a genuine variable
        # contaminant modulates at a comparable level to the bump it produces.
        # Requiring amp_frac of the event amplitude separates the two on
        # physical grounds rather than on significance.
        amp_note = ""
        if event_delta_flux is not None and np.isfinite(event_delta_flux) and event_delta_flux > 0:
            semi_amp = _sinusoid_semi_amplitude(
                time, mag, mag_err, period
            )
            if np.isfinite(semi_amp):
                ratio = semi_amp / event_delta_flux
                if ratio < amp_frac:
                    continue  # too weak to be the event — keep scanning
                amp_note = f", amp={ratio:.2f}×event"

        return True, period, fap, (
            f"repeating signal in off-event season {season_id}{amp_note}"
        )

    return False, np.nan, np.nan, "no periodic off-event season found"
