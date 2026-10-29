"""Variability and Lomb-Scargle periodicity tests."""

import warnings

import numpy as np
from astropy.timeseries import LombScargle

from .detection import has_consecutive_outliers
from .pspl import pspl_magnification, solve_fs_fb

__all__ = [
    "is_non_flat_lightcurve",
    "lomb_scargle_test",
    "is_periodic",
    "periodic_veto_from_other_seasons",
    "in_season_periodic_veto",
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

def _period_amplitude_mag(semi_amp, baseline_flux):
    """Peak-to-peak amplitude in magnitudes of a sinusoid of semi-amplitude
    `semi_amp` about `baseline_flux` (both in flux). nan if undefined."""
    if not (np.isfinite(semi_amp) and np.isfinite(baseline_flux)) or baseline_flux <= 0:
        return np.nan
    if semi_amp >= baseline_flux:
        return np.nan
    return float(2.5 * np.log10((baseline_flux + semi_amp) / (baseline_flux - semi_amp)))


def _record_period(info, time, flux, flux_err, period, fap, residual=None):
    """Fill `info` with the period, FAP and peak-to-peak amplitude (mag)."""
    if info is None:
        return
    signal = flux if residual is None else residual
    semi_amp = _sinusoid_semi_amplitude(time, signal, flux_err, period)
    info["period"] = float(period)
    info["fap"] = float(fap) if np.isfinite(fap) else np.nan
    info["amplitude_mag"] = _period_amplitude_mag(semi_amp, float(np.median(flux)))

def periodic_veto_from_other_seasons(
    obj_df_primary, best_season,
    time_col="bjd", mag_col="mag", err_col="mag_err", season_col="season_id",
    min_points=40, fap_threshold=1e-6, min_period=0.1, max_period=50,
    min_cycles=4, ceiling_fraction=0.7,
    event_delta_flux=None, amp_frac=0.1, info=None,
):
    """
    Veto if any off-event season shows significant internal periodicity.
    Each season is tested independently so inter-season gaps never enter
    the Lomb-Scargle, avoiding survey-cadence aliases.
    """
    strongest = None  # (fap, time, flux, err, period) of the lowest-FAP peak
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

        if np.isfinite(period) and np.isfinite(fap) and (strongest is None or fap < strongest[0]):
            strongest = (fap, time, mag, mag_err, period)

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
        _record_period(info, time, mag, mag_err, period, fap)
        return True, period, fap, (
            f"repeating signal in off-event season {season_id}{amp_note}"
        )
    if strongest is not None:
        fap_s, t_s, f_s, e_s, p_s = strongest
        _record_period(info, t_s, f_s, e_s, p_s, fap_s)

    return False, np.nan, np.nan, "no periodic off-event season found"


def _sinusoid_fit(t, f, e, period, t_ref):
    """
    Weighted [1, sin, cos] fit at a FIXED period, phases referenced to t_ref
    (pass the same t_ref to compare phases between subsets).
    Returns (semi_amplitude, phase) or (nan, nan).
    """
    if len(t) < 4 or not np.isfinite(period) or period <= 0:
        return np.nan, np.nan
    omega = 2.0 * np.pi / period
    X = np.column_stack([np.ones_like(t),
                         np.sin(omega * (t - t_ref)),
                         np.cos(omega * (t - t_ref))])
    w = 1.0 / e**2
    try:
        _, a, b = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * f))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def in_season_periodic_veto(
    time, flux, flux_err, t0, u0, tE, event_delta_flux,
    fap_threshold=1e-6, min_period=0.1, max_period=50, ceiling_fraction=0.7,
    min_cycles=3.0, amp_frac=0.1,
    n_segments=3, coherence_min=0.5, max_phase_diff=0.25, info=None,
):
    """
    Periodicity in the PSPL residuals of the event season itself. Used when
    there is no off-event season to test (e.g. a single short season).

    A single-bump PSPL fit absorbs at most one cycle of a pulsator; the other
    cycles survive in the residuals. A real event's residuals are white.

    Non-PSPL events (2L1S, 2S1L) also leave structured residuals, so the
    signal is additionally required to REPEAT COHERENTLY: fitted at the fixed
    LS period in each of n_segments disjoint time segments, every segment must
    carry comparable amplitude (min/max >= coherence_min) at the same phase
    (within max_phase_diff cycles of the whole-season fit). Binary features
    are tied to t0 and cannot light up every segment alike; a pulsator does.
    min_cycles should be >= n_segments so each segment holds >= 1 cycle.

    The amplitude gate is the same as in periodic_veto_from_other_seasons:
    the residual sinusoid must be >= amp_frac of the event Δflux.

    Returns (veto, period, fap, reason).
    """
    t = np.asarray(time,     dtype=float)
    f = np.asarray(flux,     dtype=float)
    e = np.asarray(flux_err, dtype=float)
    valid = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    t, f, e = t[valid], f[valid], e[valid]

    if not np.all(np.isfinite([t0, u0, tE, event_delta_flux])) or event_delta_flux <= 0 or tE <= 0:
        return False, np.nan, np.nan, "in-season: no PSPL fit"
    if len(t) < 40:
        return False, np.nan, np.nan, "in-season: too few points"

    A = pspl_magnification(t, t0, u0, tE)
    Fs, Fb = solve_fs_fb(A, f, e)
    if not (np.isfinite(Fs) and np.isfinite(Fb)):
        return False, np.nan, np.nan, "in-season: PSPL flux solve failed"
    resid = f - (Fs * A + Fb)

    baseline = np.ptp(t)
    p_max = min(max_period, ceiling_fraction * baseline)
    if p_max <= min_period:
        return False, np.nan, np.nan, "in-season: baseline too short"

    periodic_flag, period, fap = is_periodic(
        t, resid, e, fap_threshold=fap_threshold, min_points=40,
        min_period=min_period, max_period=p_max,
    )

    if np.isfinite(period) and period > 0:
        # Amplitude of the residual sinusoid, relative to the season's flux.
        _record_period(info, t, f, e, period, fap, residual=resid)

    if not periodic_flag:
        return False, period, fap, "in-season: no significant residual period"
    if period >= ceiling_fraction * p_max:
        return False, period, fap, "in-season: period at search ceiling"
    if baseline / period < min_cycles:
        return False, period, fap, f"in-season: {baseline / period:.1f} cycles < {min_cycles}"

    amp, phi_all = _sinusoid_fit(t, resid, e, period, t[0])
    if not (np.isfinite(amp) and amp / event_delta_flux >= amp_frac):
        return False, period, fap, f"in-season: amp={amp / event_delta_flux:.2f}×event < {amp_frac}"

    edges = t[0] + baseline * np.arange(n_segments + 1) / n_segments
    edges[-1] = np.inf
    seg = [(t >= lo) & (t < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    fits = [_sinusoid_fit(t[m], resid[m], e[m], period, t[0]) for m in seg]
    amps = np.array([a for a, _ in fits])
    phis = np.array([p for _, p in fits])
    if not np.all(np.isfinite(amps)) or amps.max() <= 0:
        return False, period, fap, "in-season: segments unfittable"

    ratio = amps.min() / amps.max()
    dphi = np.max(np.abs((phis - phi_all + np.pi) % (2 * np.pi) - np.pi)) / (2 * np.pi)
    if ratio < coherence_min or dphi > max_phase_diff:
        return False, period, fap, (
            f"in-season: incoherent (seg amp ratio={ratio:.2f}, dphase={dphi:.2f} cyc)"
        )

    return True, period, fap, (
        f"in-season residual period {period:.3f}d, amp={amp / event_delta_flux:.2f}×event, "
        f"seg amp ratio={ratio:.2f}"
    )