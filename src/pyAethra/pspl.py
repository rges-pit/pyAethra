"""Point-source point-lens (PSPL) magnification model and fitting."""

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "pspl_magnification",
    "solve_fs_fb",
    "fit_pspl",
    "fit_pspl_candidate",
]


def pspl_magnification(t, t0, u0, tE):
    tau = (t - t0) / tE
    u = np.sqrt(u0**2 + tau**2)
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


def solve_fs_fb(A, flux, flux_err):
    w = 1.0 / flux_err**2
    S_AA = np.sum(w * A * A)
    S_A1 = np.sum(w * A)
    S_11 = np.sum(w)
    S_AF = np.sum(w * A * flux)
    S_1F = np.sum(w * flux)
    M = np.array([[S_AA, S_A1], [S_A1, S_11]])
    b = np.array([S_AF, S_1F])
    try:
        Fs, Fb = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        Fs, Fb = np.nan, np.nan
    return Fs, Fb


def fit_pspl(time, flux, flux_err):
    """Fit a PSPL model in flux space. Fs/Fb are solved analytically."""
    time     = np.asarray(time,     dtype=float)
    flux     = np.asarray(flux,     dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]

    if len(time) < 10 or np.ptp(time) <= 0:
        return None

    peak_idx      = np.argmax(flux)
    t0_guess      = time[peak_idx]
    baseline_flux = np.median(np.sort(flux)[:max(10, len(flux) // 5)])
    peak_flux     = np.max(flux)
    half_level    = baseline_flux + 0.5 * (peak_flux - baseline_flux)
    above_half    = time[flux >= half_level]
    tE_guess      = (
        max((above_half.max() - above_half.min()) / 2.0, 1.0)
        if len(above_half) >= 2
        else max(np.ptp(time) / 20.0, 1.0)
    )
    u0_guess = 0.3

    def residuals(p):
        t0, tE, u0 = p
        if tE <= 0 or u0 <= 0:
            return np.full_like(flux, 1e6)
        A = pspl_magnification(time, t0, u0, tE)
        Fs, Fb = solve_fs_fb(A, flux, flux_err)
        if not np.isfinite(Fs) or not np.isfinite(Fb):
            return np.full_like(flux, 1e6)
        return (flux - (Fs * A + Fb)) / flux_err

    t0_pad = max(5.0, min(20.0, 0.25 * np.ptp(time)))
    bounds  = (
        [t0_guess - t0_pad, 0.1,  1e-3],
        [t0_guess + t0_pad, max(100.0, np.ptp(time)), 2.0],
    )

    try:
        res = least_squares(
            residuals,
            x0=np.array([t0_guess, tE_guess, u0_guess]),
            bounds=bounds,
            max_nfev=50000,
        )
    except Exception:
        return None

    if not res.success:
        return None

    t0_fit, tE_fit, u0_fit = res.x
    A_fit          = pspl_magnification(time, t0_fit, u0_fit, tE_fit)
    Fs_fit, Fb_fit = solve_fs_fb(A_fit, flux, flux_err)

    if not np.isfinite(Fs_fit) or not np.isfinite(Fb_fit):
        return None

    model_flux = Fs_fit * A_fit + Fb_fit
    chi2       = np.sum(((flux - model_flux) / flux_err) ** 2)
    dof        = len(flux) - 3
    chi2_red   = chi2 / dof if dof > 0 else np.nan

    return {"t0_fit": t0_fit, "tE_fit": tE_fit, "u0_fit": u0_fit, "chi2_red_pspl": chi2_red}


def fit_pspl_candidate(time, flux, flux_err, good_pspl_chi2=2.5, ffp_tE_max=2.0):
    pspl_result = fit_pspl(time, flux, flux_err)
    if pspl_result is None:
        return {"t0_fit_raw": np.nan, "u0_fit": np.nan, "tE_fit": np.nan,
                "chi2_red_pspl": np.nan, "is_candidate": False, "is_ffp_candidate": False}

    good_pspl_fit = (
        np.isfinite(pspl_result["tE_fit"])
        and np.isfinite(pspl_result["chi2_red_pspl"])
        and pspl_result["chi2_red_pspl"] < good_pspl_chi2
    )
    is_candidate     = bool(good_pspl_fit)
    is_ffp_candidate = bool(is_candidate and pspl_result["tE_fit"] < ffp_tE_max)

    return {
        "t0_fit_raw":    pspl_result["t0_fit"],
        "u0_fit":        pspl_result["u0_fit"],
        "tE_fit":        pspl_result["tE_fit"],
        "chi2_red_pspl": pspl_result["chi2_red_pspl"],
        "is_candidate":     is_candidate,
        "is_ffp_candidate": is_ffp_candidate,
    }
