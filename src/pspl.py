"""Point-source (PSPL) and finite-source (FSPL) point-lens models and fitting.

FSPL uses the exact uniform-source magnification of Witt & Mao (1994),
evaluated with scipy's elliptic integrals only (no VBMicrolensing).
Both models are fitted to every candidate; the finite-source fit is
*reported* always and *selected* only when the data require it.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ellipe, ellipk, elliprf, elliprj

__all__ = [
    "pspl_magnification",
    "fspl_magnification",
    "solve_fs_fb",
    "fit_pspl",
    "fit_fspl",
    "fit_pspl_candidate",
]

# Beyond this many source radii the finite-source correction is
# ~rho^2/(8u^2) < 1e-6, so the cheaper point-source formula is used.
_FS_SWITCH_U_OVER_RHO = 500.0
# Within |u/rho - 1| < this, the elliptic-integral form loses precision
# (Pi(n|m) diverges as n -> 1). There the magnification is interpolated
# linearly between the analytic u == rho limit and the Witt & Mao value at
# the band edge; checked against direct quadrature to < 2e-5 relative.
_FS_LIMB_BAND = 1e-4


# ---------------------------------------------------------------------------
# Magnification models
# ---------------------------------------------------------------------------

def pspl_magnification(t, t0, u0, tE):
    tau = (t - t0) / tE
    u = np.sqrt(u0**2 + tau**2)
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


def _pspl_from_u(u):
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


def _ellip_pi(n, m):
    """Complete elliptic integral of the third kind, Pi(n | m), via Carlson."""
    return elliprf(0.0, 1.0 - m, 1.0) + (n / 3.0) * elliprj(0.0, 1.0 - m, 1.0, 1.0 - n)


def _witt_mao(u, rho):
    """Uniform-source point-lens magnification (Witt & Mao 1994), u != rho.

    Elliptic integrals use the parameter convention m = k^2 (as scipy does).
    """
    d = u - rho
    s = u + rho
    root = np.sqrt(4.0 + d**2)
    a1 = 0.5 * s * root / rho**2
    a2 = -d * (4.0 + 0.5 * (u**2 - rho**2)) / (rho**2 * root)
    a3 = 2.0 * d**2 * (1.0 + rho**2) / (rho**2 * s * root)
    n = 4.0 * u * rho / s**2
    m = 4.0 * n / (4.0 + d**2)
    return (a1 * ellipe(m) + a2 * ellipk(m) + a3 * _ellip_pi(n, m)) / np.pi


def _witt_mao_equal(u):
    """Analytic limit of the Witt & Mao expression at u == rho."""
    u2 = u**2
    return (2.0 / u + (1.0 + u2) / u2 * (np.pi / 2.0 + np.arcsin((u2 - 1.0) / (u2 + 1.0)))) / np.pi


def fspl_magnification(t, t0, u0, tE, rho):
    """Finite (uniform) source, point lens magnification.

    Exact for any rho, including rho >> 1. Uses the PSPL formula where
    u > 500 rho (agreement < 1e-6).
    """
    t = np.asarray(t, dtype=float)
    tau = (t - t0) / tE
    u = np.sqrt(u0**2 + tau**2)
    A = np.empty_like(u)

    delta = u / rho - 1.0
    far = u > _FS_SWITCH_U_OVER_RHO * rho
    limb = np.abs(delta) < _FS_LIMB_BAND
    near = ~(far | limb)

    A[far] = _pspl_from_u(u[far])
    A[near] = _witt_mao(u[near], rho)
    if np.any(limb):
        d = delta[limb]
        A_eq = _witt_mao_equal(rho)
        A_edge = _witt_mao(rho * (1.0 + np.where(d >= 0, 1.0, -1.0) * _FS_LIMB_BAND), rho)
        A[limb] = A_eq + (A_edge - A_eq) * np.abs(d) / _FS_LIMB_BAND
    return A


# ---------------------------------------------------------------------------
# Linear flux solve
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean(time, flux, flux_err):
    time     = np.asarray(time,     dtype=float)
    flux     = np.asarray(flux,     dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    return time[valid], flux[valid], flux_err[valid]


def _initial_guess(time, flux):
    """Data-driven (t0, tE) starting point shared by PSPL and FSPL."""
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
    return t0_guess, tE_guess


def _t0_pad(time):
    return max(5.0, min(20.0, 0.25 * np.ptp(time)))


def _chi2_summary(model_flux, flux, flux_err, n_params):
    chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
    dof = len(flux) - n_params
    return chi2, (chi2 / dof if dof > 0 else np.nan)


# ---------------------------------------------------------------------------
# PSPL fit
# ---------------------------------------------------------------------------

def fit_pspl(time, flux, flux_err):
    """Fit a PSPL model in flux space. Fs/Fb are solved analytically."""
    time, flux, flux_err = _clean(time, flux, flux_err)

    if len(time) < 10 or np.ptp(time) <= 0:
        return None

    t0_guess, tE_guess = _initial_guess(time, flux)
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

    t0_pad = _t0_pad(time)
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

    chi2, chi2_red = _chi2_summary(Fs_fit * A_fit + Fb_fit, flux, flux_err, 3)

    return {"t0_fit": t0_fit, "tE_fit": tE_fit, "u0_fit": u0_fit,
            "chi2_pspl": chi2, "chi2_red_pspl": chi2_red, "n_points": len(flux)}


# ---------------------------------------------------------------------------
# FSPL fit
# ---------------------------------------------------------------------------

def fit_fspl(time, flux, flux_err, pspl_seed=None,
             rho_seeds=(0.01, 0.1, 1.0, 3.0),
             rho_bounds=(1e-4, 20.0), tE_min=0.01, u0_max=10.0):
    """Fit a uniform-source FSPL model in flux space (Fs/Fb analytic).

    Free parameters are (t0, tE, u0, log10 rho). The fit is started from
    each value in ``rho_seeds`` and the lowest-chi2 solution is kept, so a
    fit started near rho -> 0 cannot trap a flat-topped (rho > 1) event in
    the point-source minimum. ``pspl_seed`` (the fit_pspl result) supplies
    t0/tE/u0 starting values and adds one extra start at the PSPL solution.
    """
    time, flux, flux_err = _clean(time, flux, flux_err)

    if len(time) < 10 or np.ptp(time) <= 0:
        return None

    t0_data, tE_data = _initial_guess(time, flux)
    if pspl_seed is not None:
        t0_base, tE_base, u0_base = pspl_seed["t0_fit"], pspl_seed["tE_fit"], pspl_seed["u0_fit"]
    else:
        t0_base, tE_base, u0_base = t0_data, tE_data, 0.3

    t0_pad = _t0_pad(time)
    log_rho_lo, log_rho_hi = np.log10(rho_bounds[0]), np.log10(rho_bounds[1])
    lower = np.array([t0_data - t0_pad, tE_min, 0.0, log_rho_lo])
    upper = np.array([t0_data + t0_pad, max(100.0, np.ptp(time)), u0_max, log_rho_hi])

    def residuals(p):
        t0, tE, u0, log_rho = p
        if tE <= 0 or u0 < 0:
            return np.full_like(flux, 1e6)
        A = fspl_magnification(time, t0, u0, tE, 10.0**log_rho)
        if not np.all(np.isfinite(A)):
            return np.full_like(flux, 1e6)
        Fs, Fb = solve_fs_fb(A, flux, flux_err)
        if not np.isfinite(Fs) or not np.isfinite(Fb):
            return np.full_like(flux, 1e6)
        return (flux - (Fs * A + Fb)) / flux_err

    # Starting points. For rho > 1 the event duration is ~2 rho tE, so tE is
    # scaled down to keep the seeded duration close to the observed one, and
    # u0 is placed inside the source disk.
    starts = []
    if pspl_seed is not None:
        starts.append([t0_base, tE_base, u0_base, log_rho_lo + 1.0])
    for rho in rho_seeds:
        tE_s = tE_base / max(rho, 1.0)
        u0_s = u0_base if rho < u0_base else 0.5 * rho
        starts.append([t0_base, tE_s, u0_s, np.log10(rho)])

    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float), lower + 1e-9, upper - 1e-9)
        try:
            res = least_squares(residuals, x0=x0, bounds=(lower, upper), max_nfev=20000)
        except Exception:
            continue
        if not res.success:
            continue
        if best is None or res.cost < best.cost:
            best = res

    if best is None:
        return None

    t0_fit, tE_fit, u0_fit, log_rho_fit = best.x
    rho_fit = 10.0**log_rho_fit
    A_fit = fspl_magnification(time, t0_fit, u0_fit, tE_fit, rho_fit)
    Fs_fit, Fb_fit = solve_fs_fb(A_fit, flux, flux_err)

    if not np.isfinite(Fs_fit) or not np.isfinite(Fb_fit):
        return None

    chi2, chi2_red = _chi2_summary(Fs_fit * A_fit + Fb_fit, flux, flux_err, 4)

    return {"t0_fit": t0_fit, "tE_fit": tE_fit, "u0_fit": u0_fit,
            "rho_fit": rho_fit, "t_star_fit": rho_fit * tE_fit,
            "chi2_fspl": chi2, "chi2_red_fspl": chi2_red, "n_points": len(flux),
            "Fs": float(Fs_fit), "Fb": float(Fb_fit)}


# ---------------------------------------------------------------------------
# Candidate wrapper: run both, decide whether finite source is detected
# ---------------------------------------------------------------------------

def _empty_candidate():
    return {
        "t0_fit_raw": np.nan, "u0_fit": np.nan, "tE_fit": np.nan,
        "chi2_red_pspl": np.nan,
        "t0_fit_fspl_raw": np.nan, "u0_fit_fspl": np.nan, "tE_fit_fspl": np.nan,
        "rho_fit_fspl": np.nan, "t_star_fspl": np.nan, "chi2_red_fspl": np.nan,
        "chi2_fspl": np.nan, "n_points": np.nan,
        "source_flux_fspl": np.nan, "blend_flux_fspl": np.nan,
        "finite_source_delta_chi2": np.nan, "finite_source": False,
        "is_candidate": False, "is_ffp_candidate": False,
    }


def fit_pspl_candidate(time, flux, flux_err, good_fit_chi2=2.5, ffp_tE_max=2.0,
                       rho_seeds=(0.01, 0.1, 1.0, 3.0),
                       fs_delta_chi2_min=None, fs_u0_rho_max=2.0,
                       fs_min_crossing_points=3):
    """Fit PSPL and FSPL to every candidate and report both.

    ``finite_source`` is True only when all three hold:
      * delta_chi2 = chi2_PSPL - chi2_FSPL exceeds ``fs_delta_chi2_min``
        (default None -> ln N, i.e. the BIC penalty for one extra parameter);
      * the lens passes over or near the source: u0 < fs_u0_rho_max * rho;
      * the source crossing is sampled: at least ``fs_min_crossing_points``
        points within |t - t0| < t_star.

    Candidate gate: the reduced chi2 of the selected model (FSPL if
   ``finite_source`` or PSPL failed, otherwise PSPL) must be below
   ``good_fit_chi2``. The FFP timescale comes from the same model.

    FFP gate: timescale of the selected model. If FSPL is selected with
    rho > 1, tE is degenerate with rho and Fs, so t_star (>= tE) is used.
    """
    out = _empty_candidate()

    time_c, flux_c, err_c = _clean(time, flux, flux_err)
    pspl = fit_pspl(time_c, flux_c, err_c)
    fspl = fit_fspl(time_c, flux_c, err_c, pspl_seed=pspl, rho_seeds=rho_seeds)

    if pspl is not None:
        out.update({
            "t0_fit_raw":    pspl["t0_fit"],
            "u0_fit":        pspl["u0_fit"],
            "tE_fit":        pspl["tE_fit"],
            "chi2_red_pspl": pspl["chi2_red_pspl"],
        })
    if fspl is not None:
        out.update({
            "t0_fit_fspl_raw": fspl["t0_fit"],
            "u0_fit_fspl":     fspl["u0_fit"],
            "tE_fit_fspl":     fspl["tE_fit"],
            "rho_fit_fspl":    fspl["rho_fit"],
            "t_star_fspl":     fspl["t_star_fit"],
            "chi2_red_fspl":   fspl["chi2_red_fspl"],
            "chi2_fspl": fspl["chi2_fspl"],
            "n_points": fspl["n_points"],
            "source_flux_fspl": fspl["Fs"],
            "blend_flux_fspl": fspl["Fb"],
        })

    # --- finite-source selection ---
    finite_source = False
    if pspl is not None and fspl is not None:
        delta_chi2 = pspl["chi2_pspl"] - fspl["chi2_fspl"]
        out["finite_source_delta_chi2"] = delta_chi2
        threshold = np.log(fspl["n_points"]) if fs_delta_chi2_min is None else fs_delta_chi2_min
        n_crossing = int(np.sum(np.abs(time_c - fspl["t0_fit"]) < fspl["t_star_fit"]))
        finite_source = bool(
            delta_chi2 > threshold
            and fspl["u0_fit"] < fs_u0_rho_max * fspl["rho_fit"]
            and n_crossing >= fs_min_crossing_points
        )
    elif pspl is None and fspl is not None:
        # PSPL failed outright (e.g. flat-topped event); FSPL is the only
        # model, but without a null model the source size is not "detected".
        finite_source = False
    out["finite_source"] = finite_source

    # --- candidate gate: reduced chi2 of the selected model ---
    use_fspl = finite_source or pspl is None
    chi2_sel = out["chi2_red_fspl"] if use_fspl else out["chi2_red_pspl"]
    is_candidate = bool(np.isfinite(chi2_sel) and chi2_sel < good_fit_chi2)

    # --- FFP gate: timescale of the selected model ---
    if finite_source:
        timescale = out["t_star_fspl"] if out["rho_fit_fspl"] > 1.0 else out["tE_fit_fspl"]
    elif pspl is not None:
        timescale = out["tE_fit"]
    else:
        timescale = out["tE_fit_fspl"]
    is_ffp_candidate = bool(is_candidate and np.isfinite(timescale) and timescale < ffp_tE_max)

    out["is_candidate"] = is_candidate
    out["is_ffp_candidate"] = is_ffp_candidate
    return out
