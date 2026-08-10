"""Achromaticity test: compare the primary detection band to secondary bands."""

import numpy as np

__all__ = ["achromatic_test_from_event"]


def achromatic_test_from_event(
    obj_df, time_col, flux_col, flux_err_col, filter_col, season_col,
    best_season, t0_guess_raw, best_window_days,
    primary_filter="F146", secondary_filters=None, min_points=4,
):
    """
    Test whether the event is achromatic by comparing the primary detection
    band against one or more secondary bands.

    Works entirely in flux: a brightening is a POSITIVE delta, and the event
    peak is the flux maximum.

    secondary_filters can be a string (one band) or a list of strings
    (multiple bands, e.g. ["F087", "F213"]). Missing or too-sparse bands
    abstain (nan) rather than veto.

    Return values
    -------------
    True  : primary brightens AND at least one secondary confirms,
            none clearly disagree.
    False : primary brightens BUT at least one secondary clearly disagrees
            (wrong direction or peak time offset > tolerance) -> chromatic veto.
    nan   : cannot make a statement (no filter col, no t0, primary too sparse,
            or all secondaries abstained).
    """
    if filter_col is None or filter_col not in obj_df.columns:
        return np.nan

    if not np.isfinite(t0_guess_raw):
        return np.nan

    if secondary_filters is None:
        return np.nan
    if isinstance(secondary_filters, str):
        secondary_filters = [secondary_filters]

    season_df = obj_df[obj_df[season_col] == best_season].copy()
    if len(season_df) == 0:
        return np.nan

    window = best_window_days
    if not np.isfinite(window) or window <= 0:
        window = 5.0
    half_window = max(2.0, 1.5 * window)
    time_tol    = max(2.0, half_window)

    def summarize(df_band):
        """Return summary dict, or None if the band is too sparse to judge."""
        t = df_band[time_col].to_numpy(dtype=float)
        f = df_band[flux_col].to_numpy(dtype=float)
        e = df_band[flux_err_col].to_numpy(dtype=float)

        valid = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        t, f, e = t[valid], f[valid], e[valid]
        if len(t) < min_points:
            return None

        event_mask = np.abs(t - t0_guess_raw) <= half_window
        base_mask  = np.abs(t - t0_guess_raw) >  half_window
        if event_mask.sum() < 2 or base_mask.sum() < 3:
            return None

        baseline_flux    = np.median(f[base_mask])
        baseline_scatter = 1.4826 * np.median(np.abs(f[base_mask] - baseline_flux))
        if not np.isfinite(baseline_scatter) or baseline_scatter <= 0:
            baseline_scatter = np.std(f[base_mask])

        event_time = t[event_mask]
        event_flux = f[event_mask]
        event_err  = e[event_mask]

        # Brightening = flux MAXIMUM (was argmin in magnitudes).
        i_peak = np.argmax(event_flux)
        delta  = event_flux[i_peak] - baseline_flux

        local_err = np.median(event_err)
        denom     = max(local_err, baseline_scatter, 1e-6)

        return {
            "peak_time":    event_time[i_peak],
            "delta_flux":   delta,
            "significance": delta / denom,
        }

    # -- primary band ---------------------------------------------------------
    df_primary = season_df[season_df[filter_col] == primary_filter].copy()
    s_primary  = summarize(df_primary)
    if s_primary is None:
        return np.nan
    if s_primary["delta_flux"] <= 0:
        return False

    # -- secondary bands ------------------------------------------------------
    band_votes = []
    for filt in secondary_filters:
        if filt not in season_df[filter_col].values:
            band_votes.append(None)
            continue

        s = summarize(season_df[season_df[filter_col] == filt].copy())

        if s is None:
            band_votes.append(None)
            continue

        if s["delta_flux"] <= 0:
            band_votes.append(False)
            continue

        if abs(s["peak_time"] - s_primary["peak_time"]) > time_tol:
            band_votes.append(False)
            continue

        if s["significance"] < 2.0:
            band_votes.append(None)
            continue

        band_votes.append(True)

    # -- combine votes --------------------------------------------------------
    if any(v is False for v in band_votes):
        return False
    if any(v is True for v in band_votes):
        return True
    return np.nan
