"""Main per-DataFrame pipeline driver."""

import numpy as np
import pandas as pd

from .achromatic import achromatic_test_from_event
from .detection import recurrent_bump_veto
from .global_detection import detect_bump_global
from .photometry import flux_to_mag, resolve_flux
from .pspl import fit_pspl_candidate
from .schema import OUTPUT_COLUMNS
from .seasons import analyze_season_scan, split_into_seasons
from .variability import periodic_veto_from_other_seasons

__all__ = ["run_pipeline_from_dataframe"]


def _empty_result(obj_name):
    """Return a zeroed-out result row for objects with insufficient data."""
    return {
        "name": obj_name, "ID": "flat",
        "scan_candidate": False, "best_season": np.nan, "is_flat": True,
        "chi2_flat": np.nan, "dof_flat": np.nan, "chi2_red_flat": np.nan,
        "bump_flag": False, "bump_snr": np.nan,
        "t0_fit": np.nan, "u0_fit": np.nan, "tE_fit": np.nan,
        "chi2_red_pspl": np.nan,
        "baseline_flux": np.nan, "peak_flux": np.nan,
        "baseline_mag": np.nan, "peak_mag": np.nan,
        "periodic": False, "recurrent": False, "chromatic": False,
        "n_seasons": np.nan, "is_achromatic": np.nan,
        "season_candidate": False,
    }

def _classify(scan_candidate, any_veto, is_ffp):
    """Collapse the bump/veto/PSPL flags into a single label."""
    if not scan_candidate:
        return "flat"
    if any_veto:
        return "variable star"
    return "ffp" if is_ffp else "microlensing"


def run_pipeline_from_dataframe(df, config, debug=False):
    """
    Run the microlensing pipeline on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain config['time_col'], plus either
        config['flux_col']/config['flux_err_col'] (used as-is) or
        config['mag_col']/config['err_col'] (converted to flux on entry).
    config : dict
        See example_run.ipynb (Configuration cell) for all keys.
    debug : bool
        If True, print a per-object breakdown of which vetoes fired and why.

    Returns
    -------
    pd.DataFrame with OUTPUT_COLUMNS plus veto/diagnostic columns.
    """
    time_col   = config["time_col"]
    group_col  = config.get("group_col")
    filter_col = config.get("filter_col")
    target_filter    = config.get("target_filter",    "F146")
    primary_filter   = config.get("primary_filter",   "F146")
    _sf = config.get("secondary_filters", config.get("secondary_filter", None))
    secondary_filters = [_sf] if isinstance(_sf, str) else (_sf or [])
    min_points        = config.get("min_points",           10)
    gap_days          = config.get("season_gap_days",     100)
    ffp_tE_max        = config.get("ffp_tE_max",          2.0)
    good_pspl_chi2    = config.get("good_pspl_chi2",      2.5)
    # NOTE: the periodic veto below currently hardcodes fap_threshold=1e-6
    # rather than reading config["fap_threshold"]. See README "Known quirks".
    chromatic_min_pts = config.get("chromatic_min_points",  5)
    zero_point        = config.get("zero_point",           0.0)
    recurrent_floor   = config.get("recurrent_snr_floor",  25.0)
    global_bin_days = config.get("global_bin_days", 1.0)
    global_windows = tuple(config.get("global_window_days", (20.0, 50.0, 100.0, 200.0, 400.0)))

    if time_col not in df.columns:
        raise ValueError(f"Required column '{time_col}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")

    working = df.copy()
    # Single conversion boundary: mag input -> flux here, flux input passes
    # through untouched. Everything downstream is flux-only.
    working["_flux"], working["_flux_err"] = resolve_flux(working, config)
    grouped = [("single_object", working)] if group_col is None else working.groupby(group_col)

    results = []

    for obj_name, obj_df_all in grouped:
        obj_df_all = obj_df_all.sort_values(time_col).copy()
        if len(obj_df_all) == 0:
            continue

        season_labels = split_into_seasons(obj_df_all[time_col].to_numpy(), gap_days=gap_days)
        obj_df_all["season_id"] = season_labels
        n_seasons = int(obj_df_all["season_id"].nunique())

        if filter_col is not None and filter_col in obj_df_all.columns:
            obj_df_primary = obj_df_all[obj_df_all[filter_col] == target_filter].copy()
        else:
            obj_df_primary = obj_df_all.copy()

        if len(obj_df_primary) == 0:
            if debug:
                print(f"[{obj_name}] SKIP — no rows in primary band '{target_filter}'")
            results.append({**_empty_result(obj_name), "n_seasons": n_seasons})
            continue

        season_rows = []
        for season_id, season_df in obj_df_primary.groupby("season_id"):
            result = analyze_season_scan(
                season_df[time_col].to_numpy(),
                season_df["_flux"].to_numpy(),
                season_df["_flux_err"].to_numpy(),
                min_points=min_points,
            )
            season_rows.append({
                "season": season_id,
                "time":     season_df[time_col].to_numpy(),
                "flux":     season_df["_flux"].to_numpy(),
                "flux_err": season_df["_flux_err"].to_numpy(),
                "n_pts":    len(season_df),
                **result,
            })

        if not season_rows:
            results.append({**_empty_result(obj_name), "n_seasons": n_seasons})
            continue

        best = max(season_rows, key=lambda r: r["bump_snr"] if np.isfinite(r["bump_snr"]) else -np.inf)

        # Whole-curve scan at long windows only. The per-season scan cannot see
        # an event whose rise, peak and fall are split across seasonal gaps;
        # this one cannot see a sub-day event. The union keeps each detector in
        # the regime it is good at.
        glob = detect_bump_global(
            obj_df_primary[time_col].to_numpy(),
            obj_df_primary["_flux"].to_numpy(),
            obj_df_primary["_flux_err"].to_numpy(),
            bin_days=global_bin_days,
            window_days=global_windows,
        )

        season_candidate = bool(best["bump_flag"] and best["is_non_flat"])
        scan_candidate = bool(season_candidate or glob["global_flag"])

        periodic_veto, veto_period, veto_fap, veto_reason = periodic_veto_from_other_seasons(
            obj_df_primary=obj_df_primary, best_season=best["season"],
            time_col=time_col, mag_col="_flux", err_col="_flux_err", season_col="season_id",
            min_points=max(40, min_points), fap_threshold=1e-6,
            min_period=config.get("periodic_min_period", 0.02),
            max_period=50, min_cycles=4, ceiling_fraction=0.7,
            event_delta_flux=best["peak_flux"] - best["baseline_flux"],
            amp_frac=config.get("periodic_amp_frac", 0.1),
        )

        recurrent_veto, bump_seasons, n_excursions = recurrent_bump_veto(
            season_rows, snr_floor=recurrent_floor,
        )
        # To disable the recurrent veto, uncomment the line below and comment out the block above:
        # recurrent_veto = False

        is_achromatic = achromatic_test_from_event(
            obj_df=obj_df_all,
            time_col=time_col, flux_col="_flux", flux_err_col="_flux_err",
            filter_col=filter_col,
            season_col="season_id",
            best_season=best["season"],
            t0_guess_raw=best["t0_guess_raw"],
            best_window_days=best["best_window"],
            primary_filter=primary_filter,
            secondary_filters=secondary_filters,
            min_points=chromatic_min_pts,
        )
        veto_chromatic = (is_achromatic is False)

        if debug:
            season_summary = ", ".join(
                f"S{r['season']}(n={r['n_pts']},snr={r['bump_snr']:.1f},bump={r['bump_flag']})"
                for r in season_rows
            )
            print(
                f"[{obj_name}] "
                f"n_seasons={n_seasons} | "
                f"scan_candidate={scan_candidate} "
                f"(season={season_candidate}, global={glob['global_flag']}"
                f"@snr={glob['global_snr']:.1f}/thr={glob['global_threshold']:.1f}"
                f",win={glob['global_window']:.0f}d,dur={glob['global_duration_days']:.0f}d) "
                f"(bump={best['bump_flag']}, non_flat={best['is_non_flat']}, "
                f"best_snr={best['bump_snr']:.2f}, best_season=S{best['season']}) | "
                f"vetoes: periodic={periodic_veto}({veto_reason}), "
                f"recurrent={recurrent_veto}({n_excursions} excursions, seasons={bump_seasons}), "
                f"chromatic={veto_chromatic}(is_achromatic={is_achromatic}) | "
                f"seasons: [{season_summary}]"
            )

        if scan_candidate:
            pspl_info = fit_pspl_candidate(
                best["time"], best["flux"], best["flux_err"],
                good_pspl_chi2=good_pspl_chi2, ffp_tE_max=ffp_tE_max,
            )
        else:
            pspl_info = {"t0_fit_raw": np.nan, "u0_fit": np.nan, "tE_fit": np.nan,
                         "chi2_red_pspl": np.nan, "is_candidate": False, "is_ffp_candidate": False}

        any_veto = bool(periodic_veto or recurrent_veto or veto_chromatic)
        ident = _classify(scan_candidate, any_veto, pspl_info["is_ffp_candidate"])

        t0_output = (
            pspl_info["t0_fit_raw"] - 2450000
            if np.isfinite(pspl_info["t0_fit_raw"]) else np.nan
        )

        results.append({
            "name":             obj_name,
            "label":               ident,
            "scan_candidate":   scan_candidate,
            "periodic":    periodic_veto,
            "recurrent":   recurrent_veto,
            "chromatic":   veto_chromatic,
            "n_seasons":        n_seasons,
            "best_season":      best["season"],
            "is_flat":          not best["is_non_flat"],
            "chi2_flat":        best["chi2_flat"],
            "dof_flat":         best["dof_flat"],
            "chi2_red_flat":    best["chi2_red_flat"],
            "bump_flag":        best["bump_flag"],
            "bump_snr":         best["bump_snr"],
            "season_candidate": season_candidate,
            "t0_fit":           t0_output,
            "u0_fit":           pspl_info["u0_fit"],
            "tE_fit":           pspl_info["tE_fit"],
            "chi2_red_pspl":    pspl_info["chi2_red_pspl"],
            "baseline_flux":    best["baseline_flux"],
            "peak_flux":        best["peak_flux"],
            "baseline_mag":     float(flux_to_mag(best["baseline_flux"], zero_point)),
            "peak_mag":         float(flux_to_mag(best["peak_flux"], zero_point)),
        })

    result_df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = np.nan
    return result_df[OUTPUT_COLUMNS]

def run_and_report(input_path, config, debug=False, output_csv=None):
    from pyAethra import load_and_run
    results = load_and_run(input_path, config, debug=debug)

    n = len(results)
    print(f"\nPipeline complete — {n} object(s) processed.")
    print(f"  periodic:    {results['periodic'].sum()}")
    print(f"  recurrent:   {results['recurrent'].sum()}")
    print(f"  chromatic:   {results['chromatic'].sum()}")
    counts = results["label"].value_counts()
    for label in ("microlensing", "ffp", "variable star", "flat"):
        print(f"  {label:<15} {int(counts.get(label, 0))}")
    if "scan_candidate" in results.columns:
        print(f"  scan_candidate:   {results['scan_candidate'].sum()}")

    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"  Saved → {output_csv}")

    return results
