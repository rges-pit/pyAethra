"""Main per-DataFrame pipeline driver."""

import numpy as np
import pandas as pd

from .achromatic import achromatic_test_from_event
from .detection import recurrent_bump_veto
from .pspl import fit_pspl_candidate
from .schema import OUTPUT_COLUMNS
from .seasons import analyze_season_scan, split_into_seasons
from .variability import periodic_veto_from_other_seasons

__all__ = ["run_pipeline_from_dataframe"]


def _empty_result(obj_name):
    """Return a zeroed-out result row for objects with insufficient data."""
    return {
        "name": obj_name, "is_candidate": False, "is_ffp_candidate": False,
        "is_variable_star": False, "best_season": np.nan, "is_flat": True,
        "chi2_flat": np.nan, "dof_flat": np.nan, "chi2_red_flat": np.nan,
        "bump_flag": False, "bump_snr": np.nan,
        "t0_fit": np.nan, "u0_fit": np.nan, "tE_fit": np.nan,
        "chi2_red_pspl": np.nan, "baseline_mag": np.nan, "peak_mag": np.nan,
        "veto_periodic": False, "veto_recurrent": False, "veto_chromatic": False,
        "n_seasons": np.nan, "is_achromatic": np.nan,
    }


def run_pipeline_from_dataframe(df, config, debug=False):
    """
    Run the microlensing pipeline on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns named in config['time_col'],
        config['mag_col'], and config['err_col'].
    config : dict
        See example_run.ipynb (Configuration cell) for all keys.
    debug : bool
        If True, print a per-object breakdown of which vetoes fired and why.

    Returns
    -------
    pd.DataFrame with OUTPUT_COLUMNS plus veto/diagnostic columns.
    """
    time_col   = config["time_col"]
    mag_col    = config["mag_col"]
    err_col    = config["err_col"]
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

    for col in [time_col, mag_col, err_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame. "
                             f"Available columns: {list(df.columns)}")

    working = df.copy()
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
                season_df[mag_col].to_numpy(),
                season_df[err_col].to_numpy(),
                min_points=min_points,
            )
            season_rows.append({
                "season": season_id,
                "time":    season_df[time_col].to_numpy(),
                "mag":     season_df[mag_col].to_numpy(),
                "mag_err": season_df[err_col].to_numpy(),
                "n_pts":   len(season_df),
                **result,
            })

        if not season_rows:
            results.append({**_empty_result(obj_name), "n_seasons": n_seasons})
            continue

        best = max(season_rows, key=lambda r: r["bump_snr"] if np.isfinite(r["bump_snr"]) else -np.inf)

        scan_candidate = bool(best["bump_flag"] and best["is_non_flat"])

        periodic_veto, veto_period, veto_fap, veto_reason = periodic_veto_from_other_seasons(
            obj_df_primary=obj_df_primary, best_season=best["season"],
            time_col=time_col, mag_col=mag_col, err_col=err_col, season_col="season_id",
            min_points=max(40, min_points), fap_threshold=1e-6,
            min_period=0.1, max_period=50, min_cycles=4, ceiling_fraction=0.7,
        )

        recurrent_veto, recurrent_hits = recurrent_bump_veto(
            obj_df_primary=obj_df_primary, best_season=best["season"],
            best_bump_snr=best["bump_snr"],
            time_col=time_col, mag_col=mag_col, err_col=err_col, season_col="season_id",
            min_points=min_points, min_other_bump_seasons=1,
        )
        # To disable the recurrent veto, uncomment the line below and comment out the block above:
        # recurrent_veto = False

        is_achromatic = achromatic_test_from_event(
            obj_df=obj_df_all,
            time_col=time_col, mag_col=mag_col, err_col=err_col,
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
                f"(bump={best['bump_flag']}, non_flat={best['is_non_flat']}, "
                f"best_snr={best['bump_snr']:.2f}, best_season=S{best['season']}) | "
                f"vetoes: periodic={periodic_veto}({veto_reason}), "
                f"recurrent={recurrent_veto}({len(recurrent_hits)} other seasons), "
                f"chromatic={veto_chromatic}(is_achromatic={is_achromatic}) | "
                f"seasons: [{season_summary}]"
            )

        if scan_candidate:
            pspl_info = fit_pspl_candidate(
                best["time"], best["mag"], best["mag_err"],
                good_pspl_chi2=good_pspl_chi2, ffp_tE_max=ffp_tE_max,
            )
        else:
            pspl_info = {"t0_fit_raw": np.nan, "u0_fit": np.nan, "tE_fit": np.nan,
                         "chi2_red_pspl": np.nan, "is_candidate": False, "is_ffp_candidate": False}

        is_variable_star = bool(scan_candidate and (periodic_veto or recurrent_veto or veto_chromatic))
        is_candidate     = bool(
            scan_candidate and pspl_info["is_candidate"] and not is_variable_star
        )
        is_ffp_candidate = bool(is_candidate and pspl_info["is_ffp_candidate"])

        t0_output = (
            pspl_info["t0_fit_raw"] - 2450000
            if np.isfinite(pspl_info["t0_fit_raw"]) else np.nan
        )

        results.append({
            "name":             obj_name,
            "is_candidate":     is_candidate,
            "is_ffp_candidate": is_ffp_candidate,
            "is_variable_star": is_variable_star,
            "scan_candidate":   scan_candidate,
            "veto_periodic":    periodic_veto,
            "veto_recurrent":   recurrent_veto,
            "veto_chromatic":   veto_chromatic,
            "is_achromatic":    is_achromatic,
            "n_seasons":        n_seasons,
            "best_season":      best["season"],
            "is_flat":          not best["is_non_flat"],
            "chi2_flat":        best["chi2_flat"],
            "dof_flat":         best["dof_flat"],
            "chi2_red_flat":    best["chi2_red_flat"],
            "bump_flag":        best["bump_flag"],
            "bump_snr":         best["bump_snr"],
            "t0_fit":           t0_output,
            "u0_fit":           pspl_info["u0_fit"],
            "tE_fit":           pspl_info["tE_fit"],
            "chi2_red_pspl":    pspl_info["chi2_red_pspl"],
            "baseline_mag":     best["baseline_mag"],
            "peak_mag":         best["peak_mag"],
        })

    result_df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = np.nan
    return result_df[OUTPUT_COLUMNS]

def run_and_report(input_path, config, debug=False, output_csv=None):
    from aethra import load_and_run
    results = load_and_run(input_path, config, debug=debug)

    n = len(results)
    print(f"\nPipeline complete — {n} object(s) processed.")
    print(f"  veto_periodic:    {results['veto_periodic'].sum()}")
    print(f"  veto_recurrent:   {results['veto_recurrent'].sum()}")
    print(f"  veto_chromatic:   {results['veto_chromatic'].sum()}")
    print(f"  is_variable_star: {results['is_variable_star'].sum()}  (scan_candidate + any veto)")
    print(f"  is_candidate:     {results['is_candidate'].sum()}")
    print(f"  is_ffp_candidate: {results['is_ffp_candidate'].sum()}")
    if "scan_candidate" in results.columns:
        print(f"  scan_candidate:   {results['scan_candidate'].sum()}")

    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"  Saved → {output_csv}")

    return results
