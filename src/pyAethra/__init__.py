"""
pyAethra — Microlensing event-detection pipeline
===================================================

Works with any photometric dataset that has time, magnitude, and
magnitude-error columns (names are configurable).

Quick start
-----------
    from aethra import load_and_run

    config = {
        "time_col": "bjd",
        "mag_col":  "mag",
        "err_col":  "mag_err",
        # ... see the README for the full configuration reference
    }
    results_df = load_and_run("data.parquet", config)

The most common entry points are :func:`load_and_run` (auto-dispatches on the
input type) and :func:`run_pipeline_from_dataframe` (runs on an in-memory
DataFrame). The lower-level building blocks (bump detection, periodicity vetoes,
PSPL fitting, etc.) are also importable from their respective submodules.
"""

from .achromatic import achromatic_test_from_event
from .config import load_config
from .detection import (
    detect_bump,
    flatness_chi2,
    has_consecutive_outliers,
    recurrent_bump_veto,
)
from .io import load_and_run
from .photometry import flux_to_mag, mag_to_flux, resolve_flux
from .pipeline import run_pipeline_from_dataframe
from .pspl import (
    fit_pspl,
    fit_pspl_candidate,
    pspl_magnification,
    solve_fs_fb,
)
from .schema import OUTPUT_COLUMNS
from .seasons import analyze_season_scan, split_into_seasons
from .variability import (
    is_non_flat_lightcurve,
    is_periodic,
    lomb_scargle_test,
    periodic_veto_from_other_seasons,)
from .global_detection import (bin_to_nights, detect_bump_global)


__version__ = "0.1.1"

__all__ = [
    "__version__",
    "OUTPUT_COLUMNS",
    # entry points
    "load_and_run",
    "run_pipeline_from_dataframe",
    "load_config",
    # photometry
    "mag_to_flux",
    "flux_to_mag",
    "resolve_flux",
    # detection
    "has_consecutive_outliers",
    "flatness_chi2",
    "detect_bump",
    "recurrent_bump_veto",
    # variability / periodicity
    "is_non_flat_lightcurve",
    "lomb_scargle_test",
    "is_periodic",
    "periodic_veto_from_other_seasons",
    # achromatic
    "achromatic_test_from_event",
    # PSPL
    "pspl_magnification",
    "solve_fs_fb",
    "fit_pspl",
    "fit_pspl_candidate",
    # seasons
    "split_into_seasons",
    "analyze_season_scan",
    #global detection
    "bin_to_nights",
    "detect_bump_global",
]
