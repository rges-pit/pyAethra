"""Command-line interface.

Two ways to configure a run:

    # everything in a YAML file
    pyAethra data.parquet --config config.yaml

    # or via flags (flags override anything also set in --config)
    pyAethra "lc/*.txt" --columns bjd mag mag_err -o results.csv
"""

import argparse
import sys

from .config import load_config
from .io import load_and_run

# Maps an argparse destination to the config key it sets. Most match 1:1;
# ffp_te_max differs because config uses the mixed-case physics symbol tE.
_FLAG_TO_CONFIG = {
    "time_col": "time_col",
    "mag_col": "mag_col",
    "err_col": "err_col",
    "flux_col": "flux_col",
    "flux_err_col": "flux_err_col",
    "zero_point": "zero_point",
    "recurrent_snr_floor": "recurrent_snr_floor",
    "group_col": "group_col",
    "filter_col": "filter_col",
    "target_filter": "target_filter",
    "primary_filter": "primary_filter",
    "secondary_filters": "secondary_filters",
    "min_points": "min_points",
    "season_gap_days": "season_gap_days",
    "ffp_te_max": "ffp_tE_max",
    "good_fit_chi2": "good_fit_chi2",
    "chromatic_min_points": "chromatic_min_points",
    "sep": "sep",
    "columns": "columns",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyAethra",
        description="Run the microlensing event-detection pipeline on a dataset.",
    )
    p.add_argument(
        "input_path",
        help="A table file (.parquet/.fits/.csv/.txt) or a glob pattern for "
        "per-object files.",
    )
    p.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to a YAML config file. Any flags given below override it.",
    )

    # All config flags default to None so we can tell whether the user set them.
    # Unset keys fall back to the pipeline's built-in defaults (shown below).
    p.add_argument("--time-col", default=None, help="Time column name (default: bjd)")
    p.add_argument("--mag-col", default=None, help="Magnitude column name (default: mag)")
    p.add_argument("--err-col", default=None, help="Error column name (default: mag_err)")
    p.add_argument("--flux-col", default=None, help="Flux column name (use instead of --mag-col)")
    p.add_argument("--flux-err-col", default=None, help="Flux error column name (required with --flux-col)")
    p.add_argument("--zero-point", type=float, default=None, help="Zero point for reporting mags from flux (default: 0.0)")
    p.add_argument("--recurrent-snr-floor", type=float, default=None, help="bump_snr floor for a season to count as a bump season (default: 25.0)")
    p.add_argument("--group-col", default=None, help="Column identifying each object (e.g. 'name')")
    p.add_argument("--filter-col", default=None, help="Filter/band column name (enables achromatic test)")
    p.add_argument("--target-filter", default=None, help="Primary detection band (default: F146)")
    p.add_argument("--primary-filter", default=None, help="Primary band for the achromatic test (default: F146)")
    p.add_argument("--secondary-filters", default=None, nargs="*", help="Secondary bands for the achromatic test")
    p.add_argument("--min-points", type=int, default=None, help="Minimum points per season (default: 10)")
    p.add_argument("--season-gap-days", type=float, default=None, help="Day gap separating seasons (default: 100)")
    p.add_argument("--ffp-te-max", type=float, default=None, help="Max tE (days) for an FFP candidate (default: 2.0)")
    p.add_argument("--good-fit-chi2", "--good-pspl-chi2", dest="good_fit_chi2", type=float, default=None,
                   help="Reduced chi2 threshold for a good fit of the selected model (default: 2.5)")
    p.add_argument("--chromatic-min-points", type=int, default=None, help="Min points per band for the achromatic test (default: 5)")
    p.add_argument("--sep", default=None, help="Column separator regex for text files (default: whitespace)")
    p.add_argument("--header", default=None, help="Header row index for text/CSV files (e.g. 0)")
    p.add_argument("--columns", default=None, nargs="*", help="Column names when the file has no header")

    p.add_argument("-o", "--output", default=None, help="Path to write results as CSV")
    p.add_argument("--debug", action="store_true", help="Print a per-object veto breakdown")
    return p


def build_config(args) -> dict:
    """Assemble the config dict. Precedence: CLI flag > --config YAML > defaults."""
    config = {}
    if args.config:
        config.update(load_config(args.config))

    for dest, key in _FLAG_TO_CONFIG.items():
        value = getattr(args, dest)
        if value is not None:
            config[key] = value

    if args.header is not None:
        config["header"] = int(args.header)

    # time is always required; photometry falls back to conventional mag names
    # only when no flux columns were given.
    config.setdefault("time_col", "bjd")
    if not config.get("flux_col"):
        config.setdefault("mag_col", "mag")
        config.setdefault("err_col", "mag_err")
    return config


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config(args)

    results = load_and_run(args.input_path, config, debug=args.debug)

    print(f"\nPipeline complete — {len(results)} object(s) processed.")
    counts = results["label"].value_counts()
    for label in ("microlensing", "ffp", "variable star", "flat"):
        print(f"  {label:<15} {int(counts.get(label, 0))}")

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
