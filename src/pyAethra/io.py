"""Input loaders and the auto-dispatching entry point."""

import glob
import os

import pandas as pd

from .pipeline import run_pipeline_from_dataframe
from .schema import OUTPUT_COLUMNS

__all__ = ["load_and_run"]


def _read_table_file(filepath, config):
    """
    Read a single table file into a DataFrame.
    Supports: .parquet, .fits / .fit, .csv, .tsv, and any whitespace-delimited
    text file (including headerless .txt).
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".parquet":
        return pd.read_parquet(filepath)

    if ext in (".fits", ".fit"):
        from astropy.table import Table
        return Table.read(filepath).to_pandas()

    return pd.read_csv(
        filepath,
        sep=config.get("sep", r"\s+"),
        names=config.get("columns"),
        header=config.get("header", None),
        engine="python",
    )


def _file_stem(path, primary_filter, secondary_filter):
    """Strip filter tags and extension to get a matchable object key."""
    base = os.path.basename(path)
    for tag in (f"_{primary_filter}", f"_{secondary_filter}",
                f"-{primary_filter}", f"-{secondary_filter}"):
        base = base.replace(tag, "")
    return os.path.splitext(base)[0]


def _run_single_table(filepath_or_df, config, debug=False):
    """One file (or DataFrame) that may contain many objects via group_col."""
    if isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df
        label = "<DataFrame>"
    else:
        label = filepath_or_df
        print(f"Reading {label} …")
        df = _read_table_file(filepath_or_df, config)

    print(f"  {len(df):,} rows, columns: {list(df.columns)}")
    return run_pipeline_from_dataframe(df, config, debug=debug)


def _run_per_object_files(pattern, config, debug=False):
    """One lightcurve per file; filename becomes the object name."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARNING] No files matched: {pattern!r}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    print(f"Found {len(files)} file(s) matching {pattern!r}")
    all_results = []
    for filepath in files:
        try:
            df = _read_table_file(filepath, config)
        except Exception as exc:
            print(f"  [WARNING] Could not read {filepath}: {exc}")
            continue
        result = run_pipeline_from_dataframe(df, {**config, "group_col": None}, debug=debug)
        result = result.copy()
        result.loc[:, "name"] = os.path.basename(filepath)
        all_results.append(result)

    if not all_results:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(all_results, ignore_index=True)[OUTPUT_COLUMNS]


def _run_paired_files(primary_pattern, secondary_pattern, config, debug=False):
    """Two per-filter files per object, matched by filename stem."""
    pf = config.get("primary_filter",   "W149")
    sf = config.get("secondary_filter", "Z087")
    fc = config.get("filter_col",       "filt")

    primary_files   = {_file_stem(f, pf, sf): f for f in glob.glob(primary_pattern)}
    secondary_files = {_file_stem(f, pf, sf): f for f in glob.glob(secondary_pattern)}
    all_keys = sorted(set(primary_files) | set(secondary_files))

    if not all_keys:
        print(f"[WARNING] No files matched: {primary_pattern!r} / {secondary_pattern!r}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    print(f"Found {len(all_keys)} object(s) across paired filter files")
    all_results = []
    for key in all_keys:
        pieces = []
        for filt, fmap in [(pf, primary_files), (sf, secondary_files)]:
            if key not in fmap:
                continue
            try:
                part = _read_table_file(fmap[key], config)
            except Exception as exc:
                print(f"  [WARNING] Could not read {fmap[key]}: {exc}")
                continue
            part[fc] = filt
            pieces.append(part)
        if not pieces:
            continue
        df_obj = pd.concat(pieces, ignore_index=True)
        df_obj["name"] = key
        result = run_pipeline_from_dataframe(df_obj, config, debug=debug)
        result = result.copy()
        result.loc[:, "name"] = key
        all_results.append(result)

    if not all_results:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(all_results, ignore_index=True)[OUTPUT_COLUMNS]


def load_and_run(input_path, config, debug=False):
    """
    Auto-dispatch based on the type / value of input_path.

    Parameters
    ----------
    input_path : str, tuple, or pd.DataFrame
        - pd.DataFrame         → single table; objects identified by group_col
        - tuple of two strings → paired per-filter glob patterns
                                 e.g. ("lc/*_W149.txt", "lc/*_Z087.txt")
        - str (no wildcards)   → single table file (.parquet, .fits, .csv, .txt …)
        - str (with wildcards) → one lightcurve per matched file
    config : dict
        Pipeline configuration. See example_run.ipynb for all keys.
    debug : bool
        If True, print per-object veto breakdown while running.

    Returns
    -------
    pd.DataFrame with OUTPUT_COLUMNS.
    """
    if isinstance(input_path, pd.DataFrame):
        return _run_single_table(input_path, config, debug=debug)

    if isinstance(input_path, tuple):
        if len(input_path) != 2:
            raise ValueError("When input_path is a tuple it must have exactly two elements: "
                             "(primary_glob, secondary_glob).")
        return _run_paired_files(input_path[0], input_path[1], config, debug=debug)

    if not isinstance(input_path, str):
        raise TypeError(f"input_path must be a str, tuple, or DataFrame; got {type(input_path).__name__}")

    TABLE_EXTS = {".parquet", ".fits", ".fit", ".csv", ".tsv"}
    ext = os.path.splitext(input_path)[1].lower()
    has_wildcards = any(c in input_path for c in ("*", "?", "["))

    if not has_wildcards and ext in TABLE_EXTS:
        return _run_single_table(input_path, config, debug=debug)

    if not has_wildcards:
        return _run_single_table(input_path, config, debug=debug)

    return _run_per_object_files(input_path, config, debug=debug)
