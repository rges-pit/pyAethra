# pyAethra

**A**utomated **E**vent **T**racker and c**H**aracterizer for **R**oman **A**lerts

A configurable pipeline for detecting **microlensing events** in photometric
light curves. It works with any dataset that has a time column plus either
flux/flux-error or magnitude/magnitude-error columns (names are configurable),
automatically handling file-format detection, multi-object grouping,
observing-season splitting, and multi-band ("achromatic") vetoes.

The pipeline is **flux-native**: magnitudes are converted to flux once on entry
and every statistic downstream is computed in flux space. Magnitudes are
reconstructed only for reporting.

## What it does

For each object the pipeline:

1. Splits the light curve into observing **seasons** (by time gaps).
2. Scans each season for a brightening **bump** (trials-corrected rolling flux
   significance) and checks the curve is **non-flat** (reduced χ² + consecutive
   outliers). The best season is the one with the highest bump significance.
3. Applies three **vetoes** to reject variable stars:
   - **periodic** — significant Lomb–Scargle periodicity in off-event seasons,
   - **recurrent** — bumps in more than one *excursion* (non-contiguous bump
     seasons); a single contiguous run of bump seasons is one excursion and
     does **not** veto, so a long-tE event split across a season gap survives,
   - **chromatic** — the brightening disagrees across photometric bands.
4. Fits a **point-source point-lens (PSPL)** model to surviving candidates and
   flags short-timescale **free-floating-planet (FFP)** candidates.

Each object gets a single label in the `ID` column: `microlensing`, `ffp`,
`variable star`, or `flat`. The result is a tidy `pandas.DataFrame`, one row per
object, with the columns listed in [`OUTPUT_COLUMNS`](src/pyAethra/schema.py).

## Installation

For the most up-to-date version:

```bash
git clone https://github.com/rges-pit/pyAethra.git
cd pyAethra
pip install -e .
```

(Note: don't forget the `.` after the `-e`.)

Or the stable version:

```bash
pip install pyAethra
```

Requires Python ≥ 3.8. Core dependencies: NumPy, pandas, SciPy, Astropy.

## Quick start

```python
from pyAethra import load_and_run

config = {
    "time_col": "bjd",
    "mag_col":  "mag",
    "err_col":  "mag_err",
    "group_col": "name",   # column identifying each object in the table
}

results = load_and_run("data.parquet", config)
candidates = results[results["ID"].isin(["microlensing", "ffp"])]
```

If your data is already in flux, give the flux columns instead — nothing else
changes:

```python
config = {
    "time_col":     "bjd",
    "flux_col":     "flux",
    "flux_err_col": "flux_err",
    "group_col":    "name",
}
```

`load_and_run` auto-dispatches on the input type:

| `input_path`                                | Interpreted as |
|---------------------------------------------|----------------|
| `pd.DataFrame`                              | One table; objects split by `group_col` |
| `"data.parquet"` / `.fits` / `.csv` / `.txt`| One table file |
| `"lc/*.txt"` (glob)                         | One light curve per matched file |
| `("lc/*_W149.txt", "lc/*_Z087.txt")`        | Paired per-filter files matched by filename stem |

You can also call the per-DataFrame driver directly:

```python
from pyAethra import run_pipeline_from_dataframe
results = run_pipeline_from_dataframe(df, config)
```

Or run and print a summary of the veto counts and label tally in one call:

```python
from pyAethra.pipeline import run_and_report
results = run_and_report("data.parquet", config, output_csv="results.csv")
```

## Configuration from a YAML file

Rather than writing the `config` dict inline, you can keep all settings in a
YAML file and version it alongside your results — so every run records exactly
how it was configured. See [`examples/config.yaml`](examples/config.yaml) for a
fully commented template.

```python
from pyAethra import load_config, load_and_run

config = load_config("config.yaml")
results = load_and_run("data.parquet", config)
```

Keys you omit fall back to the built-in defaults; YAML `null` maps to Python
`None` (e.g. `group_col: null` means one object per file).

Run `pyAethra --help` for the full option list.

## Configuration reference

Passed as the `config` dict (or the matching CLI flag).

### Required

`time_col` is always required, plus **one** photometry pair.

| Key            | Meaning |
|----------------|---------|
| `time_col`     | Time column (e.g. BJD) |
| `mag_col`      | Magnitude column (converted to flux on entry) |
| `err_col`      | Magnitude-uncertainty column |
| `flux_col`     | Flux column — use instead of `mag_col`; passed through untouched |
| `flux_err_col` | Flux-uncertainty column — required whenever `flux_col` is set |

### Grouping & input parsing

| Key        | Default | Meaning |
|------------|---------|---------|
| `group_col`| `None`  | Column whose unique values identify each source. `None` = one object per file/DataFrame. |
| `sep`      | `r"\s+"`| Separator regex for text files (`","` for CSV). |
| `header`   | `None`  | Header row index for text files; `None` = no header. |
| `columns`  | `None`  | Column names to assign when there is no header row. |

### Photometry

| Key          | Default | Meaning |
|--------------|---------|---------|
| `zero_point` | `0.0`   | Zero point used for the mag ↔ flux conversion and for reporting `baseline_mag` / `peak_mag`. |

### Filters / achromatic test

| Key                | Default  | Meaning |
|--------------------|----------|---------|
| `filter_col`       | `None`   | Band column. `None` skips the achromatic test. |
| `target_filter`    | `"F146"` | Band used for event detection. |
| `primary_filter`   | `"F146"` | Primary band in the achromatic test. |
| `secondary_filters`| `None`   | One band (str) or several (list) to compare against. `secondary_filter` (singular) is accepted as an alias. |

### Tuning

| Key                   | Default | Meaning |
|-----------------------|---------|---------|
| `min_points`          | `10`    | Minimum points per season to analyze. |
| `season_gap_days`     | `100`   | Day gap that separates observing seasons. |
| `ffp_tE_max`          | `2.0`   | Max tE (days) to flag a free-floating-planet candidate. |
| `good_pspl_chi2`      | `2.5`   | Reduced-χ² threshold for an acceptable PSPL fit. |
| `chromatic_min_points`| `5`     | Min points per band for the achromatic test. |
| `recurrent_snr_floor` | `25.0`  | `bump_snr` floor for a season to count as a bump season in the recurrent veto. |
| `fap_threshold`       | `0.01`  | False-alarm threshold for periodicity. |

## Output columns

One row per object.

| Column | Type | Description |
|---|---|---|
| `name` | str | Object identifier |
| `label` | str | Classification: `microlensing`, `ffp`, `variable star`, or `flat` |
| `periodic` | bool | Rejected by periodicity veto |
| `recurrent` | bool | Rejected by recurrent bump veto |
| `chromatic` | bool | Rejected by chromatic veto |
| `best_season` | int | Season ID with the highest bump SNR |
| `is_flat` | bool | Best season is consistent with a flat baseline |
| `chi2_flat` | float | χ² of the flat-model fit |
| `dof_flat` | int | Degrees of freedom |
| `chi2_red_flat` | float | Reduced χ² of the flat model |
| `bump_flag` | bool | Trials-corrected bump detected |
| `bump_snr` | float | Peak bump significance |
| `t0_fit` | float | PSPL best-fit peak time (HJD − 2450000) |
| `u0_fit` | float | PSPL best-fit impact parameter |
| `tE_fit` | float | PSPL best-fit Einstein crossing time (days) |
| `chi2_red_pspl` | float | Reduced χ² of the PSPL fit |
| `baseline_flux` | float | Median baseline flux of the best season |
| `peak_flux` | float | Maximum flux of the best season |
| `baseline_mag` | float | Baseline flux converted back to magnitude |
| `peak_mag` | float | Peak flux converted back to magnitude |
| `scan_candidate` | bool | Bump detected **and** season is non-flat (input to `ID`) |

`label` is derived as: `flat` if `scan_candidate` is False; otherwise
`variable star` if any veto fired; otherwise `ffp` if `tE_fit < ffp_tE_max`,
else `microlensing`.

## How the label is decided

```
scan_candidate = bump_flag AND is_non_flat          (best season)
any_veto       = periodic OR recurrent OR chromatic

not scan_candidate            -> "flat"
scan_candidate AND any_veto   -> "variable star"
scan_candidate, PSPL tE < ffp_tE_max -> "ffp"
otherwise                     -> "microlensing"
```

A candidate also needs a PSPL fit with `chi2_red_pspl < good_pspl_chi2` to be
called an FFP; a poor fit still leaves the object as `microlensing`.


## Package layout

```
src/pyAethra/
├── __init__.py      # public API
├── schema.py        # OUTPUT_COLUMNS
├── config.py        # load_config (YAML → config dict)
├── photometry.py    # mag ↔ flux conversion + input-column resolution
├── detection.py     # outlier / flatness / bump detection + recurrent veto
├── global_detection.py     # whole light curve bump detection for long events
├── variability.py   # non-flatness + Lomb–Scargle periodicity veto
├── achromatic.py    # multi-band achromaticity test
├── pspl.py          # PSPL magnification model + fitting (Fs/Fb solved analytically)
├── seasons.py       # season splitting and per-season scan
├── pipeline.py      # run_pipeline_from_dataframe (main driver) + run_and_report
├── io.py            # file loaders + load_and_run dispatcher
└── cli.py           # `pyAethra` console script
```

## License

MIT — see [LICENSE](LICENSE).
