# aethra

**a**utomated **e**vent **t**racker and c**h**aracterizer for **r**oman **a**lerts

A configurable pipeline for detecting **microlensing events** in photometric
light curves. It works with any dataset that has time, magnitude, and
magnitude-error columns (column names are configurable), automatically handling
file-format detection, multi-object grouping, observing-season splitting, and
multi-band ("achromatic") vetoes.

## What it does

For each object the pipeline:

1. Splits the light curve into observing **seasons** (by time gaps).
2. Scans each season for a brightening **bump** (rolling weighted-flux SNR) and
   checks the curve is **non-flat** (reduced χ² + consecutive outliers).
3. Applies three **vetoes** to reject variable stars:
   - **periodic** — significant Lomb–Scargle periodicity in off-event seasons,
   - **recurrent** — a comparable bump repeating in other seasons,
   - **chromatic** — the brightening disagrees across photometric bands.
4. Fits a **point-source point-lens (PSPL)** model to surviving candidates and
   flags short-timescale **free-floating-planet (FFP)** candidates.


## Installation
for the most updated version:

```bash
git clone https://github.com/rges-pit/aethra.git
cd aethra
pip install -e .
```
(Note: don't forget the . after the -e)

or stable version (note: this is not working yet): 

```
pip install aethra
```
Reading Parquet input needs the optional extra:

```bash
pip install -e ".[parquet]"
```

For development (tests + linting):

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9. Core dependencies: NumPy, pandas, SciPy, Astropy.

## Quick start

```python
from aethra import load_and_run

config = {
    "time_col": "bjd",
    "mag_col":  "mag",
    "err_col":  "mag_err",
    "group_col": "name",   # column identifying each object in the table
}

results = load_and_run("data.parquet", config)
candidates = results[results["is_candidate"]]
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
from aethra import run_pipeline_from_dataframe
results = run_pipeline_from_dataframe(df, config)
```

## Configuration from a YAML file

Rather than writing the `config` dict inline, you can keep all settings in a
YAML file and version it alongside your results — so every run records exactly
how it was configured. See [`examples/config.yaml`](examples/config.yaml) for a
fully commented template.

```python
from aethra import load_config, load_and_run

config = load_config("config.yaml")
results = load_and_run("data.parquet", config)
```

Keys you omit fall back to the built-in defaults; YAML `null` maps to Python
`None` (e.g. `group_col: null` means one object per file).


## Configuration reference

Passed as the `config` dict (or the matching CLI flag).

### Required

| Key        | Meaning                          |
|------------|----------------------------------|
| `time_col` | Time column (e.g. BJD)           |
| `mag_col`  | Magnitude column                 |
| `err_col`  | Magnitude-uncertainty column     |

### Grouping & input parsing

| Key        | Default | Meaning |
|------------|---------|---------|
| `group_col`| `None`  | Column whose unique values identify each source. `None` = one object per file/DataFrame. |
| `sep`      | `r"\s+"`| Separator regex for text files (`","` for CSV). |
| `header`   | `None`  | Header row index for text files; `None` = no header. |
| `columns`  | `None`  | Column names to assign when there is no header row. |

### Filters / achromatic test

| Key                | Default        | Meaning |
|--------------------|----------------|---------|
| `filter_col`       | `None`         | Band column. `None` skips the achromatic test. |
| `target_filter`    | `"F146"`       | Band used for event detection. |
| `primary_filter`   | `"F146"`       | Primary band in the achromatic test. |
| `secondary_filters`| `None`         | One band (str) or several (list) to compare against. |

### Tuning

| Key                   | Default | Meaning |
|-----------------------|---------|---------|
| `min_points`          | `10`    | Minimum points per season to analyze. |
| `season_gap_days`     | `100`   | Day gap that separates observing seasons. |
| `ffp_tE_max`          | `2.0`   | Max tE (days) to flag a free-floating-planet candidate. |
| `good_pspl_chi2`      | `2.5`   | Reduced-χ² threshold for an acceptable PSPL fit. |
| `chromatic_min_points`| `5`     | Min points per band for the achromatic test. |
| `fap_threshold`       | `0.01`  | False-alarm threshold for periodicity. **See Known quirks.** |

## Output columns

| Column | Type | Description |
|---|---|---|
| `name` | str | Object identifier |
| `is_candidate` | bool | Passed bump test and all three vetoes |
| `is_ffp_candidate` | bool | Candidate with tE < `ffp_tE_max` days |
| `is_variable_star` | bool | Passed bump test but rejected by at least one veto |
| `veto_periodic` | bool | Rejected by periodicity veto |
| `veto_recurrent` | bool | Rejected by recurrent bump veto |
| `veto_chromatic` | bool | Rejected by chromatic veto |
| `is_achromatic` | bool/nan | Result of achromatic test (`nan` = inconclusive) |
| `best_season` | int | Season ID with the highest bump SNR |
| `is_flat` | bool | Lightcurve is consistent with a flat baseline |
| `chi2_flat` | float | χ² of flat model fit |
| `dof_flat` | int | Degrees of freedom |
| `chi2_red_flat` | float | Reduced χ² of flat model |
| `bump_flag` | bool | Rolling-SNR bump detected |
| `bump_snr` | float | Peak bump SNR |
| `t0_fit` | float | PSPL best-fit peak time (HJD − 2450000) |
| `u0_fit` | float | PSPL best-fit impact parameter |
| `tE_fit` | float | PSPL best-fit Einstein crossing time (days) |
| `chi2_red_pspl` | float | Reduced χ² of PSPL fit |
| `baseline_mag` | float | Median baseline magnitude |
| `peak_mag` | float | Peak (brightest) magnitude |

## Package layout

```
src/aethra/
├── __init__.py      # public API
├── schema.py        # OUTPUT_COLUMNS
├── config.py        # load_config (YAML → config dict)
├── detection.py     # outlier / flatness / bump detection + recurrent veto
├── variability.py   # non-flatness + Lomb–Scargle periodicity veto
├── achromatic.py    # multi-band achromaticity test
├── pspl.py          # PSPL magnification model + fitting
├── seasons.py       # season splitting and per-season scan
├── pipeline.py      # run_pipeline_from_dataframe (main driver)
├── io.py            # file loaders + load_and_run dispatcher
└── cli.py           # `aethra` console script
```


## License

MIT — see [LICENSE](LICENSE).
