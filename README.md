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

1. Assigns every observation to an observing **season**. Seasons are numbered
   once across the whole input (all objects, all bands), so season *N* is the
   same stretch of time for every object. With `current_season` set, later
   seasons are dropped to simulate a run part-way through the survey.
2. Scans each season for a brightening **bump** (trials-corrected rolling flux
   significance over windows from 0.1 to 50 days) and checks the curve is
   **non-flat** (reduced χ² + consecutive outliers). The best season is the one
   with the highest bump significance. A second scan over the whole light
   curve at long windows (20–400 days) catches long events that span season
   gaps.
3. Applies three **checks** to reject variable stars:
   - **periodic** — significant Lomb–Scargle periodicity (periods down to
     `periodic_min_period`, 0.02 d by default, which covers δ Scuti stars) in
     any off-event season, at an amplitude comparable to the event. With no
     off-event season available (e.g. a single short season), the PSPL
     residuals of the event season are searched instead, and the signal must
     repeat coherently across segments of the season so binary-lens structure
     is not mistaken for pulsation,
   - **recurrent** — bumps in more than one *excursion* (non-contiguous bump
     seasons), or one run of more than three contiguous bump seasons. A
     shorter contiguous run is one excursion and does **not** veto, so a
     long-tE event split across a season gap survives,
   - **chromatic** — the brightening disagrees across photometric bands
     (opposite direction, or peak times offset beyond a tolerance).
4. Fits both a **point-source point-lens (PSPL)** and a **finite-source
   point-lens (FSPL)** model to every candidate. The FSPL model uses the exact
   uniform-source magnification of Witt & Mao (1994), evaluated with SciPy's
   elliptic integrals only. Finite-source effects are *selected* only when the
   data require them (see [Finite source](#finite-source)).
5. Flags short-timescale **free-floating-planet (FFP)** candidates.
6. Optionally writes one JSON **alert** per non-flat object
   (see [Alerts](#alerts)).

Each object gets a single label in the `label` column: `microlensing`, `ffp`,
`variable star`, or `flat`. The result is a `pandas.DataFrame`, one row per
object, with the columns listed in
[`OUTPUT_COLUMNS`](https://github.com/rges-pit/pyAethra/blob/main/src/pyAethra/schema.py).

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

Requires Python ≥ 3.8. Core dependencies: NumPy, pandas, SciPy, Astropy, PyYAML.

## Quick start

```python
from pyAethra import load_config, load_and_run

config = load_config("config.yaml")
results = load_and_run("data.parquet", config)
print(results)
```

See [`examples/config.yaml`](https://github.com/rges-pit/pyAethra/blob/main/examples/config.yaml)
for a fully commented template, and
[`examples/example_run.ipynb`](https://github.com/rges-pit/pyAethra/blob/main/examples/example_run.ipynb)
for a complete run on the RMDC26 beginner tier read directly from Hugging Face.

The config can also be a plain dict:

```python
config = {
    "time_col": "bjd",
    "mag_col":  "mag",
    "err_col":  "mag_err",
    "group_col": "name",   # column identifying each object in the table
}
```

If your data is already in flux, give the flux columns instead — nothing else
changes:

```python
config = {
    "time_col":     "bjd",
    "flux_col":     "flux_uJy",
    "flux_err_col": "flux_err_uJy",
    "group_col":    "name",
    "zero_point":   23.9,      # AB zero point for µJy; used for the reported magnitudes
}
```

`load_and_run` auto-dispatches on the input type:

| `input_path`                                | Interpreted as |
|---------------------------------------------|----------------|
| `pd.DataFrame`                              | One table; objects split by `group_col` |
| `"data.parquet"` / `.fits` / `.csv` / `.txt`| One table file |
| `"lc/*.txt"` (glob)                         | One light curve per matched file |
| `("lc/*_F146.txt", "lc/*_F087.txt")`        | Paired per-filter files matched by filename stem |

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

Rather than writing the `config` dict inline, keep all settings in a YAML file
and version it alongside your results, so every run records exactly how it was
configured.

```python
from pyAethra import load_config, load_and_run

config = load_config("config.yaml")
results = load_and_run("data.parquet", config)
```

Keys you omit fall back to the built-in defaults; YAML `null` maps to Python
`None` (e.g. `group_col: null` means one object per file).

From the command line:

```bash
pyAethra data.parquet --config config.yaml -o results.csv
```

CLI flags override values in the YAML file. Run `pyAethra --help` for the full
option list.

## Configuration reference

### Required

`time_col` is always required, plus **one** photometry pair.

| Key            | Meaning |
|----------------|---------|
| `time_col`     | Time column (e.g. BJD) |
| `mag_col`      | Magnitude column (converted to flux on entry) |
| `err_col`      | Magnitude-uncertainty column |
| `flux_col`     | Flux column — used instead of `mag_col` when present in the data; passed through untouched |
| `flux_err_col` | Flux-uncertainty column — required whenever `flux_col` is used |

If both pairs are configured, the flux columns are used when they exist in the
data and the magnitude columns otherwise, so one config can serve datasets in
either form.

### Grouping & input parsing

| Key        | Default    | Meaning |
|------------|------------|---------|
| `group_col`| `None`     | Column whose unique values identify each source. `None` = one object per file/DataFrame. |
| `ra_col`   | `"ra_deg"` | Right-ascension column, copied to the `RA` output. Missing column → `NaN`. |
| `dec_col`  | `"dec_deg"`| Declination column, copied to the `Dec` output. Missing column → `NaN`. |
| `sep`      | `r"\s+"`   | Separator regex for text files (`","` for CSV). |
| `header`   | `None`     | Header row index for text files; `None` = no header. |
| `columns`  | `None`     | Column names to assign when there is no header row. |

### Seasons

| Key               | Default  | Meaning |
|-------------------|----------|---------|
| `season_gap_days` | `100`    | Day gap that separates observing seasons. |
| `season_edges`    | `"auto"` | `"auto"` finds seasons from gaps longer than `season_gap_days`; or a list of season start times (season 1 first). |
| `current_season`  | `None`   | Keep seasons `1..N` and drop later ones (simulates running mid-mission). `None` runs on all seasons. If the data contain fewer than `N` seasons, they are assumed to end in season `N`. |

### Photometry

| Key          | Default | Meaning |
|--------------|---------|---------|
| `zero_point` | `0.0`   | Zero point for the mag ↔ flux conversion and for every reported magnitude (`baseline_mag`, `peak_mag`, `source_mag`, `blend_mag`). Use `23.9` for flux in µJy (AB). For magnitude input it cancels out. |

### Filters / achromatic test

| Key                | Default  | Meaning |
|--------------------|----------|---------|
| `filter_col`       | `None`   | Band column. `None` skips the achromatic test. |
| `target_filter`    | `"F146"` | Band used for event detection and fitting. |
| `primary_filter`   | `"F146"` | Primary band in the achromatic test. |
| `secondary_filters`| `None`   | One band (str) or several (list) to compare against. `secondary_filter` (singular) is accepted as an alias. |
| `chromatic_min_points` | `5` | Min points per band for the achromatic test. |

### Detection

| Key                  | Default | Meaning |
|----------------------|---------|---------|
| `min_points`         | `10`    | Minimum points per season to analyse. |
| `global_bin_days`    | `1.0`   | Bin width (days) for the whole-light-curve scan. |
| `global_window_days` | `[20, 50, 100, 200, 400]` | Window widths (days) for the whole-light-curve scan. |


### Fitting

| Key                      | Default | Meaning |
|--------------------------|---------|---------|
| `good_fit_chi2`          | `2.5`   | Reduced-χ² threshold for an acceptable fit of the selected model (PSPL, or FSPL when finite source is selected). |
| `ffp_tE_max`             | `2.0`   | Max timescale (days) to flag a free-floating-planet candidate. |
| `fspl_rho_seeds`         | `[0.01, 0.1, 1.0, 3.0]` | Starting values of ρ for the FSPL fit; the lowest-χ² result is kept. |
| `fs_delta_chi2_min`      | `None`  | Δχ² (PSPL − FSPL) needed to select finite source. `None` = ln N (the BIC penalty for one extra parameter). |
| `fs_u0_rho_max`          | `2.0`   | Finite source only if u₀ < this × ρ. |
| `fs_min_crossing_points` | `3`     | Minimum points within \|t − t₀\| < t★ (the source crossing must be sampled). |

## Output columns

One row per object.

| Column | Type | Description |
|---|---|---|
| `name` | str | Object identifier |
| `RA` | float | Right ascension (from `ra_col`) |
| `Dec` | float | Declination (from `dec_col`) |
| `label` | str | Classification: `microlensing`, `ffp`, `variable star`, or `flat` |
| `periodic` | bool | Rejected by periodicity veto |
| `recurrent` | bool | Rejected by recurrent bump veto |
| `chromatic` | bool | Rejected by chromatic veto |
| `current_season` | int | The `current_season` the run used (`None` = all seasons) |
| `is_flat` | bool | Best season is consistent with a flat baseline |
| `chi2_flat` | float | χ² of the flat-model fit |
| `dof_flat` | int | Degrees of freedom of the flat model |
| `chi2_red_flat` | float | Reduced χ² of the flat model |
| `bump_flag` | bool | Trials-corrected bump detected |
| `bump_snr` | float | Peak bump significance |
| `t0_fit` | float | PSPL peak time (BJD − 2450000) |
| `u0_fit` | float | PSPL impact parameter |
| `tE_fit` | float | PSPL Einstein crossing time (days) |
| `chi2_red_pspl` | float | Reduced χ² of the PSPL fit |
| `t0_fit_fspl` | float | FSPL peak time (BJD − 2450000) |
| `u0_fit_fspl` | float | FSPL impact parameter |
| `tE_fit_fspl` | float | FSPL Einstein crossing time (days) |
| `rho_fit_fspl` | float | FSPL source radius ρ (Einstein radii) |
| `t_star_fspl` | float | Source crossing time t★ = ρ tE (days) |
| `chi2_red_fspl` | float | Reduced χ² of the FSPL fit |
| `chi2_fspl` | float | χ² of the FSPL fit |
| `n_points` | int | Number of points the fits used |
| `finite_source` | bool | Finite-source effects selected (see below) |
| `finite_source_delta_chi2` | float | χ²(PSPL) − χ²(FSPL) |
| `baseline_flux` | float | Median flux of the best season |
| `peak_flux` | float | Maximum flux of the best season |
| `baseline_mag` | float | Baseline flux as a magnitude |
| `peak_mag` | float | Peak flux as a magnitude |
| `source_mag` | float | FSPL source flux as a magnitude |
| `blend_mag` | float | FSPL blend flux as a magnitude (`NaN` for negative blending) |
| `period` | float | Strongest Lomb–Scargle period (days), reported whether or not the periodic veto fired |
| `period_amplitude` | float | Peak-to-peak amplitude of that period (mag) |
| `scan_candidate` | bool | Bump detected **and** season non-flat, or whole-curve bump detected |

## How the label is decided

```
scan_candidate = (bump_flag AND is_non_flat in the best season) OR whole-curve bump
any_veto       = periodic OR recurrent OR chromatic

not scan_candidate                       -> "flat"
scan_candidate AND any_veto              -> "variable star"
good fit AND timescale < ffp_tE_max      -> "ffp"
otherwise                                -> "microlensing"
```

"Good fit" means the reduced χ² of the selected model is below `good_fit_chi2`.
The timescale is tE of the selected model, or t★ when finite source is selected
with ρ > 1. A poor fit leaves the object as `microlensing`.

## Finite source

PSPL and FSPL are fitted to every candidate and both are reported. The FSPL
model is *selected* only when all three hold:

- Δχ² = χ²(PSPL) − χ²(FSPL) exceeds `fs_delta_chi2_min` (default ln N);
- the lens passes over or near the source: u₀ < `fs_u0_rho_max` × ρ;
- the source crossing is sampled: at least `fs_min_crossing_points` points
  within \|t − t₀\| < t★.

## Alerts

`write_alerts` writes one JSON file per non-flat object (`<objname>.json`) for
RogueDB:

```python
from pyAethra import write_alerts
paths = write_alerts(results, "alerts")
```

Fields, in order (`ALERT_FIELDS`):

| Field | Description |
|---|---|
| `alert_datetime` | UTC time the alert was written |
| `objname` | Object identifier |
| `lightcurve_file` | Light-curve location (`null` until available) |
| `RA`, `Dec` | Coordinates |
| `classification` | `Microlensing` (also for FFPs) or `Variable` |
| `FFP_candidate` | `true` for FFP candidates |
| `alert_notes` | Free-text notes |
| `alert_origin` | `pyAethra` |
| `t0` | FSPL peak time (full BJD) |
| `tE`, `u0` | FSPL timescale (days) and impact parameter |
| `rho` | FSPL ρ, or `null` unless finite source is selected |
| `chi2_red_fspl` | Reduced χ² of the FSPL fit |
| `delta_chi2_flat` | χ²(flat) − χ²(FSPL): evidence for the event over a constant light curve |
| `n_points` | Number of points the fit used |
| `peak_mag`, `baseline_mag` | Peak and baseline magnitude from the data |
| `source_mag`, `blend_mag` | FSPL source and blend magnitude (`blend_mag` is `null` for negative blending) |
| `period`, `period_amplitude` | Strongest period (days) and its peak-to-peak amplitude (mag) |

Missing or non-finite values are written as `null`.

## Package layout

```
src/pyAethra/
├── __init__.py          # public API
├── schema.py            # OUTPUT_COLUMNS
├── config.py            # load_config (YAML → config dict)
├── photometry.py        # mag ↔ flux conversion + input-column resolution
├── detection.py         # outlier / flatness / bump detection + recurrent veto
├── global_detection.py  # whole-light-curve bump detection for long events
├── variability.py       # non-flatness + Lomb–Scargle periodicity vetoes
├── achromatic.py        # multi-band achromaticity test
├── pspl.py              # PSPL and FSPL models + fitting (Fs/Fb solved analytically)
├── seasons.py           # survey-wide season numbering and per-season scan
├── pipeline.py          # run_pipeline_from_dataframe (main driver) + run_and_report
├── alerts.py            # RogueDB JSON alerts
├── io.py                # file loaders + load_and_run dispatcher
└── cli.py               # `pyAethra` console script
```

## License

MIT — see [LICENSE](https://github.com/rges-pit/pyAethra/blob/main/LICENSE).
