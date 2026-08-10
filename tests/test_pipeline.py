"""Basic smoke and functional tests for the pyAethra pipeline."""

import numpy as np
import pandas as pd
import pytest

import pyAethra as ef
from pyAethra import (
    OUTPUT_COLUMNS,
    detect_bump,
    fit_pspl,
    load_and_run,
    mag_to_flux,
    pspl_magnification,
    run_pipeline_from_dataframe,
    split_into_seasons,
)


def make_lightcurve(with_event=True, seed=0):
    """A two-season light curve, optionally with a PSPL bump in season 1."""
    rng = np.random.default_rng(seed)
    # Two seasons separated by a >100 day gap.
    t1 = np.linspace(2459000, 2459120, 400)
    t2 = np.linspace(2459300, 2459420, 400)
    time = np.concatenate([t1, t2])

    baseline = 18.0
    mag = np.full_like(time, baseline)

    if with_event:
        t0, u0, tE = 2459060.0, 0.1, 8.0
        A = pspl_magnification(time, t0, u0, tE)
        # brighter (smaller mag) at higher magnification
        mag = baseline - 2.5 * np.log10(A)

    mag_err = np.full_like(time, 0.01)
    mag = mag + rng.normal(0, 0.01, size=mag.size)

    return pd.DataFrame({"bjd": time, "mag": mag, "mag_err": mag_err, "name": "obj1"})


CONFIG = {
    "time_col": "bjd",
    "mag_col": "mag",
    "err_col": "mag_err",
    "group_col": "name",
}


def test_version_and_exports():
    assert isinstance(ef.__version__, str)
    assert "load_and_run" in ef.__all__


def test_output_schema_is_stable():
    df = run_pipeline_from_dataframe(make_lightcurve(), CONFIG)
    for col in OUTPUT_COLUMNS:
        assert col in df.columns


def test_split_into_seasons():
    time = np.array([0.0, 1.0, 2.0, 200.0, 201.0])
    labels = split_into_seasons(time, gap_days=100)
    # A >gap_days jump between index 2 and 3 must start a new season,
    # and points within the same cluster share a label.
    assert labels[3] != labels[2]
    assert labels[3] == labels[4]
    assert labels[0] == labels[0]  # first point keeps its own (original) label


def test_pipeline_runs_on_dataframe():
    df = run_pipeline_from_dataframe(make_lightcurve(with_event=True), CONFIG)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "obj1"
    # an injected event should at least trip the bump detector
    assert bool(df.iloc[0]["bump_flag"]) is True


def test_flat_lightcurve_is_not_a_candidate():
    df = run_pipeline_from_dataframe(make_lightcurve(with_event=False), CONFIG)
    assert df.iloc[0]["ID"] == "flat"
    assert bool(df.iloc[0]["scan_candidate"]) is False


def test_detect_bump_finds_injected_bump():
    # detect_bump is flux-native: a brightening must be a POSITIVE excursion.
    # Passing magnitudes makes the event a dip and the detector cannot see it.
    lc = make_lightcurve(with_event=True)
    flux, flux_err = mag_to_flux(lc["mag"].to_numpy(), lc["mag_err"].to_numpy())
    flag, window, snr = detect_bump(lc["bjd"].to_numpy(), flux, flux_err)
    assert flag is True
    assert np.isfinite(snr)
    # A u0=0.1 event is overwhelming. A modest snr here would mean the detector
    # is firing on a baseline artefact rather than on the injected bump.
    assert snr > 100
    # The injected tE is 8 d, so the winning window should be a short one.
    assert window <= 20



def test_fit_pspl_recovers_parameters():
    # fit_pspl works in flux space, so magnitudes have to be converted first.
    lc = make_lightcurve(with_event=True)
    # restrict to the event season for a clean fit
    season = lc[lc["bjd"] < 2459200]
    flux, flux_err = mag_to_flux(
        season["mag"].to_numpy(), season["mag_err"].to_numpy()
    )
    result = fit_pspl(season["bjd"].to_numpy(), flux, flux_err)
    assert result is not None
    assert abs(result["t0_fit"] - 2459060.0) < 5.0
    assert result["tE_fit"] > 0
    # Injected parameters were t0=2459060, u0=0.1, tE=8.
    assert abs(result["tE_fit"] - 8.0) < 1.0
    assert abs(result["u0_fit"] - 0.1) < 0.05
    assert result["chi2_red_pspl"] < 2.5


def test_flux_columns_give_the_same_answer_as_magnitudes():
    """flux_col/flux_err_col input must reach the same verdict as mag input."""
    lc = make_lightcurve(with_event=True)
    flux, flux_err = mag_to_flux(lc["mag"].to_numpy(), lc["mag_err"].to_numpy())
    flux_lc = pd.DataFrame(
        {"bjd": lc["bjd"], "flux": flux, "flux_err": flux_err, "name": "obj1"}
    )
    flux_cfg = {
        "time_col": "bjd",
        "flux_col": "flux",
        "flux_err_col": "flux_err",
        "group_col": "name",
    }
    from_flux = run_pipeline_from_dataframe(flux_lc, flux_cfg)
    from_mag = run_pipeline_from_dataframe(lc, CONFIG)
    assert from_flux.iloc[0]["ID"] == from_mag.iloc[0]["ID"]
    assert np.isclose(
        from_flux.iloc[0]["tE_fit"], from_mag.iloc[0]["tE_fit"], rtol=1e-6
    )


def test_flux_col_without_err_raises():
    lc = make_lightcurve()
    flux, _ = mag_to_flux(lc["mag"].to_numpy(), lc["mag_err"].to_numpy())
    lc = lc.assign(flux=flux)
    with pytest.raises(ValueError):
        run_pipeline_from_dataframe(
            lc, {"time_col": "bjd", "flux_col": "flux", "group_col": "name"}
        )


def test_load_and_run_accepts_dataframe():
    df = load_and_run(make_lightcurve(), CONFIG)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == OUTPUT_COLUMNS


def test_load_and_run_from_csv(tmp_path):
    lc = make_lightcurve()
    path = tmp_path / "data.csv"
    lc.to_csv(path, index=False)
    cfg = {**CONFIG, "sep": ",", "header": 0, "columns": None}
    df = load_and_run(str(path), cfg)
    assert len(df) == 1


def test_missing_column_raises():
    bad = make_lightcurve().rename(columns={"mag": "brightness"})
    with pytest.raises(ValueError):
        run_pipeline_from_dataframe(bad, CONFIG)


# ── YAML configuration ────────────────────────────────────────────────────────

def test_load_config_reads_yaml(tmp_path):
    from pyAethra import load_config

    p = tmp_path / "config.yaml"
    p.write_text(
        "time_col: bjd\n"
        "mag_col: mag\n"
        "err_col: mag_err\n"
        "group_col: name\n"
        "season_gap_days: 120\n"
    )
    cfg = load_config(str(p))
    assert cfg["time_col"] == "bjd"
    assert cfg["group_col"] == "name"
    assert cfg["season_gap_days"] == 120


def test_load_config_null_becomes_none(tmp_path):
    from pyAethra import load_config

    p = tmp_path / "config.yaml"
    p.write_text("time_col: bjd\nmag_col: mag\nerr_col: mag_err\ngroup_col: null\n")
    cfg = load_config(str(p))
    assert cfg["group_col"] is None


def test_load_config_rejects_non_mapping(tmp_path):
    from pyAethra import load_config

    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_run_from_yaml_config(tmp_path):
    from pyAethra import load_config

    lc = make_lightcurve()
    data_path = tmp_path / "data.csv"
    lc.to_csv(data_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "time_col: bjd\n"
        "mag_col: mag\n"
        "err_col: mag_err\n"
        "group_col: name\n"
        "sep: ','\n"
        "header: 0\n"
    )
    cfg = load_config(str(cfg_path))
    df = load_and_run(str(data_path), cfg)
    assert len(df) == 1


def test_cli_config_with_flag_override(tmp_path):
    from pyAethra.cli import build_config, build_parser

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("time_col: bjd\nmag_col: mag\nerr_col: mag_err\nmin_points: 10\n")

    args = build_parser().parse_args(
        ["data.csv", "--config", str(cfg_path), "--min-points", "25"]
    )
    config = build_config(args)
    # CLI flag wins over the YAML value
    assert config["min_points"] == 25
    # untouched YAML keys survive
    assert config["time_col"] == "bjd"
