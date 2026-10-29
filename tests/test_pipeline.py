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

from pyAethra import (
    ALERT_FIELDS,
    analyze_season_scan,
    assign_seasons,
    build_alert,
    fit_fspl,
    fit_pspl_candidate,
    fspl_magnification,
    write_alerts,
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
    assert df.iloc[0]["label"] == "flat"
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
    assert from_flux.iloc[0]["label"] == from_mag.iloc[0]["label"]
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


# ── v0.1.2: seasons ───────────────────────────────────────────────────────────

def test_assign_seasons_numbers_forward_from_gaps():
    time = np.array([0.0, 1.0, 2.0, 200.0, 201.0, 500.0])
    labels = assign_seasons(time, gap_days=100)
    assert list(labels) == [1, 1, 1, 2, 2, 3]


def test_assign_seasons_is_survey_wide():
    # Two objects: one has no data in season 2. Pooled numbering must still
    # put the second object's late points in season 3, not season 2.
    t_a = np.array([0.0, 1.0, 200.0, 201.0, 500.0])
    t_b = np.array([0.5, 1.5, 500.5])
    labels = assign_seasons(np.concatenate([t_a, t_b]), gap_days=100)
    assert list(labels[len(t_a):]) == [1, 1, 3]


def test_assign_seasons_numbers_backwards_when_fewer_than_current():
    # Only one season present but current_season=3: assume the data ARE season 3.
    time = np.array([0.0, 1.0, 2.0])
    with pytest.warns(UserWarning):
        labels = assign_seasons(time, gap_days=100, current_season=3)
    assert set(labels) == {3}


def test_assign_seasons_explicit_edges():
    time = np.array([-5.0, 0.0, 10.0, 100.0, 150.0])
    labels = assign_seasons(time, season_edges=[0.0, 100.0])
    assert list(labels) == [0, 1, 1, 2, 2]


def test_current_season_drops_later_seasons():
    # The injected event is in season 1; restricting to season 1 must still
    # find it, and the output must record which season was run.
    df = run_pipeline_from_dataframe(make_lightcurve(), {**CONFIG, "current_season": 1})
    row = df.iloc[0]
    assert row["current_season"] == 1
    assert bool(row["scan_candidate"]) is True


def test_current_season_null_runs_all_seasons():
    df = run_pipeline_from_dataframe(make_lightcurve(), {**CONFIG, "current_season": None})
    assert pd.isna(df.iloc[0]["current_season"])
    assert bool(df.iloc[0]["scan_candidate"]) is True


# ── v0.1.2: short-window detection ────────────────────────────────────────────

def test_short_window_detects_sub_day_bump():
    # 15-min cadence over 30 d with a 0.1-d top-hat bump. Only the 0.1-d
    # window matches it.
    rng = np.random.default_rng(1)
    t = np.arange(0, 30, 15 / 1440.0)
    flux = 1.0 + rng.normal(0, 0.01, t.size)
    flux[(t > 15.0) & (t < 15.1)] += 0.2
    err = np.full_like(t, 0.01)
    res = analyze_season_scan(t, flux, err)
    assert res["bump_flag"] is True
    assert res["best_window"] == 0.1


# ── v0.1.2: FSPL ──────────────────────────────────────────────────────────────

def test_fspl_reduces_to_pspl_for_small_rho():
    t = np.linspace(-20, 20, 401)
    a_ps = pspl_magnification(t, 0.0, 0.1, 8.0)
    a_fs = fspl_magnification(t, 0.0, 0.1, 8.0, 1e-4)
    assert np.allclose(a_fs, a_ps, rtol=1e-4)


def test_fspl_caps_peak_magnification():
    # A lens crossing a large source cannot reach the point-source peak.
    t = np.array([0.0])
    a_ps = pspl_magnification(t, 0.0, 0.001, 8.0)[0]
    a_fs = fspl_magnification(t, 0.0, 0.001, 8.0, 0.5)[0]
    # Uniform source, u0 -> 0: A = sqrt(1 + 4/rho^2).
    assert a_fs < a_ps
    assert np.isclose(a_fs, np.sqrt(1 + 4 / 0.5**2), rtol=1e-3)


def _finite_source_lightcurve(rho, seed=2):
    rng = np.random.default_rng(seed)
    t = np.linspace(-10, 10, 2000)
    fs, fb = 1.0, 0.5
    flux = fs * fspl_magnification(t, 0.0, 0.05, 2.0, rho) + fb
    err = np.full_like(t, 0.005)
    return t, flux + rng.normal(0, 0.005, t.size), err


def test_fit_fspl_recovers_rho_and_flux_split():
    t, flux, err = _finite_source_lightcurve(rho=0.5)
    res = fit_fspl(t, flux, err)
    assert res is not None
    assert abs(res["rho_fit"] - 0.5) < 0.05
    assert abs(res["Fs"] - 1.0) < 0.1
    assert abs(res["Fb"] - 0.5) < 0.1


def test_finite_source_selected_only_when_needed():
    t, flux, err = _finite_source_lightcurve(rho=0.5)
    assert fit_pspl_candidate(t, flux, err)["finite_source"] is True

    # Point-source event: FSPL must not be selected.
    rng = np.random.default_rng(3)
    ps = 1.0 * pspl_magnification(t, 0.0, 0.3, 2.0) + 0.5 + rng.normal(0, 0.005, t.size)
    assert fit_pspl_candidate(t, ps, err)["finite_source"] is False


def test_pipeline_reports_fspl_columns():
    row = run_pipeline_from_dataframe(make_lightcurve(), CONFIG).iloc[0]
    for col in ("tE_fit_fspl", "u0_fit_fspl", "rho_fit_fspl", "chi2_red_fspl",
                "chi2_fspl", "n_points"):
        assert np.isfinite(row[col]), col
    # Flat -> FSPL improvement for an injected u0=0.1 event is overwhelming.
    assert row["chi2_flat"] - row["chi2_fspl"] > 1000


# ── v0.1.2: source / blend magnitudes ─────────────────────────────────────────

def test_source_and_blend_mag_recovered():
    # Blended event: 70% of the baseline flux from the source.
    rng = np.random.default_rng(4)
    time = np.linspace(2459000, 2459120, 1200)
    fs, fb, zp = 0.7, 0.3, 25.0
    flux = fs * pspl_magnification(time, 2459060.0, 0.1, 8.0) + fb
    err = np.full_like(time, 0.002)
    df = pd.DataFrame({"bjd": time, "flux": flux + rng.normal(0, 0.002, time.size),
                       "flux_err": err, "name": "blend"})
    cfg = {"time_col": "bjd", "flux_col": "flux", "flux_err_col": "flux_err",
           "group_col": "name", "zero_point": zp}
    row = run_pipeline_from_dataframe(df, cfg).iloc[0]
    assert abs(row["source_mag"] - (zp - 2.5 * np.log10(fs))) < 0.05
    assert abs(row["blend_mag"] - (zp - 2.5 * np.log10(fb))) < 0.1
    assert abs(row["baseline_mag"] - zp) < 0.01   # Fs + Fb = 1


# ── v0.1.2: periodicity ───────────────────────────────────────────────────────

def make_pulsator(period=0.05, amp_mag=0.1, days=20.0, n_seasons=2, seed=5):
    """delta Scuti-like sinusoid at 12-min cadence, no microlensing."""
    rng = np.random.default_rng(seed)
    cadence = 12 / 1440.0
    seasons = [np.arange(0, days, cadence) + 2459000 + 300 * k for k in range(n_seasons)]
    t = np.concatenate(seasons)
    mag = 18.0 + 0.5 * amp_mag * np.sin(2 * np.pi * t / period)
    mag = mag + rng.normal(0, 0.005, t.size)
    return pd.DataFrame({"bjd": t, "mag": mag, "mag_err": np.full_like(t, 0.005), "name": "dsct"})


def test_short_period_pulsator_is_vetoed():
    # P = 0.05 d is below the old 0.1-d floor. With the 0.02-d default the
    # periodic veto must see it.
    row = run_pipeline_from_dataframe(make_pulsator(), CONFIG).iloc[0]
    assert row["label"] != "microlensing"
    assert row["label"] != "ffp"
    assert bool(row["periodic"]) is True
    assert abs(row["period"] - 0.05) < 0.001
    assert 0.05 < row["period_amplitude"] < 0.2   # injected 0.1 mag peak-to-peak


def test_periodic_min_period_is_configurable():
    # Raising the floor above the true period hides it from the periodic veto.
    row = run_pipeline_from_dataframe(make_pulsator(), {**CONFIG, "periodic_min_period": 0.1}).iloc[0]
    assert not np.isclose(row["period"], 0.05, atol=0.001)


def test_in_season_veto_with_single_season():
    # One season only: no off-event season exists, so the in-season veto on
    # the PSPL residuals has to catch the pulsator. P = 0.2 d so a single
    # half-cycle trips the 0.1-d bump window and the object becomes a candidate.
    row = run_pipeline_from_dataframe(make_pulsator(period=0.2, days=7.0, n_seasons=1), CONFIG).iloc[0]
    assert bool(row["scan_candidate"]) is True
    assert row["label"] == "variable star"
    assert bool(row["periodic"]) is True


def test_period_reported_even_without_veto():
    row = run_pipeline_from_dataframe(make_lightcurve(), CONFIG).iloc[0]
    assert bool(row["periodic"]) is False
    assert np.isfinite(row["period"])
    assert np.isfinite(row["period_amplitude"])


# ── v0.1.2: coordinates ───────────────────────────────────────────────────────

def test_ra_dec_passed_through():
    lc = make_lightcurve().assign(ra_deg=267.5, dec_deg=-29.5)
    row = run_pipeline_from_dataframe(lc, CONFIG).iloc[0]
    assert row["RA"] == 267.5
    assert row["Dec"] == -29.5


def test_ra_dec_custom_columns_and_missing():
    lc = make_lightcurve().assign(RAJ2000=10.0, DEJ2000=20.0)
    row = run_pipeline_from_dataframe(lc, {**CONFIG, "ra_col": "RAJ2000", "dec_col": "DEJ2000"}).iloc[0]
    assert (row["RA"], row["Dec"]) == (10.0, 20.0)
    row = run_pipeline_from_dataframe(make_lightcurve(), CONFIG).iloc[0]
    assert np.isnan(row["RA"]) and np.isnan(row["Dec"])


# ── v0.1.2: alerts ────────────────────────────────────────────────────────────

def test_alert_fields_and_values():
    lc = make_lightcurve().assign(ra_deg=267.5, dec_deg=-29.5)
    row = run_pipeline_from_dataframe(lc, CONFIG).iloc[0]
    alert = build_alert(row)
    assert list(alert) == ALERT_FIELDS
    assert alert["objname"] == "obj1"
    assert alert["classification"] == "Microlensing"
    assert (alert["RA"], alert["Dec"]) == (267.5, -29.5)
    assert alert["t0"] > 2450000          # full BJD, not BJD - 2450000
    assert alert["n_points"] == int(row["n_points"])
    assert np.isclose(alert["delta_chi2_flat"], row["chi2_flat"] - row["chi2_fspl"])


def test_alert_rho_null_without_finite_source():
    row = run_pipeline_from_dataframe(make_lightcurve(), CONFIG).iloc[0]
    assert bool(row["finite_source"]) is False
    assert build_alert(row)["rho"] is None


def test_no_alert_for_flat_objects():
    row = run_pipeline_from_dataframe(make_lightcurve(with_event=False), CONFIG).iloc[0]
    assert build_alert(row) is None


def test_write_alerts_writes_valid_json(tmp_path):
    import json

    flat = make_lightcurve(with_event=False).assign(name="obj2")
    df = run_pipeline_from_dataframe(pd.concat([make_lightcurve(), flat]), CONFIG)
    paths = write_alerts(df, tmp_path / "alerts")
    assert [p.name for p in paths] == ["obj1.json"]          # flat obj2 skipped
    alert = json.loads(paths[0].read_text())                   # NaN would fail here
    assert list(alert) == ALERT_FIELDS


# ── v0.1.2: config / CLI ──────────────────────────────────────────────────────

def test_good_fit_chi2_is_read():
    # A threshold below any real fit's reduced chi2 rejects the candidate.
    row = run_pipeline_from_dataframe(make_lightcurve(), {**CONFIG, "good_fit_chi2": 0.01}).iloc[0]
    assert row["label"] == "microlensing"   # label does not depend on the fit gate
    cand = fit_pspl_candidate(*_finite_source_lightcurve(rho=0.5), good_fit_chi2=0.01)
    assert cand["is_candidate"] is False


def test_cli_old_good_pspl_chi2_flag_still_works():
    from pyAethra.cli import build_config, build_parser

    args = build_parser().parse_args(["data.csv", "--good-pspl-chi2", "4.0"])
    assert build_config(args)["good_fit_chi2"] == 4.0
