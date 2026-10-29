"""RogueDB alert output: one JSON file per non-flat object.

Reads the results DataFrame from ``run_pipeline_from_dataframe`` and writes
the FSPL fit values only. Fields with no source yet are ``null``.
"""

import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["ALERT_FIELDS", "build_alert", "write_alerts"]

ALERT_FIELDS = [
    "alert_datetime",
    "objname",
    "lightcurve_file",
    "RA",
    "Dec",
    "classification",
    "FFP_candidate",
    "alert_notes",
    "alert_origin",
    "t0",
    "tE",
    "u0",
    "rho",              # null unless finite_source is selected
    "chi2_red_fspl",
    "delta_chi2_flat",
    "n_points",
    "peak_mag",
    "baseline_mag",
    "source_mag",
    "blend_mag",
    "period",
    "period_amplitude",
]

# pyAethra label -> RogueDB classification. "flat" is absent on purpose:
# flat objects do not produce an alert.
_CLASSIFICATION = {
    "microlensing": "Microlensing",
    "ffp": "Microlensing",
    "variable star": "Variable",
}

# The pipeline reports t0 as (raw time - 2450000); add it back for full BJD.
_T0_OFFSET = 2450000.0


def _num(x):
    """Plain float, or None for NaN/inf/missing (NaN is not valid JSON)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def build_alert(row):
    """Build one alert dict from a results row. Returns None for flat objects."""
    label = row["label"]
    if label not in _CLASSIFICATION:
        return None

    t0 = _num(row["t0_fit_fspl"])
    chi2_flat, chi2_fspl = _num(row["chi2_flat"]), _num(row["chi2_fspl"])
    n_points = _num(row["n_points"])
    return {
        "alert_datetime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "objname": str(row["name"]),
        "lightcurve_file": None,
        "RA": _num(row["RA"]),
        "Dec": _num(row["Dec"]),
        "classification": _CLASSIFICATION[label],
        "FFP_candidate": label == "ffp",
        "alert_notes": "howdy",
        "alert_origin": "pyAethra",
        "t0": None if t0 is None else t0 + _T0_OFFSET,
        "tE": _num(row["tE_fit_fspl"]),
        "u0": _num(row["u0_fit_fspl"]),
        # rho is only meaningful when the finite-source model was selected.
        "rho": _num(row["rho_fit_fspl"]) if bool(row["finite_source"]) else None,
        "chi2_red_fspl": _num(row["chi2_red_fspl"]),
        # Evidence for the event over a constant light curve (same season data).
        "delta_chi2_flat": (None if chi2_flat is None or chi2_fspl is None
                            else chi2_flat - chi2_fspl),
        "n_points": None if n_points is None else int(n_points),
        "peak_mag": _num(row["peak_mag"]),
        "baseline_mag": _num(row["baseline_mag"]),
        "source_mag": _num(row["source_mag"]),
        "blend_mag": _num(row["blend_mag"]),
        # Strongest Lomb-Scargle period (days) and its peak-to-peak amplitude
        # (mag), reported whether or not the periodic veto fired.
        "period": _num(row["period"]),
        "period_amplitude": _num(row["period_amplitude"]),
    }


def _safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def write_alerts(results_df, out_dir):
    """Write ``<Target star ID>.json`` for every non-flat row. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for _, row in results_df.iterrows():
        alert = build_alert(row)
        if alert is None:
            continue
        path = out_dir / f"{_safe_filename(alert['objname'])}.json"
        path.write_text(json.dumps(alert, indent=2))
        written.append(path)
    return written