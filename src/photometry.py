"""Magnitude ↔ flux conversion and input-column resolution."""

import numpy as np

__all__ = ["mag_to_flux", "flux_to_mag", "resolve_flux"]


def mag_to_flux(mag, mag_err, zero_point=0.0):
    mag     = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)
    flux     = 10 ** (-0.4 * (mag - zero_point))
    flux_err = 0.4 * np.log(10) * flux * mag_err
    return flux, flux_err


def flux_to_mag(flux, zero_point=0.0):
    """Non-positive flux -> nan. Only ever used for reporting, never for stats."""
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = zero_point - 2.5 * np.log10(flux)
    return np.where(np.isfinite(mag), mag, np.nan)


def resolve_flux(df, config):
    """Return (flux, flux_err) for df.

    The flux columns are used when they are actually present in the DataFrame;
    otherwise the magnitude columns are converted. This lets one config serve a
    mixed dataset where some files carry flux and others only magnitudes.
    Flux input passes through untouched.
    """
    flux_col     = config.get("flux_col")
    flux_err_col = config.get("flux_err_col")
    mag_col      = config.get("mag_col", "mag")
    err_col      = config.get("err_col", "mag_err")

    if flux_col is not None and flux_col in df.columns:
        if flux_err_col is None:
            raise ValueError("config sets 'flux_col' but not 'flux_err_col'.")
        if flux_err_col not in df.columns:
            raise ValueError(
                f"Flux column '{flux_col}' is present but its error column "
                f"'{flux_err_col}' is not. Available columns: {list(df.columns)}"
            )
        return (df[flux_col].to_numpy(dtype=float),
                df[flux_err_col].to_numpy(dtype=float))

    if mag_col in df.columns:
        if err_col not in df.columns:
            raise ValueError(
                f"Magnitude column '{mag_col}' is present but its error column "
                f"'{err_col}' is not. Available columns: {list(df.columns)}"
            )
        return mag_to_flux(
            df[mag_col].to_numpy(dtype=float),
            df[err_col].to_numpy(dtype=float),
            zero_point=config.get("zero_point", 0.0),
        )

    wanted = f"'{mag_col}'/'{err_col}'"
    if flux_col is not None:
        wanted = f"'{flux_col}'/'{flux_err_col}' or " + wanted
    raise ValueError(
        f"No usable photometry columns found. Looked for {wanted}. "
        f"Available columns: {list(df.columns)}"
    )