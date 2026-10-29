"""Whole-lightcurve bump detection for long events.

The per-season scan in :mod:`pyAethra.seasons` is near-optimal for short events
but structurally cannot see a long one: an event with tE of a few hundred days
is chopped at every seasonal gap, so no single season contains a rise, a peak
and a fall, and the fitted tE can never exceed the season length.

This module scans the *whole* curve at long window widths only. It deliberately
does not go short: sub-day events are the per-season scan's job, and searching
short windows here would only add trials (raising the threshold for everything)
while duplicating a detector that is already better at them.

The statistic is the same one :func:`pyAethra.detection.detect_bump` uses --
error-normalised excess summed over a window and divided by the square root of
the number of points in it, so it is ~N(0,1) under the flat null -- and the
threshold is the same trials-corrected sqrt(2 ln n_trials) + margin. That means
``global_snr`` and ``bump_snr`` are on the same scale and can be compared
directly.

Two things differ from the per-season scan, both forced by the gaps:

* Windows are defined in **time**, not in number of points. A fixed point count
  spans a few days inside a season and several years across one.
* Observations are binned to nights first. A 100-day window does not need
  15-minute sampling, and binning turns ~10^5 points into ~10^3 with no loss of
  sensitivity to anything this module is looking for.
"""

import numpy as np

__all__ = ["bin_to_nights", "detect_bump_global"]


def bin_to_nights(time, flux, flux_err, bin_days=1.0):
    """Inverse-variance-weighted binning onto a regular time grid.

    Only bins containing observations are returned, so seasonal gaps produce no
    empty bins and never enter the statistics.

    Returns
    -------
    (bin_time, bin_flux, bin_err, bin_n) : tuple of ndarray
        Flux-weighted mean time, weighted mean flux, error on that mean, and
        the number of raw points in each bin.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    valid = (np.isfinite(time) & np.isfinite(flux)
             & np.isfinite(flux_err) & (flux_err > 0))
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]
    if len(time) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.array([], dtype=int)

    if bin_days is None or bin_days <= 0:
        return time, flux, flux_err, np.ones(len(time), dtype=int)

    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    idx = np.floor((time - time[0]) / bin_days).astype(np.int64)
    _, inverse, counts = np.unique(idx, return_inverse=True, return_counts=True)
    n_bins = len(counts)

    w = 1.0 / flux_err**2
    wsum = np.bincount(inverse, weights=w,        minlength=n_bins)
    fsum = np.bincount(inverse, weights=w * flux, minlength=n_bins)
    tsum = np.bincount(inverse, weights=w * time, minlength=n_bins)

    good = wsum > 0
    bin_time = tsum[good] / wsum[good]
    bin_flux = fsum[good] / wsum[good]
    bin_err = 1.0 / np.sqrt(wsum[good])
    return bin_time, bin_flux, bin_err, counts[good]


def _windowed_significance(bin_time, z, width_days, min_bins):
    """Sum z over a centred time window, normalised to ~N(0,1).

    Windows are in days, so a gap contributes no points rather than silently
    stretching the window across it.
    """
    half = 0.5 * width_days
    lo = np.searchsorted(bin_time, bin_time - half, side="left")
    hi = np.searchsorted(bin_time, bin_time + half, side="right")

    csum = np.concatenate(([0.0], np.cumsum(z)))
    total = csum[hi] - csum[lo]
    counts = (hi - lo).astype(float)

    sig = np.full(len(bin_time), np.nan)
    enough = counts >= min_bins
    sig[enough] = total[enough] / np.sqrt(counts[enough])
    return sig, counts


def _excursion_extent(bin_time, z, peak_index, sigma=1.0, max_gap_days=None):
    """Walk out from the peak while bins stay above `sigma`, and return the
    time span covered. Used as the event duration for downstream masking.

    A run is allowed to bridge a seasonal gap: the walk stops on the first bin
    that drops below the threshold, not on the first large time step, because a
    long event that spans a gap is precisely the case this module exists for.
    """
    n = len(z)
    if not (0 <= peak_index < n):
        return np.nan, np.nan, np.nan

    i = peak_index
    while i - 1 >= 0 and z[i - 1] > sigma:
        if max_gap_days is not None and (bin_time[i] - bin_time[i - 1]) > max_gap_days:
            break
        i -= 1
    j = peak_index
    while j + 1 < n and z[j + 1] > sigma:
        if max_gap_days is not None and (bin_time[j + 1] - bin_time[j]) > max_gap_days:
            break
        j += 1

    return bin_time[i], bin_time[j], float(bin_time[j] - bin_time[i])


def detect_bump_global(time, flux, flux_err,
                       bin_days=1.0,
                       window_days=(20.0, 50.0, 100.0, 200.0, 400.0),
                       min_bins=5,
                       threshold_margin=1.0,
                       extent_sigma=1.0):
    """Scan the whole light curve for a long-duration bump.

    Parameters
    ----------
    time, flux, flux_err : array_like
        The full curve in the primary band, in flux (not magnitudes).
    bin_days : float
        Bin width before scanning. Must be well below the shortest window.
        Set to None or 0 to scan unbinned points.
    window_days : sequence of float
        Window widths to search. These are deliberately long -- short events
        belong to the per-season scan.
    min_bins : int
        Minimum bins inside a window for it to be scored.
    threshold_margin : float
        Added to the trials-corrected threshold, as in
        :func:`pyAethra.detection.detect_bump`.
    extent_sigma : float
        Bins above this many sigma count as part of the excursion when
        measuring its duration.

    Returns
    -------
    dict
        ``global_flag``
            Whether the trials-corrected threshold was cleared.
        ``global_snr``
            Peak significance, on the same scale as ``bump_snr``.
        ``global_window``
            Window width that produced the peak.
        ``global_t0_raw``
            Time of the peak, in the input time system.
        ``global_duration_days``
            Time span of the excursion around the peak. This is the duration
            estimate downstream masking should use; it is measured across the
            whole curve and so remains correct for events that span gaps.
        ``global_t_start`` / ``global_t_end``
            Ends of that excursion.
        ``n_bins``, ``global_threshold``
            Diagnostics.
    """
    empty = {
        "global_flag": False, "global_snr": np.nan, "global_window": np.nan,
        "global_t0_raw": np.nan, "global_duration_days": np.nan,
        "global_t_start": np.nan, "global_t_end": np.nan,
        "n_bins": 0, "global_threshold": np.nan,
    }

    bin_time, bin_flux, bin_err, _ = bin_to_nights(time, flux, flux_err, bin_days)
    n = len(bin_time)
    if n < min_bins:
        return empty

    span = float(bin_time[-1] - bin_time[0])
    if not np.isfinite(span) or span <= 0:
        return empty

    # Median baseline: unbiased for flat data and barely moved by one localised
    # excursion, the same reasoning as in detect_bump. It does degrade if an
    # event covers more than half the observed span, which no real event does
    # over a multi-year baseline.
    baseline = np.median(bin_flux)
    z = (bin_flux - baseline) / bin_err

    best_sig = -np.inf
    best_window = np.nan
    best_index = -1
    n_trials = 0.0

    for width in window_days:
        if width > span:
            continue
        sig, _ = _windowed_significance(bin_time, z, width, min_bins)
        if not np.any(np.isfinite(sig)):
            continue

        # Roughly independent window positions at this scale.
        n_trials += max(span / width, 1.0)

        index = int(np.nanargmax(sig))
        if np.isfinite(sig[index]) and sig[index] > best_sig:
            best_sig = float(sig[index])
            best_window = float(width)
            best_index = index

    if not np.isfinite(best_sig) or n_trials < 1:
        return empty

    threshold = np.sqrt(2.0 * np.log(n_trials)) + threshold_margin
    t_start, t_end, duration = _excursion_extent(
        bin_time, z, best_index, sigma=extent_sigma)

    return {
        "global_flag": bool(best_sig > threshold),
        "global_snr": best_sig,
        "global_window": best_window,
        "global_t0_raw": float(bin_time[best_index]),
        "global_duration_days": duration,
        "global_t_start": t_start,
        "global_t_end": t_end,
        "n_bins": n,
        "global_threshold": float(threshold),
    }