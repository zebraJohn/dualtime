"""Hourly subsampling of the 10-minute weather series, applied AT LOAD TIME.

WHY THIS EXISTS
---------------
PROGRESS.md section 7 names "change the dataset" as the only avenue left, and
weather is the only non-ETT dataset on disk. Training on the full 10-minute
series costs ~19 h for the rho guard; an hourly version costs ~3-4 h.

WHY AT LOAD TIME AND NOT AS A NEW CSV
-------------------------------------
A pre-generated weather1h.csv would be a second copy of the data on disk with no
record of how it was derived, and it would silently drift if anyone edited one
and not the other. Keeping the transform in code means the derivation is
reviewable and the original file stays the single source of truth.

WHY resample().first() AND NOT iloc[::6] OR .mean()
--------------------------------------------------
The source file is NOT a clean 10-minute grid. It contains one duplicated
timestamp (2020-05-12 06:00:00 appears twice) and one 90-minute hole
(2020-05-29 09:30 -> 11:10). Measured consequences:

    iloc[::6]                 8783 rows, 0 dupes, 1x50min + 1x2:30 gaps
    date.dt.minute == 0       8783 rows, 1 DUPE, 1x0 + 1x3:00 gaps
    resample('1h').first()    8785 rows, 0 dupes, NO irregular gaps   <-- this

Index-based slicing drifts at both defects; time-based binning is immune.

`.first()` (take the earliest sample in each hour) is deliberate over `.mean()`.
Averaging the six 10-minute values would suppress sub-hourly noise and raise the
signal-to-noise ratio of the target -- and the entire hypothesis under test is
about noise versus structure. Downsampling by averaging could lower rho for a
reason that is an artifact of the smoothing rather than of weather being
structurally richer. Subsampling preserves the noise level, which is what makes
the rho comparable to ETTh1's.

WHAT THIS CHANGES ABOUT THE TASK -- read before interpreting a result
---------------------------------------------------------------------
The run keeps seq_len=720 and pred_len in {96,192,336,720}; only the sampling
interval changes. So the numbers no longer mean what they mean elsewhere:

                      10-minute data        1-hour data (this)
    pred_len=96       next 16 hours         next 4 DAYS
    pred_len=720      next 5 days           next 30 days
    seq_len=720       5 days of context     30 days of context

This is NOT the weather benchmark and cannot be compared with any published
weather number. It is a cheap screen on a non-ETTh1 dataset. The tags, log
directory and --data value are all named weather1h to keep that visible.
"""
__all__ = ['hourly_subsample', 'Dataset_Weather1h']

import pandas as pd

from data_provider.data_loader import Dataset_Custom_weather

# One sample per hour from a 10-minute series.
RESAMPLE_RULE = '1h'


def hourly_subsample(df, date_col='date'):
    """Take the earliest sample in each 1-hour bin.

    Returns a new frame with an hourly grid and `date_col` as a column.
    Row count goes 52696 -> 8784.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    channels = [c for c in out.columns if c != date_col]
    out = (out.set_index(date_col)
              .resample(RESAMPLE_RULE)
              .first()
              .reset_index())

    # resample() emits a row for EVERY hour on the grid, including hours the
    # source has no data for, and .first() on an empty bin is all-NaN. The
    # source has a 90-minute hole (2020-05-29 09:30 -> 11:10), so exactly one
    # such row appears, at 2020-05-29 10:00.
    #
    # This was a real bug, not a theoretical one: leaving the row in made
    # StandardScaler.fit compute NaN mean/std, which turned EVERY value in the
    # dataset into NaN and produced 21 runs of `mse:nan` from epoch 1.
    #
    # Drop only rows that are entirely empty, and refuse to drop anything else.
    # A partial NaN would mean a different problem and must not be papered over.
    empty_bin = out[channels].isna().all(axis=1)
    out = out[~empty_bin].reset_index(drop=True)

    partial = int(out[channels].isna().sum().sum())
    if partial:
        bad = list(out.loc[out[channels].isna().any(axis=1), date_col].astype(str))
        raise ValueError(
            f'{partial} NaN cells remain after dropping empty resample bins. The '
            f'source has 0 NaN, so this is a new defect, not a gap in coverage. '
            f'Rows: {bad[:5]}')
    return out


class Dataset_Weather1h(Dataset_Custom_weather):
    """Dataset_Custom_weather on the 10-minute file, resampled to hourly.

    Only the CSV read is changed; column reordering, the 7:1:2 split, scaling
    and time features are all the parent's, unchanged.
    """

    def _read_raw(self):
        return hourly_subsample(super()._read_raw())
