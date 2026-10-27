# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Tariff periods P1 to P6 of Spain's six-period access tariffs (6.1TD to 6.4TD).

Mainland calendar of CNMC Circular 3/2020 (BOE-A-2020-1066, art. 7.2), in local
time: the peak bands 9–14 h and 18–22 h and the shoulder bands 8–9, 14–18 and
22–24 h of a weekday fall into a season-dependent pair of periods, 0–8 h is P6,
and Saturdays, Sundays and fixed-date national holidays are P6 all day. The
revised draft royal decree of 26 October 2026 (DT1) phases in the hourly
correlation floor for data centers in the automatic load-reduction scheme by
these periods.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

TZ = "Europe/Madrid"
# season -> (period of the peak bands, period of the shoulder bands)
SEASON_PERIODS: dict[str, tuple[str, str]] = {
    "high": ("P1", "P2"),
    "mid_high": ("P2", "P3"),
    "mid": ("P3", "P4"),
    "low": ("P4", "P5"),
}
SEASON_OF_MONTH: dict[int, str] = {
    1: "high",
    2: "high",
    7: "high",
    12: "high",
    3: "mid_high",
    11: "mid_high",
    6: "mid",
    8: "mid",
    9: "mid",
    4: "low",
    5: "low",
    10: "low",
}
PEAK_HOURS = frozenset(range(9, 14)) | frozenset(range(18, 22))
NIGHT_HOURS = frozenset(range(0, 8))
# national holidays with a fixed date (the Circular excludes movable and
# substitutable ones), plus 6 January, which it names explicitly
HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 6),
    (5, 1),
    (8, 15),
    (10, 12),
    (11, 1),
    (12, 6),
    (12, 8),
    (12, 25),
)


def national_holidays(year: int) -> set[dt.date]:
    """Fixed-date national holidays of ``year`` that count as P6 days."""
    return {dt.date(year, m, d) for m, d in HOLIDAYS}


def tariff_period(snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Tariff period of each snapshot, indexed like ``snapshots``.

    Parameters
    ----------
    snapshots : pd.DatetimeIndex
        Naive timestamps in UTC (the PyPSA-Eur convention) or tz-aware ones.

    Returns
    -------
    pd.Series
        ``'P1'`` to ``'P6'`` per snapshot.
    """
    local = snapshots if snapshots.tz is not None else snapshots.tz_localize("UTC")
    local = local.tz_convert(TZ)
    holidays = set().union(*(national_holidays(y) for y in local.year.unique()))

    def period(ts: pd.Timestamp) -> str:
        if ts.weekday() >= 5 or ts.date() in holidays or ts.hour in NIGHT_HOURS:
            return "P6"
        peak, shoulder = SEASON_PERIODS[SEASON_OF_MONTH[ts.month]]
        return peak if ts.hour in PEAK_HOURS else shoulder

    return pd.Series([period(ts) for ts in local], index=snapshots, name="period")
