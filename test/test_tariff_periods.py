# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Tariff periods of Circular 3/2020 on naive UTC snapshots."""

import pandas as pd
import pytest

from scripts.tariff_periods import tariff_period


@pytest.mark.parametrize(
    ("utc", "period"),
    [
        ("2024-01-02 08:00", "P1"),  # 09:00 CET, high-season weekday peak
        ("2024-01-02 07:00", "P2"),  # 08:00 CET, shoulder
        ("2024-01-02 21:00", "P2"),  # 22:00 CET, late shoulder
        ("2024-01-02 23:00", "P6"),  # 00:00 CET next day, night
        ("2024-07-01 07:00", "P1"),  # 09:00 CEST, DST shift
        ("2024-03-04 08:00", "P2"),  # mid-high season peak
        ("2024-06-03 08:00", "P3"),  # mid season peak
        ("2024-04-01 08:00", "P4"),  # low season peak
        ("2024-04-01 06:00", "P5"),  # 08:00 CEST, low season shoulder
        ("2024-01-06 10:00", "P6"),  # 6 January
        ("2024-12-25 10:00", "P6"),  # national holiday
        ("2024-01-07 10:00", "P6"),  # Sunday
    ],
)
def test_period_of_snapshot(utc: str, period: str) -> None:
    idx = pd.DatetimeIndex([pd.Timestamp(utc)])
    assert tariff_period(idx).iloc[0] == period


def test_year_hour_counts() -> None:
    idx = pd.date_range("2024-01-01", "2025-01-01", freq="h", inclusive="left")
    counts = tariff_period(idx).value_counts()
    # 4 high-season months of weekday peaks (9 h/day), minus the 1 and 6 January
    assert counts["P1"] == 774
    assert counts["P2"] == 971
    assert counts.sum() == len(idx)
    assert counts["P6"] > len(idx) / 2
