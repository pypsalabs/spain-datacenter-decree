# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Tests for the published actuals the validation report compares against."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_network import _actual_cf


def test_actual_cf_uses_average_operating_capacity_over_leap_year() -> None:
    caps = pd.DataFrame(
        {
            "country": ["ES_peninsular", "ES_baleares"] * 3 + ["PT"],
            "year": [2022, 2022, 2023, 2023, 2024, 2024, 2024],
            "carrier_or_metric": "onwind",
            "value": [1.0, 1.0, 900.0, 100.0, 1900.0, 100.0, 5000.0],
            "source_id": "redeia",
        }
    )
    cf, source = _actual_cf(1.5 * 8784 / 1e3, caps, "onwind", 2024)
    assert cf == pytest.approx(1.0)
    assert source == "redeia"
