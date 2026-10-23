# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Tests for the data behind the validation figures."""

from __future__ import annotations

import pandas as pd
import pypsa
import pytest

from scripts.plot_validation import corridors, published_capacity


def test_corridors_count_ntc_caps_and_merge_parallel_elements(
    toy_base: pypsa.Network,
) -> None:
    toy_base.add(
        "Line", "parallel", bus0="ES0 7AC", bus1="ES0 6AC", x=0.1, s_nom=1000.0
    )
    toy_base.lines["s_max_pu"] = 0.5
    toy_base.add(
        "Link", "hvdc", bus0="ES0 6AC", bus1="ES0 7AC", carrier="DC", p_nom=400.0
    )
    toy_base.add(
        "Link", "planned", bus0="ES0 6AC", bus1="ES0 7AC", carrier="DC", p_nom=0.0
    )
    toy_base.links.loc["hvdc", "p_max_pu"] = 0.5
    lines, links = corridors(toy_base)
    assert lines.to_dict() == {("ES0 6AC", "ES0 7AC"): pytest.approx(3.0)}
    assert links.to_dict() == {("ES0 6AC", "ES0 7AC"): pytest.approx(0.2)}


def test_published_capacity_averages_year_ends_over_scopes() -> None:
    caps = pd.DataFrame(
        [
            ("ES_peninsular", 2023, "onwind", 900.0),
            ("ES_peninsular", 2024, "onwind", 1100.0),
            ("ES_baleares", 2023, "onwind", 10.0),
            ("ES_baleares", 2024, "onwind", 10.0),
            ("ES_peninsular", 2023, "CCGT", 1.0),
            ("ES_peninsular", 2024, "CCGT", 500.0),
            ("PT", 2024, "onwind", 7000.0),
        ],
        columns=["country", "year", "carrier_or_metric", "value"],
    )
    out = published_capacity(caps, ("ES_peninsular", "ES_baleares"))
    assert out["onwind"] == pytest.approx(1010.0)
    assert out["CCGT"] == pytest.approx(500.0)
