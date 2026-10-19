# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Unit tests for the prepare_network patch engine (boundary closure, wind factors, freeze)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.prepare_network import (
    _patch_wind_correction,
    attach_boundary_flows,
    freeze_fleet,
    prepare_network,
)


def _toy_network(freq: str = "h", periods: int = 6) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=periods, freq=freq))
    if freq == "3h":
        n.snapshot_weightings.loc[:, :] = 3.0
    n.add("Bus", "FR0 0AC", carrier="AC", country="FR", x=-1.0, y=47.0)
    n.add("Bus", "FR0 1AC", carrier="AC", country="FR", x=6.0, y=49.0)
    n.add("Bus", "ES0 0AC", carrier="AC", country="ES", x=-3.7, y=40.4)
    n.add(
        "Generator",
        "ES0 0AC gas",
        bus="ES0 0AC",
        carrier="CCGT",
        p_nom=100.0,
        p_nom_extendable=True,
    )
    return n


def _flows(snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for nb, values in {
        "GB": [500.0] * 3 + [-300.0] * 3,
        "DE_BE": [-1000.0] * 6,
    }.items():
        for ts, v in zip(snapshots, values):
            rows.append({"snapshot": ts, "neighbour": nb, "flow_mw_into_FR": v})
    return pd.DataFrame(rows)


def test_attach_boundary_flows_nearest_bus_and_fixed() -> None:
    """Imports become fixed generators, exports fixed loads, each at the nearest FR bus."""
    n = _toy_network()
    attach_boundary_flows(n, _flows(n.snapshots))

    gb_gen = "FR0 0AC boundary import GB"
    assert gb_gen in n.generators.index
    assert n.generators.at[gb_gen, "carrier"] == "boundary"
    assert not n.generators.at[gb_gen, "p_nom_extendable"]
    assert np.allclose(n.generators_t.p_min_pu[gb_gen], n.generators_t.p_max_pu[gb_gen])
    imp_mw = n.generators_t.p_max_pu[gb_gen] * n.generators.at[gb_gen, "p_nom"]
    assert np.allclose(imp_mw, [500.0] * 3 + [0.0] * 3)

    assert "FR0 0AC boundary export GB" in n.loads.index
    assert np.allclose(
        n.loads_t.p_set["FR0 0AC boundary export GB"], [0.0] * 3 + [300.0] * 3
    )
    # DE_BE is pure export, attached at the eastern bus
    assert "FR0 1AC boundary import DE_BE" not in n.generators.index
    assert np.allclose(n.loads_t.p_set["FR0 1AC boundary export DE_BE"], 1000.0)


def test_attach_boundary_flows_averages_to_3h() -> None:
    """Hourly flows are averaged onto a 3-hourly network, conserving energy."""
    n = _toy_network(freq="3h", periods=2)
    hourly = pd.date_range("2024-01-01", periods=6, freq="h")
    attach_boundary_flows(n, _flows(hourly))

    gb_gen = "FR0 0AC boundary import GB"
    imp_mw = n.generators_t.p_max_pu[gb_gen] * n.generators.at[gb_gen, "p_nom"]
    assert np.allclose(imp_mw, [500.0, 0.0])
    assert np.allclose(n.loads_t.p_set["FR0 0AC boundary export GB"], [0.0, 300.0])


def test_attach_boundary_flows_rejects_gaps() -> None:
    """Flows not covering all snapshots raise instead of silently filling."""
    n = _toy_network()
    short = _flows(n.snapshots[:3])
    with pytest.raises(ValueError, match="do not cover"):
        attach_boundary_flows(n, short)


def test_prepare_network_freezes_and_rejects_unknown_patch() -> None:
    """prepare_network applies patches, freezes the fleet, and fails fast on unknown names."""
    n = _toy_network()
    prepare_network(
        n, patches={"boundary_flows": True}, ctx={"boundary_flows": _flows(n.snapshots)}
    )
    assert not n.generators["p_nom_extendable"].any()

    with pytest.raises(ValueError, match="Unknown patches"):
        prepare_network(_toy_network(), patches={"nonsense": True}, ctx={})


def test_freeze_fleet_idempotent() -> None:
    """freeze_fleet is a no-op on an already frozen network."""
    n = _toy_network()
    freeze_fleet(n)
    freeze_fleet(n)
    assert not n.generators["p_nom_extendable"].any()


def _wind_toy() -> pypsa.Network:
    n = _toy_network()
    n.add(
        "Generator",
        "ES0 0AC 0 onwind",
        bus="ES0 0AC",
        carrier="onwind",
        p_nom=100.0,
        p_max_pu=pd.Series(0.5, index=n.snapshots),
    )
    n.add(
        "Generator",
        "ES0 0AC 1 onwind",
        bus="ES0 0AC",
        carrier="onwind",
        p_nom=100.0,
        p_max_pu=pd.Series(0.9, index=n.snapshots),
    )
    return n


def _factor_csv(tmp_path: Path, countries: tuple[str, ...] = ("ES", "PT")) -> Path:
    rows = {"ES": 1.2, "PT": 2.0}
    lines = ["country,year,carrier_or_metric,value,unit,source_id,note"]
    lines += [
        f"{c},2025,wind_correction_factor,{rows[c]},per_unit,study_wind_backcast_2024,test"
        for c in countries
    ]
    (tmp_path / "wind_correction.csv").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_wind_correction_applies_fixed_factor_capped(tmp_path: Path) -> None:
    """Case-study runs scale onwind by the cited back-cast factor, capped at 1.0."""
    n = _wind_toy()
    ctx = {"custom_dir": _factor_csv(tmp_path), "data_year": 2025, "weather_year": 2024}
    _patch_wind_correction(n, ctx)
    assert np.allclose(n.generators_t.p_max_pu["ES0 0AC 0 onwind"], 0.6)
    assert np.allclose(n.generators_t.p_max_pu["ES0 0AC 1 onwind"], 1.0)


def test_wind_correction_missing_factor_raises(tmp_path: Path) -> None:
    """A case-study run without a cited factor fails fast instead of silently skipping."""
    n = _wind_toy()
    ctx = {
        "custom_dir": _factor_csv(tmp_path, countries=("PT",)),
        "data_year": 2025,
        "weather_year": 2024,
    }
    with pytest.raises(ValueError, match="No wind correction factor for ES"):
        _patch_wind_correction(n, ctx)


def test_wind_correction_backcast_requires_wy2024(tmp_path: Path) -> None:
    """Deriving the factor on 2024 data under a different weather year is invalid."""
    n = _wind_toy()
    ctx = {"custom_dir": _factor_csv(tmp_path), "data_year": 2024, "weather_year": 2013}
    with pytest.raises(ValueError, match="requires weather year 2024"):
        _patch_wind_correction(n, ctx)
