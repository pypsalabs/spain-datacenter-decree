# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT
"""Unit tests for import-traced hourly grid RES / CFE share calculation."""

import pandas as pd
import pypsa
import pytest

from scripts.calculate_grid_res import (
    calculate_grid_res,
    calculate_grid_share,
    get_clean_carriers,
)


def _make_import_tracing_network() -> pypsa.Network:
    """
    Create a 4-snapshot hand-calculable ES-FR-PT network with lines and DC links in both orientations.

    Buses:
    - ``ES0`` (Spain)
    - ``FR0`` (France)
    - ``PT0`` (Portugal)

    Interconnectors (testing both orientations across lines and DC links):
    - Line ``line_FR_to_ES``: ``bus0="FR0"``, ``bus1="ES0"`` (exporter=bus0, importer=bus1)
    - Line ``line_ES_to_PT``: ``bus0="ES0"``, ``bus1="PT0"`` (importer=bus0, exporter=bus1)
    - Link ``link_FR_to_ES``: ``bus0="FR0"``, ``bus1="ES0"``, ``carrier="DC"``
    - Link ``link_ES_to_FR_rev``: ``bus0="ES0"``, ``bus1="FR0"``, ``carrier="DC"``
    """
    snapshots = pd.date_range("2025-01-01", periods=4, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)

    n.add("Bus", "ES0", carrier="AC", country="ES")
    n.add("Bus", "FR0", carrier="AC", country="FR")
    n.add("Bus", "PT0", carrier="AC", country="PT")

    # Generators in ES: onwind, ror, CCGT, nuclear
    n.add("Generator", "ES_wind", bus="ES0", carrier="onwind")
    n.add("Generator", "ES_ror", bus="ES0", carrier="ror")
    n.add("Generator", "ES_gas", bus="ES0", carrier="CCGT")
    n.add("Generator", "ES_nuc", bus="ES0", carrier="nuclear")

    # Reservoir hydro in ES (StorageUnit with carrier="hydro")
    n.add("StorageUnit", "ES_hydro", bus="ES0", carrier="hydro")

    # Generators in FR: onwind, nuclear, CCGT
    n.add("Generator", "FR_wind", bus="FR0", carrier="onwind")
    n.add("Generator", "FR_nuc", bus="FR0", carrier="nuclear")
    n.add("Generator", "FR_gas", bus="FR0", carrier="CCGT")

    # Generators in PT: solar, CCGT
    n.add("Generator", "PT_solar", bus="PT0", carrier="solar")
    n.add("Generator", "PT_gas", bus="PT0", carrier="CCGT")

    # Lines and DC links in both orientations
    n.add("Line", "line_FR_to_ES", bus0="FR0", bus1="ES0", x=0.1, s_nom=2000.0)
    n.add("Line", "line_ES_to_PT", bus0="ES0", bus1="PT0", x=0.1, s_nom=2000.0)
    n.add("Link", "link_FR_to_ES", bus0="FR0", bus1="ES0", carrier="DC", p_nom=2000.0)
    n.add(
        "Link", "link_ES_to_FR_rev", bus0="ES0", bus1="FR0", carrier="DC", p_nom=2000.0
    )

    # Populate hourly dispatch (4 snapshots)
    # ES domestic:
    #   wind  = [100, 100,  50, 200]
    #   ror   = [ 50,  50,  50,   0]
    #   hydro = [ 50,   0, 100,   0]  (reservoir hydro StorageUnit)
    #   nuc   = [100, 100, 100, 100]
    #   gas   = [200, 150, 100, 100]
    # -> ES domestic RES = [200, 150, 200, 200]
    # -> ES domestic CFE = [300, 250, 300, 300]
    # -> ES domestic total = [500, 400, 400, 400]
    n.generators_t.p = pd.DataFrame(
        {
            "ES_wind": [100.0, 100.0, 50.0, 200.0],
            "ES_ror": [50.0, 50.0, 50.0, 0.0],
            "ES_nuc": [100.0, 100.0, 100.0, 100.0],
            "ES_gas": [200.0, 150.0, 100.0, 100.0],
            # FR domestic:
            #   wind = [200, 100, 300, 0], nuc = [600, 300, 500, 400], gas = [200, 100, 200, 100]
            # -> FR RES share = [200/1000=0.20, 100/500=0.20, 300/1000=0.30, 0/500=0.0]
            # -> FR CFE share = [800/1000=0.80, 400/500=0.80, 800/1000=0.80, 400/500=0.80]
            "FR_wind": [200.0, 100.0, 300.0, 0.0],
            "FR_nuc": [600.0, 300.0, 500.0, 400.0],
            "FR_gas": [200.0, 100.0, 200.0, 100.0],
            # PT domestic:
            #   solar = [150, 0, 100, 50], gas = [50, 200, 100, 50]
            # -> PT RES (=CFE) share = [150/200=0.75, 0/200=0.0, 100/200=0.50, 50/100=0.50]
            "PT_solar": [150.0, 0.0, 100.0, 50.0],
            "PT_gas": [50.0, 200.0, 100.0, 50.0],
        },
        index=snapshots,
    )
    n.storage_units_t.p = pd.DataFrame(
        {"ES_hydro": [50.0, 0.0, 100.0, 0.0]},
        index=snapshots,
    )

    # Line flows:
    # line_FR_to_ES (bus0=FR0, bus1=ES0):
    #   t0: +100 MW FR->ES (import to ES = 100)
    #   t1: -80 MW ES->FR (export from ES, import to ES = 0)
    #   t2: +50 MW FR->ES (import to ES = 50)
    #   t3: 0 MW
    # line_ES_to_PT (bus0=ES0, bus1=PT0):
    #   t0: -80 MW PT->ES (p1 = +80, p0 = -80 -> import to ES = 80)
    #   t1: +60 MW ES->PT (p1 = -60, p0 = +60 -> export from ES, import to ES = 0)
    #   t2: -100 MW PT->ES (p1 = +100, p0 = -100 -> import to ES = 100)
    #   t3: 0 MW
    n.lines_t.p0 = pd.DataFrame(
        {
            "line_FR_to_ES": [100.0, -80.0, 50.0, 0.0],
            "line_ES_to_PT": [-80.0, 60.0, -100.0, 0.0],
        },
        index=snapshots,
    )
    n.lines_t.p1 = -n.lines_t.p0

    # Link flows:
    # link_FR_to_ES (bus0=FR0, bus1=ES0):
    #   t0: +50 MW FR->ES (import to ES = 50)
    #   t1: -40 MW ES->FR (export from ES = 0)
    #   t2: 0 MW
    #   t3: +100 MW FR->ES (import to ES = 100)
    # link_ES_to_FR_rev (bus0=ES0, bus1=FR0):
    #   t0: 0 MW
    #   t1: -100 MW FR->ES (p1 = +100, p0 = -100 -> import to ES = 100)
    #   t2: +70 MW ES->FR (export from ES = 0)
    #   t3: 0 MW
    n.links_t.p0 = pd.DataFrame(
        {
            "link_FR_to_ES": [50.0, -40.0, 0.0, 100.0],
            "link_ES_to_FR_rev": [0.0, -100.0, 70.0, 0.0],
        },
        index=snapshots,
    )
    n.links_t.p1 = -n.links_t.p0

    return n


def test_import_tracing() -> None:
    """Verify hand-calculated import-traced RES and CFE shares across lines & links."""
    n = _make_import_tracing_network()

    r_res = calculate_grid_res(n, country="ES", clean_carriers="res")
    r_cfe = calculate_grid_res(n, country="ES", clean_carriers="cfe")

    # Hand-calculated expected values per snapshot:
    # t0:
    #   ES dom: RES=200, CFE=300, tot=500
    #   Imp FR->ES = 100 (line) + 50 (link) = 150; FR_res = 0.20, FR_cfe = 0.80
    #   Imp PT->ES = 80 (line); PT_res = PT_cfe = 0.75
    #   Total supply = 500 + 150 + 80 = 730
    #   Clean RES = 200 + 150*0.20 + 80*0.75 = 200 + 30 + 60 = 290 -> 290 / 730
    #   Clean CFE = 300 + 150*0.80 + 80*0.75 = 300 + 120 + 60 = 480 -> 480 / 730
    # t1:
    #   ES dom: RES=150, CFE=250, tot=400
    #   Imp FR->ES = 0 (line_FR_to_ES exports) + 0 (link_FR_to_ES exports) + 100 (link_ES_to_FR_rev) = 100
    #   FR_res = 0.20, FR_cfe = 0.80
    #   Imp PT->ES = 0 (line_ES_to_PT exports)
    #   Total supply = 400 + 100 = 500
    #   Clean RES = 150 + 100*0.20 = 170 -> 170 / 500 = 0.34
    #   Clean CFE = 250 + 100*0.80 = 330 -> 330 / 500 = 0.66
    # t2:
    #   ES dom: RES=200, CFE=300, tot=400
    #   Imp FR->ES = 50 (line); FR_res = 0.30, FR_cfe = 0.80
    #   Imp PT->ES = 100 (line); PT_res = PT_cfe = 0.50
    #   Total supply = 400 + 50 + 100 = 550
    #   Clean RES = 200 + 50*0.30 + 100*0.50 = 200 + 15 + 50 = 265 -> 265 / 550
    #   Clean CFE = 300 + 50*0.80 + 100*0.50 = 300 + 40 + 50 = 390 -> 390 / 550
    # t3:
    #   ES dom: RES=200, CFE=300, tot=400
    #   Imp FR->ES = 100 (link); FR_res = 0.0, FR_cfe = 0.80
    #   Imp PT->ES = 0
    #   Total supply = 400 + 100 = 500
    #   Clean RES = 200 + 100*0.0 = 200 -> 200 / 500 = 0.40
    #   Clean CFE = 300 + 100*0.80 = 380 -> 380 / 500 = 0.76
    expected_res = pd.Series(
        [290.0 / 730.0, 170.0 / 500.0, 265.0 / 550.0, 200.0 / 500.0],
        index=n.snapshots,
    )
    expected_cfe = pd.Series(
        [480.0 / 730.0, 330.0 / 500.0, 390.0 / 550.0, 380.0 / 500.0],
        index=n.snapshots,
    )

    pd.testing.assert_series_equal(r_res, expected_res, check_names=False, atol=1e-9)
    pd.testing.assert_series_equal(r_cfe, expected_cfe, check_names=False, atol=1e-9)

    # Confirm RES vs CFE carrier sets differ specifically by nuclear
    res_carriers = get_clean_carriers(mode="res")
    cfe_carriers = get_clean_carriers(mode="cfe")
    assert "nuclear" not in res_carriers
    assert "nuclear" in cfe_carriers
    assert (r_cfe > r_res).all()


def test_storage_excluded() -> None:
    """Verify PHS and battery discharge/charge are excluded while hydro and ror count."""
    n = _make_import_tracing_network()
    baseline_res = calculate_grid_res(n, country="ES", clean_carriers="res")

    # Add PHS StorageUnit, battery StorageUnit, and Store+Link battery discharger in ES with huge output
    n.add("StorageUnit", "ES_phs", bus="ES0", carrier="PHS")
    n.add("StorageUnit", "ES_batt_su", bus="ES0", carrier="battery")
    n.add("Bus", "ES_batt_bus", carrier="battery", country="ES")
    n.add(
        "Link",
        "ES_batt_discharger",
        bus0="ES_batt_bus",
        bus1="ES0",
        carrier="battery discharger",
    )

    n.storage_units_t.p["ES_phs"] = [500.0, -300.0, 800.0, 400.0]
    n.storage_units_t.p["ES_batt_su"] = [300.0, 200.0, -100.0, 600.0]
    n.links_t.p0["ES_batt_discharger"] = [250.0, 250.0, 250.0, 250.0]
    n.links_t.p1["ES_batt_discharger"] = [-240.0, -240.0, -240.0, -240.0]

    with_storage_res = calculate_grid_res(n, country="ES", clean_carriers="res")
    pd.testing.assert_series_equal(
        with_storage_res, baseline_res, check_names=False, atol=1e-12
    )


def test_dc_assets_excluded() -> None:
    """Verify data center PPA generators and batteries are excluded from grid share."""
    n = _make_import_tracing_network()
    baseline_res = calculate_grid_res(n, country="ES", clean_carriers="res")

    # Add large DC PPA wind and solar generators on ES0
    n.add("Generator", "DC ES0 onwind", bus="ES0", carrier="onwind")
    n.add("Generator", "DC_ES0_solar", bus="ES0", carrier="solar")
    n.generators_t.p["DC ES0 onwind"] = [900.0, 900.0, 900.0, 900.0]
    n.generators_t.p["DC_ES0_solar"] = [700.0, 700.0, 700.0, 700.0]

    with_dc_res = calculate_grid_res(n, country="ES", clean_carriers="res")
    pd.testing.assert_series_equal(
        with_dc_res, baseline_res, check_names=False, atol=1e-12
    )


def test_boundary_carrier_excluded() -> None:
    """Verify fixed 'boundary' flows are excluded: a neighbour's share is its domestic mix."""
    snapshots = pd.date_range("2025-01-01", periods=3, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "ES0", carrier="AC", country="ES")
    n.add("Bus", "FR0", carrier="AC", country="FR")

    n.add("Generator", "ES_wind", bus="ES0", carrier="onwind")
    n.add("Generator", "ES_gas", bus="ES0", carrier="CCGT")
    n.add("Generator", "FR_wind", bus="FR0", carrier="onwind")
    n.add("Generator", "FR_boundary_DE", bus="FR0", carrier="boundary")
    n.add("Line", "line_FR_ES", bus0="FR0", bus1="ES0", x=0.1, s_nom=1000.0)

    n.generators_t.p = pd.DataFrame(
        {
            "ES_wind": [100.0, 100.0, 100.0],
            "ES_gas": [100.0, 100.0, 100.0],
            "FR_wind": [100.0, 100.0, 100.0],
            "FR_boundary_DE": [100.0, 100.0, 100.0],
        },
        index=snapshots,
    )
    # 100 MW imported from FR into ES every hour
    n.lines_t.p0 = pd.DataFrame({"line_FR_ES": [100.0, 100.0, 100.0]}, index=snapshots)
    n.lines_t.p1 = -n.lines_t.p0

    r_es = calculate_grid_share(n, country="ES", clean_carriers="res")
    # FR boundary import is excluded from FR's supply, so FR share = 100 wind / 100 = 1.0.
    # ES clean share: (100 wind + 100 import * 1.0) / (200 domestic + 100 import) = 2/3.
    expected = pd.Series([2.0 / 3.0] * 3, index=snapshots)
    assert r_es.tolist() == pytest.approx(expected.tolist(), abs=1e-9)
