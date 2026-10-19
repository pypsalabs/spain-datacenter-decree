# SPDX-FileCopyrightText: : 2026 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Tidy France's measured cross-border exchanges into the boundary-flows CSV.

One-off data preparation, deliberately not part of the Snakemake workflow: the raw
RTE eCO2mix extract is vendored in ``data/custom/sources/`` and this script converts
it to ``data/custom/fr_boundary_flows_<year>.csv`` (columns ``snapshot``,
``neighbour``, ``flow_mw_into_FR``; positive = import into France; hourly means of
the half-hourly commercial exchanges, short gaps interpolated). Germany and Belgium
are reported as one combined border (``DE_BE``) in the RTE national file. Spain is
excluded: the ES-FR border is modelled endogenously. With ``--download``, the raw
extract is first fetched from the ODRE open-data API (no key required).

References
----------
- [RTE eCO2mix definitive national data](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve

import pandas as pd

logger = logging.getLogger(__name__)

API_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "eco2mix-national-cons-def/exports/csv"
)
EXCHANGE_COLUMNS: dict[str, str] = {
    "ech_comm_angleterre": "GB",
    "ech_comm_italie": "IT",
    "ech_comm_suisse": "CH",
    "ech_comm_allemagne_belgique": "DE_BE",
}


def download_raw(year: int, raw_path: Path) -> None:
    """Fetch the eCO2mix exchange columns for ``year`` from the ODRE API."""
    params = urlencode(
        {
            "select": "date_heure," + ",".join(EXCHANGE_COLUMNS) + ",ech_comm_espagne,ech_physiques",
            "where": f"date_heure >= date'{year}-01-01' AND date_heure < date'{year + 1}-01-01'",
            "order_by": "date_heure",
        }
    )
    logger.info("Downloading eCO2mix %d exchanges from %s", year, API_URL)
    urlretrieve(f"{API_URL}?{params}", raw_path)


def tidy_boundary_flows(raw_path: Path, year: int) -> pd.DataFrame:
    """Return the long-format hourly boundary flows from the raw eCO2mix extract."""
    df = pd.read_csv(raw_path, sep=";", encoding="utf-8-sig")
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True).dt.tz_localize(None)
    ex = (
        df.set_index("date_heure")
        .sort_index()[list(EXCHANGE_COLUMNS)]
        .rename(columns=EXCHANGE_COLUMNS)
        .apply(pd.to_numeric, errors="coerce")
        .resample("1h")
        .mean()
        .interpolate(limit=6)
    )
    if ex.isna().any().any():
        raise ValueError("Gaps longer than 6 h in the eCO2mix exchange data.")
    net_twh = ex.sum().sum() / 1e6
    logger.info("FR net flow over %d borders in %d: %.2f TWh (negative = export).", len(ex.columns), year, net_twh)
    long = ex.rename_axis("snapshot").reset_index().melt(
        id_vars="snapshot", var_name="neighbour", value_name="flow_mw_into_FR"
    )
    return long.sort_values(["neighbour", "snapshot"], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--download", action="store_true", help="Fetch the raw extract first.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "custom" / "sources" / f"eco2mix_national_cons_def_{args.year}_exchanges.csv"
    out_path = root / "data" / "custom" / f"fr_boundary_flows_{args.year}.csv"

    if args.download:
        download_raw(args.year, raw_path)
    long = tidy_boundary_flows(raw_path, args.year)
    long.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(long), out_path)


if __name__ == "__main__":
    main()
