# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT

"""
Guard tests for the data provenance registry (doc/data_sources.md).

Every data/custom CSV row must cite a registered source_id with URL and retrieval
date, and every vendored file in data/custom/sources/ must match its registered SHA-256.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_DATA_DIR = REPO_ROOT / "data" / "custom"
SOURCES_DIR = CUSTOM_DATA_DIR / "sources"
DATA_SOURCES_DOC = REPO_ROOT / "doc" / "data_sources.md"


def _parse_data_sources_table(md_path: Path) -> dict[str, dict[str, object]]:
    """Parse the Markdown registry table in doc/data_sources.md keyed by source_id."""
    registry: dict[str, dict[str, object]] = {}
    for raw_line in md_path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        source_id = cells[0].strip("`").strip()
        if source_id == "source_id":
            continue
        title = cells[1]
        publisher = cells[2]
        url = cells[3].strip("`").strip()
        retrieval_date = cells[4].strip("`").strip()
        license_terms = cells[5]
        vendored_files = re.findall(r"`([^`]+)`", cells[6])
        sha256_list = re.findall(r"`([0-9a-fA-F]{64})`", cells[7])
        registry[source_id] = {
            "title": title,
            "publisher": publisher,
            "url": url,
            "retrieval_date": retrieval_date,
            "license": license_terms,
            "vendored_files": vendored_files,
            "sha256_list": [s.lower() for s in sha256_list],
        }
    return registry


def test_every_custom_csv_has_source() -> None:
    """Every row in data/custom/*.csv has a valid source_id registered in doc/data_sources.md."""
    assert DATA_SOURCES_DOC.exists(), f"Missing {DATA_SOURCES_DOC}"
    registry = _parse_data_sources_table(DATA_SOURCES_DOC)
    assert len(registry) >= 30, f"Expected >=30 source_ids in doc/data_sources.md, got {len(registry)}"

    target_csvs = [
        "annual_demand.csv",
        "chp.csv",
        "national_capacities.csv",
        "ntc.csv",
        "prices.csv",
        "validation_2024_es.csv",
    ]
    for csv_name in target_csvs:
        csv_path = CUSTOM_DATA_DIR / csv_name
        assert csv_path.exists(), f"Missing custom CSV: {csv_path}"
        df = pd.read_csv(csv_path)
        assert "source_id" in df.columns, f"{csv_path.name} is missing a 'source_id' column"
        assert not df["source_id"].isna().any(), f"{csv_path.name} has NaN source_id entries"
        for raw_sid in df["source_id"].astype(str).str.strip().unique():
            sub_ids = [s.strip() for s in re.split(r"[;+]", raw_sid) if s.strip()]
            for sid in sub_ids:
                assert sid in registry, (
                    f"source_id {sid!r} in {csv_path.name} is not documented in doc/data_sources.md"
                )
                entry = registry[sid]
                if sid != "study_assumption_user_confirmed":
                    assert str(entry["url"]).startswith("http"), (
                        f"source_id {sid!r} missing valid URL: {entry['url']!r}"
                    )
                assert re.match(r"^\d{4}-\d{2}-\d{2}$", str(entry["retrieval_date"])), (
                    f"source_id {sid!r} missing ISO retrieval date: {entry['retrieval_date']!r}"
                )


def test_vendored_files_sha256_match() -> None:
    """Every vendored file in doc/data_sources.md exists in data/custom/sources/ and matches SHA-256."""
    registry = _parse_data_sources_table(DATA_SOURCES_DOC)
    assert registry, "No entries parsed from doc/data_sources.md"

    verified_count = 0
    for sid, entry in registry.items():
        vfiles = list(entry["vendored_files"])  # type: ignore[arg-type]
        shas = list(entry["sha256_list"])  # type: ignore[arg-type]
        assert len(vfiles) == len(shas), (
            f"source_id {sid!r} has {len(vfiles)} vendored files but {len(shas)} SHA-256 digests"
        )
        for rel_path_str, expected_sha in zip(vfiles, shas, strict=True):
            full_path = (
                (REPO_ROOT / rel_path_str).resolve()
                if rel_path_str.startswith("data/")
                else (SOURCES_DIR / rel_path_str).resolve()
            )
            assert full_path.exists(), f"Vendored file {rel_path_str} for {sid!r} does not exist"
            assert full_path.stat().st_size > 0, f"Vendored file {rel_path_str} is empty"
            actual_sha = hashlib.sha256(full_path.read_bytes()).hexdigest().lower()
            assert actual_sha == expected_sha, (
                f"SHA-256 mismatch for {rel_path_str} ({sid!r}): "
                f"expected {expected_sha}, got {actual_sha}"
            )
            verified_count += 1

    assert verified_count >= 35, f"Expected >=35 verified vendored file entries, got {verified_count}"
