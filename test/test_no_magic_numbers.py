# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT

"""AST guard test preventing undocumented magic numbers in scripts/*.py."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Universal mathematical, unit-conversion, time, tolerance, and calendar-year literals
GLOBAL_ALLOW_LIST: set[float | int] = {
    0,
    1,
    -1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    24,
    60,
    100,
    365,
    366,
    1000,
    8760,
    8784,
    0.5,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-12,
    1e4,
    1e5,
    1e6,
    1e9,
    1e12,
    # Study weather and planning years
    2010,
    2012,
    2013,
    2019,
    2020,
    2022,
    2023,
    2024,
    2025,
    2030,
}

# Documented inline fallback defaults or diagnostic/plotting literals in existing scripts
SCRIPT_DOCUMENTED_LITERALS: dict[str, set[float | int]] = {
    "data_center.py": {
        25.0,  # fallback lifetime (yr) when missing from processed cost table
        0.369,  # CHP electric-side CO2 intensity (tCO2/MWh_el, MITECO/IDAE)
    },
}


def _scan_script_for_magic_numbers(py_path: Path) -> list[tuple[int, float | int, str]]:
    """Return (lineno, value, line_text) for numeric literals not in module constants or allow-lists."""
    source = py_path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(py_path))

    # Collect AST nodes belonging to module-level constant assignments
    module_constant_nodes: set[ast.AST] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            for sub in ast.walk(stmt):
                module_constant_nodes.add(sub)

    script_allow = GLOBAL_ALLOW_LIST | SCRIPT_DOCUMENTED_LITERALS.get(
        py_path.name, set()
    )
    violations: list[tuple[int, float | int, str]] = []

    for node in ast.walk(tree):
        if node in module_constant_nodes:
            continue
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            val = node.value
            if val not in script_allow and abs(val) not in script_allow:
                line_str = (
                    lines[node.lineno - 1].strip()
                    if 1 <= node.lineno <= len(lines)
                    else ""
                )
                violations.append((node.lineno, val, line_str))
    return sorted(violations)


def test_core_cfe_scripts_have_zero_magic_numbers() -> None:
    """Core CFE optimization, grid-supply, iteration, and boundary scripts have zero magic numbers."""
    core_scripts = [
        "cfe_constraints.py",
        "calculate_grid_res.py",
        "iterate.py",
        "make_summary.py",
        "data_center.py",
        "prepare_network.py",
        "solve_network.py",
    ]
    for name in core_scripts:
        path = SCRIPTS_DIR / name
        assert path.exists(), f"Missing core script: {path}"
        violations = _scan_script_for_magic_numbers(path)
        assert not violations, f"Undocumented magic numbers in {name}: {violations}"


def test_all_cfe_scripts_have_no_unregistered_magic_numbers() -> None:
    """No script in scripts/*.py contains unregistered numeric assumption literals."""
    py_files = sorted(SCRIPTS_DIR.glob("*.py"))
    assert len(py_files) >= 9
    for path in py_files:
        violations = _scan_script_for_magic_numbers(path)
        assert not violations, (
            f"Unregistered numeric literal(s) found in {path.name}:\n"
            + "\n".join(f"  L{ln}: {val} ({src})" for ln, val, src in violations)
        )
