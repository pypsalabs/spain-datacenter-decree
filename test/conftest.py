# SPDX-FileCopyrightText: : 2026 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""Pytest configuration ensuring the project root is first on ``sys.path``."""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
