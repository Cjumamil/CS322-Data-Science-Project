"""Trading bot package.

The repo can vendor a small dependency bundle under ``.vendor_bundle`` for
environments where user-site packages or shell activation are unreliable.
Importing the package makes those vendored modules available automatically for
project entry points such as ``python -m trading_bot.backtest``.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
VENDORED_PACKAGES = REPO_ROOT / ".vendor_bundle"
USER_SITE_PACKAGES = Path(site.getusersitepackages()).resolve()

sys.path[:] = [
    path
    for path in sys.path
    if not path or Path(path).resolve() != USER_SITE_PACKAGES
]

if VENDORED_PACKAGES.is_dir():
    vendored_path = str(VENDORED_PACKAGES)
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)
