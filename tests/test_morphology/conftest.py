"""Shared fixtures for the learned-morphology test suite.

The AE/flow checkpoints live in ``wandb_weights/`` in the repo, stored
with git-lfs. Tests that need real weights point at them via environment
variables (defaulting to those repo paths) and skip cleanly when the
directories aren't present -- e.g. in a CI job that skips ``git lfs pull``,
or when pointing at a newly trained pair elsewhere.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

AE_CHECKPOINT_DIR = Path(
    os.environ.get(
        "SHINE_TEST_AE_CHECKPOINT_DIR",
        REPO_ROOT / "wandb_weights" / "i344nq38" / "epoch_2000",
    )
)
AE_EPOCH = int(os.environ.get("SHINE_TEST_AE_EPOCH", "2000"))

FLOW_CHECKPOINT_DIR = Path(
    os.environ.get(
        "SHINE_TEST_FLOW_CHECKPOINT_DIR",
        REPO_ROOT / "wandb_weights" / "2815kuay" / "epoch_500",
    )
)
FLOW_EPOCH = int(os.environ.get("SHINE_TEST_FLOW_EPOCH", "500"))

requires_checkpoints = pytest.mark.skipif(
    not (AE_CHECKPOINT_DIR.is_dir() and FLOW_CHECKPOINT_DIR.is_dir()),
    reason="AE/flow checkpoints not found on this machine "
    "(set SHINE_TEST_AE_CHECKPOINT_DIR / SHINE_TEST_FLOW_CHECKPOINT_DIR)",
)
