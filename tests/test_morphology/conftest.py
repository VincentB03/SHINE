"""Shared fixtures for the learned-morphology test suite.

The AE/flow checkpoints are large (~30 MB) local files, not committed to
the repo. Tests that need real weights point at them via environment
variables (with a default matching this machine's checkpoint layout) and
skip cleanly when the directories aren't present, e.g. in CI.
"""

import os
from pathlib import Path

import pytest

AE_CHECKPOINT_DIR = Path(
    os.environ.get(
        "SHINE_TEST_AE_CHECKPOINT_DIR",
        "/Users/vb287274/Desktop/wandb_weights/i1pf186a/epoch_2000",
    )
)
AE_EPOCH = int(os.environ.get("SHINE_TEST_AE_EPOCH", "2000"))

FLOW_CHECKPOINT_DIR = Path(
    os.environ.get(
        "SHINE_TEST_FLOW_CHECKPOINT_DIR",
        "/Users/vb287274/Desktop/wandb_weights/95f2vnu6/epoch_50",
    )
)
FLOW_EPOCH = int(os.environ.get("SHINE_TEST_FLOW_EPOCH", "50"))

requires_checkpoints = pytest.mark.skipif(
    not (AE_CHECKPOINT_DIR.is_dir() and FLOW_CHECKPOINT_DIR.is_dir()),
    reason="AE/flow checkpoints not found on this machine "
    "(set SHINE_TEST_AE_CHECKPOINT_DIR / SHINE_TEST_FLOW_CHECKPOINT_DIR)",
)
