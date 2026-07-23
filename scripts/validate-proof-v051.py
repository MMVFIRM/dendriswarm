#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "PROOF_RUN_V051.json"
report = json.loads(path.read_text())
assert report["format"] == "dendriswarm.proof.v051"
assert report["version"] == "0.5.1"
assert report["baseline_training_included"] is False
assert report["historical_trained_weights_exercised"] is False
assert report["gate_count"] == 15
assert report["passed"] is True
assert all(value["pass"] for value in report["gates"].values())
print("v0.5.1 proof valid: 15/15 gates")
