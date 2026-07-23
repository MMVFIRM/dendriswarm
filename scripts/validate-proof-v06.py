#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "docs" / "PROOF_RUN_V06.json").read_text())
assert report["proof"] == "dendriswarm-v0.6.0-trainable-native10"
assert report["all_pass"] is True
assert len(report["gates"]) == 20
assert all(item["pass"] for item in report["gates"])
assert report["exact_profile"]["parameter_count"] == 4_898_812
assert report["exact_profile"]["reachability"]["reachable_float_fraction"] == 1.0
assert report["network_round"]["selection_hash"] != report["network_round"]["replication_hash"]
assert report["network_round"]["root_before"] != report["network_round"]["root_after"]
print("v0.6.0 proof validated: 20/20 gates")
