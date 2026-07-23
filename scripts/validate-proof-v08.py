#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/PROOF_RUN_V08.json")
report = json.loads(path.read_text())
assert report["proof"] == "dendriswarm-v0.8.0-local-dashboard"
assert report["gate_count"] == 12
assert report["all_pass"] is True
assert all(item["pass"] for item in report["gates"])
print("v0.8.0 local dashboard proof validated: 12/12 gates")
