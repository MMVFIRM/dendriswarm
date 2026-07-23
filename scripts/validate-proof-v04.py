#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/PROOF_RUN_V04.json")
    report = json.loads(path.read_text())
    assert report["format"] == "dendriswarm.heterogeneous-seeding-proof.v1"
    assert report["version"] == "0.4.0"
    assert report["all_gates_pass"] is True
    assert len(report["gates"]) >= 8
    assert all(gate["pass"] for gate in report["gates"])
    low = report["evidence"]["low_resource_task"]
    assert low["cpu_count"] == 1
    assert low["machine"] == "arm64"
    assert low["kind"] == "exploration"
    assert low["credit_units"] > 0
    assert report["evidence"]["five_percent_limits"]["duty_cycle"] < 1
    print(f"validated {path}: {len(report['gates'])}/{len(report['gates'])} gates")


if __name__ == "__main__":
    main()
