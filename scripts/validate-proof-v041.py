#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/PROOF_RUN_V041.json")
    report = json.loads(path.read_text())
    assert report["format"] == "dendriswarm.hostile-participation-proof.v1"
    assert report["version"] == "0.4.1"
    assert report["all_gates_pass"] is True
    assert len(report["gates"]) == 11
    assert report["test_nodes_executed"] >= 25
    assert all(gate["pass"] for gate in report["gates"])
    print(f"validated {path}: {len(report['gates'])}/{len(report['gates'])} gates")


if __name__ == "__main__":
    main()
