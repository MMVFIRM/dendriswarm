#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "docs" / "PROOF_RUN_V07.json").read_text())
assert report["proof"] == "dendriswarm-v0.7.0-cifar100-swarm-campaign"
assert report["all_pass"] is True
assert len(report["gates"]) == 16
assert all(item["pass"] for item in report["gates"])
assert report["claim_boundary"]["real_cifar100_campaign_code"] is True
assert report["claim_boundary"]["external_benchmark_accuracy_claim"] is False
print("v0.7.0 CIFAR-100 campaign proof validated: 16/16 gates")
