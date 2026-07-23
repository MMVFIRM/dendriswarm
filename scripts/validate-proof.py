from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/PROOF_RUN.json")
proof = json.loads(path.read_text())
stats = proof["stats"]
canonical = stats["canonical"]
baselines = stats["benchmark"]["accuracy"]
inference = proof["inference"]
checkpoint = proof["audit_checkpoint"]["checkpoint"]
assert canonical["test_accuracy"] >= baselines["logistic_regression"]
assert canonical["hidden_accuracy"] >= 0.94
assert canonical["verifications"] >= 2
assert inference["correct"] is True
assert inference["output"]["active_branches"] < inference["output"]["total_branches"]
assert checkpoint["valid"] is True
assert stats["audit"]["valid"] is True
print(f"validated {path}")
