from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/PROOF_RUN_V03.json")
proof = json.loads(path.read_text())
assert proof["format"] == "dendriswarm.proof-run.v3.2"
assert proof["all_gates_pass"] is True
assert len(proof["gates"]) == 12
assert all(gate["pass"] is True for gate in proof["gates"].values())
metrics = proof["metrics"]
assert metrics["verification_locality"]["behavioral_binding_full_passes"] == 0
assert metrics["replicated_leverage"]["final_replication_passed"] is True
assert metrics["replicated_leverage"]["replicated_net_wins"] > 0
assert metrics["identity_economics"]["permissionless_registration_grant_units"] == 0
assert proof["gates"]["5_stale_work_never_loses_bond_or_challenge_budget"]["pass"] is True
print(f"validated {path}: 12/12 gates")
