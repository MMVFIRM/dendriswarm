#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf .demo-state
mkdir -p .demo-state
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m dendriswarm coordinator --state .demo-state/coordinator --bootstrap --inference-audit-rate 1.0 >.demo-state/coordinator.log 2>&1 &
COORD_PID=$!
SEED1=""; SEED2=""; SEED3=""
cleanup() { kill "$COORD_PID" ${SEED1:+"$SEED1"} ${SEED2:+"$SEED2"} ${SEED3:+"$SEED3"} 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 40); do
  curl -fsS http://127.0.0.1:8787/v1/meta >/dev/null 2>&1 && break
  sleep 0.25
done
curl -fsS http://127.0.0.1:8787/v1/meta >/dev/null

python -m dendriswarm seed --state .demo-state/seed-a --poll 0.1 >.demo-state/seed-a.log 2>&1 & SEED1=$!
python -m dendriswarm seed --state .demo-state/seed-b --poll 0.1 >.demo-state/seed-b.log 2>&1 & SEED2=$!
python -m dendriswarm seed --state .demo-state/seed-c --poll 0.1 >.demo-state/seed-c.log 2>&1 & SEED3=$!

READY=0
for _ in $(seq 1 600); do
  STATUS="$(curl -fsS http://127.0.0.1:8787/v1/stats)"
  if python -c 'import json,sys; s=json.load(sys.stdin); raise SystemExit(0 if s.get("canonical") and s.get("completed_tasks",0)>=40 and s.get("queued_tasks")==0 and s.get("assigned_tasks")==0 else 1)' <<<"$STATUS"; then
    READY=1
    break
  fi
  sleep 0.1
done

if [[ "$READY" != "1" ]]; then
  echo "DendriSwarm demo did not converge" >&2
  for log in .demo-state/*.log; do echo "--- $log ---" >&2; tail -80 "$log" >&2; done
  exit 1
fi

kill "$SEED1" 2>/dev/null || true
wait "$SEED1" 2>/dev/null || true
SEED1=""
python -m dendriswarm infer-sample 0 --state .demo-state/seed-a --wait 60 > .demo-state/inference.out
FINAL_STATS="$(curl -fsS http://127.0.0.1:8787/v1/stats)"
CHECKPOINT="$(curl -fsS http://127.0.0.1:8787/v1/audit/checkpoint)"

if grep -Eq '500 Internal Server Error|Traceback \(most recent call last\)' .demo-state/*.log; then
  echo "HTTP demo produced a server error" >&2
  grep -En '500 Internal Server Error|Traceback \(most recent call last\)' .demo-state/*.log >&2 || true
  exit 1
fi
if grep -Eq "seed connection error: Client error '40[014]" .demo-state/seed-*.log; then
  echo "HTTP demo produced an artifact/authentication client error" >&2
  grep -En "seed connection error: Client error '40[014]" .demo-state/seed-*.log >&2 || true
  exit 1
fi

python - "$FINAL_STATS" "$CHECKPOINT" .demo-state/inference.out <<'PY'
import json
import re
import sys
from pathlib import Path

stats = json.loads(sys.argv[1])
checkpoint = json.loads(sys.argv[2])
text = Path(sys.argv[3]).read_text()
match = re.search(r"Expected label:\s*(\d+)", text)
if not match:
    raise SystemExit("missing expected label in inference output")
start = text.find("{")
if start < 0:
    raise SystemExit("missing inference job JSON")
job = json.loads(text[start:])
expected = int(match.group(1))
report = {
    "format": "dendriswarm.local-http-run.v2",
    "processes": {"coordinator": 1, "seeds": 3},
    "stats": stats,
    "inference": {
        "expected_label": expected,
        "task_id": job["id"],
        "output": job["output"],
        "correct": job["output"]["prediction"] == expected,
    },
    "audit_checkpoint": checkpoint,
    "server_errors_detected": False,
}
print(json.dumps(report, indent=2, sort_keys=True))
PY
