from __future__ import annotations

import json
from contextlib import contextmanager
from functools import wraps
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dendriswarm.core.models import NodeCapabilities, SeedPolicy, TaskKind, TaskRequirements
from dendriswarm.core.resources import (
    adjusted_runtime_seconds,
    derive_payload_requirements,
    node_can_run,
    requirements_from_value,
)


def synchronized(method):
    """Serialize access to the single SQLite connection used by the reference coordinator."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapped


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    public_key TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    policy TEXT NOT NULL DEFAULT '{}',
                    credit_units INTEGER NOT NULL DEFAULT 0,
                    completed INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    last_seen REAL NOT NULL,
                    quarantine_until REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS nonces (
                    node_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(node_id, action, nonce)
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    dedupe_key TEXT UNIQUE,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    requirements TEXT NOT NULL DEFAULT '{}',
                    reward_units INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    assigned_to TEXT,
                    lease_token TEXT,
                    lease_expires_at REAL,
                    lease_deadline_at REAL,
                    claim_bond_units INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    assigned_at REAL,
                    completed_at REAL,
                    output TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    parent_task TEXT,
                    excluded_nodes TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_queue
                    ON tasks(status, priority DESC, created_at ASC);
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    contributor TEXT NOT NULL,
                    license TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    artifact TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    artifact_hash TEXT NOT NULL UNIQUE,
                    artifact TEXT NOT NULL,
                    config TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    trainer_node TEXT NOT NULL,
                    trainer_nodes TEXT NOT NULL DEFAULT '[]',
                    training_task TEXT NOT NULL UNIQUE,
                    train_accuracy REAL NOT NULL,
                    coordinator_accuracy REAL NOT NULL,
                    hidden_accuracy REAL NOT NULL,
                    test_accuracy REAL,
                    verification_quorum INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_verifications (
                    candidate_id TEXT NOT NULL,
                    verifier_node TEXT NOT NULL,
                    task_id TEXT NOT NULL UNIQUE,
                    accuracy REAL NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(candidate_id, verifier_node),
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                );
                CREATE TABLE IF NOT EXISTS ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_key TEXT NOT NULL UNIQUE,
                    node_id TEXT NOT NULL,
                    amount_units INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    reference TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inference_requests (
                    request_key TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    task_id TEXT NOT NULL UNIQUE,
                    cost_units INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    refunded INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS work_reports (
                    work_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    task_id TEXT NOT NULL UNIQUE,
                    output_hash TEXT NOT NULL,
                    output TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(work_key, node_id)
                );
                CREATE INDEX IF NOT EXISTS idx_work_reports_key ON work_reports(work_key, kind);
                CREATE TABLE IF NOT EXISTS audit (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    body TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Forward-compatible migrations for state directories created by
            # v0.2/v0.3 packages. SQLite has no ADD COLUMN IF NOT EXISTS.
            node_columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(nodes)")}
            if "policy" not in node_columns:
                self.conn.execute("ALTER TABLE nodes ADD COLUMN policy TEXT NOT NULL DEFAULT '{}'")
            if "quarantine_until" not in node_columns:
                self.conn.execute("ALTER TABLE nodes ADD COLUMN quarantine_until REAL NOT NULL DEFAULT 0")
            task_columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(tasks)")}
            if "requirements" not in task_columns:
                self.conn.execute("ALTER TABLE tasks ADD COLUMN requirements TEXT NOT NULL DEFAULT '{}'")
            if "lease_deadline_at" not in task_columns:
                self.conn.execute("ALTER TABLE tasks ADD COLUMN lease_deadline_at REAL")
            if "claim_bond_units" not in task_columns:
                self.conn.execute("ALTER TABLE tasks ADD COLUMN claim_bond_units INTEGER NOT NULL DEFAULT 0")
            candidate_columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(candidates)")}
            if "trainer_nodes" not in candidate_columns:
                self.conn.execute("ALTER TABLE candidates ADD COLUMN trainer_nodes TEXT NOT NULL DEFAULT '[]'")
            inference_columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(inference_requests)")}
            if "refunded" not in inference_columns:
                self.conn.execute("ALTER TABLE inference_requests ADD COLUMN refunded INTEGER NOT NULL DEFAULT 0")
            # Repair any legacy duplicate active leases before installing the
            # database-enforced one-active-lease invariant. Keep the oldest
            # assignment and requeue the rest without consuming another attempt.
            duplicate_nodes = self.conn.execute(
                "SELECT assigned_to FROM tasks WHERE status='assigned' AND assigned_to IS NOT NULL "
                "GROUP BY assigned_to HAVING COUNT(*) > 1"
            ).fetchall()
            for duplicate in duplicate_nodes:
                rows = self.conn.execute(
                    "SELECT id FROM tasks WHERE status='assigned' AND assigned_to=? "
                    "ORDER BY assigned_at ASC,id ASC",
                    (duplicate["assigned_to"],),
                ).fetchall()
                for row in rows[1:]:
                    self.conn.execute(
                        "UPDATE tasks SET status='queued',assigned_to=NULL,assigned_at=NULL,lease_token=NULL,"
                        "lease_expires_at=NULL,lease_deadline_at=NULL,claim_bond_units=0,"
                        "attempts=MAX(0,attempts-1) WHERE id=?",
                        (row["id"],),
                    )
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_one_active_per_node "
                "ON tasks(assigned_to) WHERE status='assigned' AND assigned_to IS NOT NULL"
            )

    def _transaction(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")

    @contextmanager
    def transaction(self):
        """Atomic outer transaction over re-entrant Database method calls.

        Exceptions normally roll back. An exception carrying
        ``commit_transaction = True`` commits deliberate failure records before
        it is re-raised to the API layer.
        """
        with self.lock:
            outer = self.conn.in_transaction
            if not outer:
                self._transaction()
            try:
                yield
            except Exception as error:
                if not outer:
                    if getattr(error, "commit_transaction", False):
                        self.conn.commit()
                    else:
                        self.conn.rollback()
                raise
            else:
                if not outer:
                    self.conn.commit()

    @synchronized
    def register_node(
        self,
        node_id: str,
        public_key: str,
        capabilities: dict[str, Any],
        policy: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        policy_model = (
            SeedPolicy.model_validate(policy)
            if policy is not None
            else SeedPolicy(
                cpu_percent=100, memory_percent=100, disk_limit_mb=1_000_000,
                max_task_seconds=86400, allow_on_battery=True, max_system_cpu_percent=100,
            )
        )
        normalized_policy = policy_model.model_dump(mode="json")
        with self.lock:
            self._transaction()
            try:
                row = self.conn.execute("SELECT public_key FROM nodes WHERE id=?", (node_id,)).fetchone()
                if row and row["public_key"] != public_key:
                    raise ValueError("node id already exists with a different public key")
                self.conn.execute(
                    """INSERT INTO nodes(id,public_key,capabilities,policy,last_seen)
                       VALUES(?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         capabilities=excluded.capabilities,policy=excluded.policy,last_seen=excluded.last_seen""",
                    (
                        node_id,
                        public_key,
                        json.dumps(capabilities, sort_keys=True),
                        json.dumps(normalized_policy, sort_keys=True),
                        now,
                    ),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    @synchronized
    def node(self, node_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()

    @synchronized
    def consume_nonce(self, node_id: str, action: str, nonce: str) -> bool:
        with self.lock:
            try:
                self.conn.execute(
                    "INSERT INTO nonces(node_id,action,nonce,created_at) VALUES(?,?,?,?)",
                    (node_id, action, nonce, time.time()),
                )
                self.conn.execute("DELETE FROM nonces WHERE created_at < ?", (time.time() - 86400,))
                return True
            except sqlite3.IntegrityError:
                return False

    @synchronized
    def heartbeat(self, node_id: str) -> None:
        self.conn.execute("UPDATE nodes SET last_seen=? WHERE id=?", (time.time(), node_id))

    @synchronized
    def add_task(
        self,
        kind: TaskKind,
        payload: dict[str, Any],
        reward_units: int,
        priority: int,
        parent_task: str | None = None,
        excluded_nodes: list[str] | None = None,
        max_attempts: int = 3,
        dedupe_key: str | None = None,
        requirements: TaskRequirements | dict[str, Any] | None = None,
    ) -> str:
        normalized_requirements = requirements_from_value(requirements, kind)
        if payload.get("required_tags") and not normalized_requirements.required_tags:
            normalized_requirements = normalized_requirements.model_copy(
                update={"required_tags": list(payload["required_tags"])}
            )
        with self.lock:
            if dedupe_key:
                existing = self.conn.execute("SELECT id FROM tasks WHERE dedupe_key=?", (dedupe_key,)).fetchone()
                if existing:
                    return str(existing["id"])
            task_id = uuid.uuid4().hex
            try:
                self.conn.execute(
                    """INSERT INTO tasks(
                        id,dedupe_key,kind,payload,requirements,reward_units,priority,status,created_at,
                        max_attempts,parent_task,excluded_nodes
                    ) VALUES(?,?,?,?,?,?,?,'queued',?,?,?,?)""",
                    (
                        task_id,
                        dedupe_key,
                        kind.value,
                        json.dumps(payload, sort_keys=True),
                        json.dumps(normalized_requirements.model_dump(mode="json"), sort_keys=True),
                        reward_units,
                        priority,
                        time.time(),
                        max_attempts,
                        parent_task,
                        json.dumps(excluded_nodes or []),
                    ),
                )
                return task_id
            except sqlite3.IntegrityError:
                if dedupe_key:
                    row = self.conn.execute("SELECT id FROM tasks WHERE dedupe_key=?", (dedupe_key,)).fetchone()
                    if row:
                        return str(row["id"])
                raise

    @synchronized
    def refresh_queued_mutation_requirements(self) -> int:
        """Monotonically upgrade stale queued Native10 mutation contracts.

        Resource estimates are part of a signed task, so upgrades must happen
        before a task is claimed.  Only queued work is changed; active and
        historical task envelopes remain immutable.
        """
        upgraded = 0
        rows = self.conn.execute(
            "SELECT id,payload,requirements,attempts,excluded_nodes,parent_task "
            "FROM tasks WHERE status='queued' AND kind=?",
            (TaskKind.DENDRITRON_MUTATION.value,),
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            declared = requirements_from_value(
                json.loads(row["requirements"] or "{}"), TaskKind.DENDRITRON_MUTATION
            )
            derived = derive_payload_requirements(TaskKind.DENDRITRON_MUTATION, payload)
            merged = declared.model_copy(update={
                "resource_class": (
                    derived.resource_class
                    if derived.estimated_runtime_seconds > declared.estimated_runtime_seconds
                    else declared.resource_class
                ),
                "min_cpu_threads": max(declared.min_cpu_threads, derived.min_cpu_threads),
                "preferred_cpu_threads": max(declared.preferred_cpu_threads, derived.preferred_cpu_threads),
                "min_memory_mb": max(declared.min_memory_mb, derived.min_memory_mb),
                "max_memory_mb": max(int(declared.max_memory_mb or 0), int(derived.max_memory_mb or 0)),
                "min_disk_mb": max(declared.min_disk_mb, derived.min_disk_mb),
                "estimated_runtime_seconds": max(
                    declared.estimated_runtime_seconds, derived.estimated_runtime_seconds
                ),
                "hard_timeout_seconds": max(
                    declared.hard_timeout_seconds, derived.hard_timeout_seconds
                ),
                "max_artifact_bytes": max(declared.max_artifact_bytes, derived.max_artifact_bytes),
                "required_tags": sorted(set(declared.required_tags) | set(derived.required_tags)),
                "supported_machines": sorted(
                    set(declared.supported_machines) | set(derived.supported_machines)
                ),
            })
            if merged != declared:
                is_root_v6_search = (
                    payload.get("engine") == "dendriswarm.native10-trainable.v6"
                    and str(payload.get("work_key", "")).startswith("native10-v6-search:")
                    and row["parent_task"] is None
                )
                self.conn.execute(
                    "UPDATE tasks SET requirements=?,attempts=?,excluded_nodes=? "
                    "WHERE id=? AND status='queued'",
                    (
                        json.dumps(merged.model_dump(mode="json"), sort_keys=True),
                        0 if is_root_v6_search else int(row["attempts"]),
                        "[]" if is_root_v6_search else row["excluded_nodes"],
                        row["id"],
                    ),
                )
                upgraded += 1
        return upgraded

    @synchronized
    def requeue_expired_leases(self, now: float | None = None) -> dict[str, int]:
        now = now or time.time()
        requeued = 0
        failed = 0
        with self.lock:
            rows = self.conn.execute(
                "SELECT id,assigned_to,attempts,max_attempts,excluded_nodes,lease_token,claim_bond_units FROM tasks "
                "WHERE status='assigned' AND lease_expires_at < ?",
                (now,),
            ).fetchall()
            for row in rows:
                assigned = str(row["assigned_to"] or "")
                excluded = set(json.loads(row["excluded_nodes"] or "[]"))
                if assigned:
                    excluded.add(assigned)
                    self.conn.execute(
                        "UPDATE nodes SET failed=failed+1,quarantine_until=MAX(quarantine_until,?) WHERE id=?",
                        (now + 60.0, assigned),
                    )
                # Claim bonds are deliberately not refunded on abandoned expiry;
                # they are the economic cost of queue hoarding.
                if row["attempts"] >= row["max_attempts"]:
                    self.conn.execute(
                        "UPDATE tasks SET status='failed',completed_at=?,excluded_nodes=?,assigned_to=NULL," 
                        "lease_token=NULL,lease_expires_at=NULL,lease_deadline_at=NULL,claim_bond_units=0 WHERE id=?",
                        (now, json.dumps(sorted(excluded)), row["id"]),
                    )
                    self._refund_inference_if_terminal(str(row["id"]), "attempt-budget-exhausted")
                    failed += 1
                else:
                    self.conn.execute(
                        "UPDATE tasks SET status='queued',assigned_to=NULL,assigned_at=NULL,lease_token=NULL," 
                        "lease_expires_at=NULL,lease_deadline_at=NULL,claim_bond_units=0,excluded_nodes=? "
                        "WHERE id=? AND status='assigned'",
                        (json.dumps(sorted(excluded)), row["id"]),
                    )
                    requeued += 1
        return {"requeued": requeued, "failed": failed}


    def _node_already_on_candidate(self, node_id: str, candidate_id: str) -> bool:
        if self.conn.execute(
            "SELECT 1 FROM candidate_verifications WHERE candidate_id=? AND verifier_node=?",
            (candidate_id, node_id),
        ).fetchone():
            return True
        assigned = self.conn.execute(
            "SELECT payload FROM tasks WHERE kind='verification' AND status='assigned' AND assigned_to=?",
            (node_id,),
        ).fetchall()
        return any(json.loads(row["payload"]).get("candidate_id") == candidate_id for row in assigned)

    def _node_already_on_work(self, node_id: str, work_key: str) -> bool:
        """Keep replicated evidence independent at the identity layer."""
        if self.conn.execute(
            "SELECT 1 FROM work_reports WHERE work_key=? AND node_id=?",
            (work_key, node_id),
        ).fetchone():
            return True
        assigned = self.conn.execute(
            "SELECT payload FROM tasks WHERE status='assigned' AND assigned_to=?",
            (node_id,),
        ).fetchall()
        return any(json.loads(row["payload"]).get("work_key") == work_key for row in assigned)

    @synchronized
    def claim_task(self, node_id: str, lease_seconds: float) -> sqlite3.Row | None:
        # BEGIN IMMEDIATE plus the partial unique index make the one-active-lease
        # invariant database-enforced across coordinator processes, not merely
        # protected by this instance's Python RLock.
        with self.transaction():
            lease_updates = self.requeue_expired_leases()
            if lease_updates["requeued"]:
                self.refresh_queued_mutation_requirements()
            node = self.node(node_id)
            now_check = time.time()
            if not node or float(node["quarantine_until"] or 0) > now_check:
                return None
            active = self.conn.execute(
                "SELECT id FROM tasks WHERE status='assigned' AND assigned_to=? "
                "AND lease_expires_at>=? LIMIT 1",
                (node_id, now_check),
            ).fetchone()
            if active:
                return None
            capabilities = NodeCapabilities.model_validate(json.loads(node["capabilities"]))
            policy = SeedPolicy.model_validate(json.loads(node["policy"] or "{}"))
            offset = 0
            page_size = 256
            while True:
                rows = self.conn.execute(
                    "SELECT * FROM tasks WHERE status='queued' "
                    "ORDER BY priority DESC,created_at ASC LIMIT ? OFFSET ?",
                    (page_size, offset),
                ).fetchall()
                if not rows:
                    return None

                def fit_order(row: sqlite3.Row) -> tuple[int, int, int, float]:
                    requirement_value = json.loads(row["requirements"] or "{}")
                    return (
                        -int(row["priority"]),
                        -int(requirement_value.get("min_memory_mb", 0)),
                        -int(requirement_value.get("preferred_cpu_threads", 0)),
                        float(row["created_at"]),
                    )

                for row in sorted(rows, key=fit_order):
                    payload = json.loads(row["payload"])
                    excluded = set(json.loads(row["excluded_nodes"] or "[]"))
                    if node_id in excluded:
                        continue
                    work_key = str(payload.get("work_key", ""))
                    if work_key and self._node_already_on_work(node_id, work_key):
                        continue
                    requirements_value = json.loads(row["requirements"] or "{}")
                    if payload.get("required_tags") and not requirements_value.get("required_tags"):
                        requirements_value["required_tags"] = payload["required_tags"]
                    requirements = requirements_from_value(requirements_value, TaskKind(row["kind"]))
                    eligible, _ = node_can_run(TaskKind(row["kind"]), requirements, capabilities, policy)
                    if not eligible:
                        continue
                    if row["kind"] == TaskKind.VERIFICATION.value:
                        candidate_id = str(payload["candidate_id"])
                        if self._node_already_on_candidate(node_id, candidate_id):
                            continue
                    claim_bond = int(payload.get("claim_bond_units", 0))
                    if claim_bond and int(node["credit_units"]) < claim_bond:
                        continue
                    lease_token = uuid.uuid4().hex
                    now = time.time()
                    node_runtime = adjusted_runtime_seconds(requirements, capabilities)
                    absolute_budget = min(
                        float(policy.max_task_seconds) + 30.0,
                        max(float(requirements.hard_timeout_seconds) + 30.0, node_runtime * 3.0 + 30.0),
                    )
                    deadline = now + absolute_budget
                    adaptive_lease = (
                        float(lease_seconds) if float(lease_seconds) < 1.0
                        else max(float(lease_seconds), min(absolute_budget, node_runtime * 1.5 + 15.0))
                    )
                    expiry = min(deadline, now + adaptive_lease)
                    try:
                        changed = self.conn.execute(
                            "UPDATE tasks SET status='assigned',assigned_to=?,assigned_at=?,lease_token=?,"
                            "lease_expires_at=?,lease_deadline_at=?,claim_bond_units=?,attempts=attempts+1 "
                            "WHERE id=? AND status='queued' "
                            "AND NOT EXISTS (SELECT 1 FROM tasks active "
                            "WHERE active.status='assigned' AND active.assigned_to=?)",
                            (node_id, now, lease_token, expiry, deadline, claim_bond, row["id"], node_id),
                        ).rowcount
                    except sqlite3.IntegrityError:
                        # Another coordinator process won the unique active-lease race.
                        return None
                    if changed:
                        if claim_bond:
                            self.conn.execute(
                                "UPDATE nodes SET credit_units=credit_units-? WHERE id=?",
                                (claim_bond, node_id),
                            )
                            self.conn.execute(
                                "INSERT INTO ledger(entry_key,node_id,amount_units,reason,reference,created_at) "
                                "VALUES(?,?,?,?,?,?)",
                                (f"bond-lock:{row['id']}:{lease_token}", node_id, -claim_bond,
                                 "task claim bond locked", row["id"], now),
                            )
                        return self.task(str(row["id"]))
                offset += page_size


    @synchronized
    def renew_task_lease(
        self, task_id: str, node_id: str, lease_token: str, extension_seconds: float
    ) -> float | None:
        now = time.time()
        row = self.conn.execute(
            "SELECT lease_expires_at,lease_deadline_at FROM tasks WHERE id=? AND status='assigned' "
            "AND assigned_to=? AND lease_token=?",
            (task_id, node_id, lease_token),
        ).fetchone()
        if not row or float(row["lease_expires_at"] or 0) < now:
            return None
        deadline = float(row["lease_deadline_at"] or row["lease_expires_at"])
        if now >= deadline:
            return None
        new_expiry = min(deadline, max(float(row["lease_expires_at"]), now) + max(10.0, float(extension_seconds)))
        self.conn.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE id=? AND status='assigned' AND assigned_to=? AND lease_token=?",
            (new_expiry, task_id, node_id, lease_token),
        )
        return new_expiry

    @synchronized
    def abandon_task(self, task_id: str, node_id: str, lease_token: str, reason: str) -> bool:
        row = self.conn.execute(
            "SELECT claim_bond_units FROM tasks WHERE id=? AND status='assigned' AND assigned_to=? AND lease_token=?",
            (task_id, node_id, lease_token),
        ).fetchone()
        if not row:
            return False
        bond = int(row["claim_bond_units"] or 0)
        if bond:
            self.conn.execute("UPDATE nodes SET credit_units=credit_units+? WHERE id=?", (bond, node_id))
            self.conn.execute(
                "INSERT OR IGNORE INTO ledger(entry_key,node_id,amount_units,reason,reference,created_at) VALUES(?,?,?,?,?,?)",
                (f"bond-refund:{task_id}:{lease_token}", node_id, bond, "task abandoned for local policy change", task_id, time.time()),
            )
        self.conn.execute(
            "UPDATE tasks SET status='queued',assigned_to=NULL,assigned_at=NULL,lease_token=NULL," 
            "lease_expires_at=NULL,lease_deadline_at=NULL,claim_bond_units=0,attempts=MAX(0,attempts-1)," 
            "output=? WHERE id=?",
            (json.dumps({"abandoned": True, "reason": reason}, sort_keys=True), task_id),
        )
        return True


    @synchronized
    def task(self, task_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()

    @synchronized
    def complete_task(self, task_id: str, node_id: str, lease_token: str, output: dict[str, Any]) -> bool:
        with self.lock:
            now = time.time()
            row = self.conn.execute(
                "SELECT claim_bond_units FROM tasks WHERE id=? AND status='assigned' AND assigned_to=? "
                "AND lease_token=? AND lease_expires_at>=? AND lease_deadline_at>=?",
                (task_id, node_id, lease_token, now, now),
            ).fetchone()
            if not row:
                return False
            changed = self.conn.execute(
                "UPDATE tasks SET status='completed',completed_at=?,output=?,claim_bond_units=0 "
                "WHERE id=? AND status='assigned' AND assigned_to=? AND lease_token=?",
                (now, json.dumps(output, sort_keys=True), task_id, node_id, lease_token),
            ).rowcount
            if changed:
                self.conn.execute("UPDATE nodes SET completed=completed+1 WHERE id=?", (node_id,))
            if changed and int(row["claim_bond_units"] or 0):
                bond = int(row["claim_bond_units"])
                self.conn.execute("UPDATE nodes SET credit_units=credit_units+? WHERE id=?", (bond, node_id))
                self.conn.execute(
                    "INSERT OR IGNORE INTO ledger(entry_key,node_id,amount_units,reason,reference,created_at) VALUES(?,?,?,?,?,?)",
                    (f"bond-refund:{task_id}:{lease_token}", node_id, bond, "successful task bond refund", task_id, now),
                )
            return bool(changed)

    @synchronized
    def reject_assigned_task(self, task_id: str, node_id: str, lease_token: str, output: dict[str, Any], reason: str) -> str:
        now = time.time()
        row = self.conn.execute(
            "SELECT attempts,max_attempts,excluded_nodes FROM tasks WHERE id=? AND status='assigned' "
            "AND assigned_to=? AND lease_token=?",
            (task_id, node_id, lease_token),
        ).fetchone()
        if not row:
            raise ValueError("task is no longer assigned to this worker")
        excluded = set(json.loads(row["excluded_nodes"] or "[]"))
        excluded.add(node_id)
        terminal = int(row["attempts"]) >= int(row["max_attempts"])
        status = "failed" if terminal else "queued"
        self.conn.execute(
            "UPDATE tasks SET status=?,completed_at=?,output=?,excluded_nodes=?,assigned_to=NULL,assigned_at=NULL," 
            "lease_token=NULL,lease_expires_at=NULL,lease_deadline_at=NULL,claim_bond_units=0 WHERE id=?",
            (status, now if terminal else None, json.dumps(output, sort_keys=True), json.dumps(sorted(excluded)), task_id),
        )
        self.conn.execute(
            "UPDATE nodes SET failed=failed+1,quarantine_until=MAX(quarantine_until,?) WHERE id=?",
            (now + 300.0, node_id),
        )
        if terminal:
            self._refund_inference_if_terminal(task_id, reason)
        return status

    @synchronized
    def credit(self, entry_key: str, node_id: str, amount_units: int, reason: str, reference: str | None = None) -> bool:
        with self.lock:
            outer = self.conn.in_transaction
            if not outer:
                self._transaction()
            try:
                if self.conn.execute("SELECT 1 FROM ledger WHERE entry_key=?", (entry_key,)).fetchone():
                    if not outer:
                        self.conn.rollback()
                    return False
                row = self.conn.execute("SELECT credit_units FROM nodes WHERE id=?", (node_id,)).fetchone()
                if not row:
                    raise ValueError("unknown node")
                if int(row["credit_units"]) + int(amount_units) < 0:
                    raise ValueError("insufficient credits")
                self.conn.execute(
                    "UPDATE nodes SET credit_units=credit_units+? WHERE id=?",
                    (int(amount_units), node_id),
                )
                self.conn.execute(
                    "INSERT INTO ledger(entry_key,node_id,amount_units,reason,reference,created_at) VALUES(?,?,?,?,?,?)",
                    (entry_key, node_id, int(amount_units), reason, reference, time.time()),
                )
                if not outer:
                    self.conn.commit()
                return True
            except Exception:
                if not outer:
                    self.conn.rollback()
                raise

    @synchronized
    def record_work_report(self, work_key: str, kind: str, node_id: str, task_id: str, output: dict[str, Any]) -> bool:
        from dendriswarm.core.crypto import content_hash
        output_hash = content_hash(output)
        try:
            self.conn.execute(
                "INSERT INTO work_reports(work_key,kind,node_id,task_id,output_hash,output,created_at) VALUES(?,?,?,?,?,?,?)",
                (work_key, kind, node_id, task_id, output_hash, json.dumps(output, sort_keys=True), time.time()),
            )
            return True
        except sqlite3.IntegrityError:
            row = self.conn.execute("SELECT output_hash FROM work_reports WHERE task_id=?", (task_id,)).fetchone()
            if row and row["output_hash"] == output_hash:
                return False
            raise ValueError("conflicting work report")

    @synchronized
    def work_reports(self, work_key: str, kind: str | None = None) -> list[sqlite3.Row]:
        if kind is None:
            return self.conn.execute(
                "SELECT * FROM work_reports WHERE work_key=? ORDER BY created_at", (work_key,)
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM work_reports WHERE work_key=? AND kind=? ORDER BY created_at", (work_key, kind)
        ).fetchall()

    @synchronized
    def add_dataset(self, artifact: dict[str, Any], contributor: str, status: str = "pending") -> str:
        dataset_id = uuid.uuid4().hex
        with self.lock:
            try:
                self.conn.execute(
                    """INSERT INTO datasets(
                        id,name,contributor,license,source,description,content_hash,artifact,status,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        dataset_id,
                        artifact["name"],
                        contributor,
                        artifact["license"],
                        artifact.get("source", ""),
                        artifact.get("description", ""),
                        artifact["sha256"],
                        json.dumps(artifact, sort_keys=True),
                        status,
                        time.time(),
                    ),
                )
                return dataset_id
            except sqlite3.IntegrityError:
                row = self.conn.execute("SELECT id FROM datasets WHERE content_hash=?", (artifact["sha256"],)).fetchone()
                if row:
                    return str(row["id"])
                raise

    @synchronized
    def dataset_by_hash(self, value: str, approved_only: bool = False) -> sqlite3.Row | None:
        sql = "SELECT * FROM datasets WHERE content_hash=?"
        args: tuple[Any, ...] = (value,)
        if approved_only:
            sql += " AND status='approved'"
        return self.conn.execute(sql, args).fetchone()

    @synchronized
    def approved_dataset(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM datasets WHERE status='approved' ORDER BY created_at LIMIT 1").fetchone()

    @synchronized
    def add_candidate(
        self,
        artifact: dict[str, Any],
        config: dict[str, Any],
        dataset_hash: str,
        trainer_node: str,
        training_task: str,
        train_accuracy: float,
        coordinator_accuracy: float,
        hidden_accuracy: float,
        verification_quorum: int,
        trainer_nodes: list[str] | None = None,
    ) -> str:
        existing = self.conn.execute("SELECT id FROM candidates WHERE artifact_hash=?", (artifact["sha256"],)).fetchone()
        if existing:
            return str(existing["id"])
        candidate_id = uuid.uuid4().hex
        try:
            self.conn.execute(
                """INSERT INTO candidates(
                    id,artifact_hash,artifact,config,dataset_hash,trainer_node,trainer_nodes,training_task,
                    train_accuracy,coordinator_accuracy,hidden_accuracy,test_accuracy,verification_quorum,status,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL,?,'candidate',?)""",
                (
                    candidate_id,
                    artifact["sha256"],
                    json.dumps(artifact, sort_keys=True),
                    json.dumps(config, sort_keys=True),
                    dataset_hash,
                    trainer_node,
                    json.dumps(sorted(set(trainer_nodes or [trainer_node]))),
                    training_task,
                    train_accuracy,
                    coordinator_accuracy,
                    hidden_accuracy,
                    verification_quorum,
                    time.time(),
                ),
            )
            return candidate_id
        except sqlite3.IntegrityError:
            row = self.conn.execute("SELECT id FROM candidates WHERE training_task=?", (training_task,)).fetchone()
            if row:
                return str(row["id"])
            raise

    @synchronized
    def candidate(self, candidate_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()

    @synchronized
    def candidate_by_hash(self, value: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM candidates WHERE artifact_hash=?", (value,)).fetchone()

    @synchronized
    def record_verification(self, candidate_id: str, verifier_node: str, task_id: str, accuracy: float) -> bool:
        try:
            self.conn.execute(
                """INSERT INTO candidate_verifications(candidate_id,verifier_node,task_id,accuracy,created_at)
                   VALUES(?,?,?,?,?)""",
                (candidate_id, verifier_node, task_id, accuracy, time.time()),
            )
            return True
        except sqlite3.IntegrityError:
            row = self.conn.execute(
                "SELECT accuracy FROM candidate_verifications WHERE candidate_id=? AND verifier_node=?",
                (candidate_id, verifier_node),
            ).fetchone()
            if row and abs(float(row["accuracy"]) - accuracy) <= 1e-12:
                return False
            raise ValueError("conflicting or duplicate candidate verification")

    @synchronized
    def candidate_verifications(self, candidate_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM candidate_verifications WHERE candidate_id=? ORDER BY created_at",
            (candidate_id,),
        ).fetchall()

    @synchronized
    def finalize_candidate(self, candidate_id: str, min_accuracy: float, tolerance: float = 1e-12) -> bool:
        with self.lock:
            candidate = self.candidate(candidate_id)
            if not candidate:
                raise ValueError("unknown candidate")
            verifications = self.candidate_verifications(candidate_id)
            if len(verifications) < int(candidate["verification_quorum"]):
                return False
            values = [float(row["accuracy"]) for row in verifications]
            if max(values) - min(values) > tolerance:
                self.conn.execute("UPDATE candidates SET status='rejected' WHERE id=?", (candidate_id,))
                return False
            accuracy = sum(values) / len(values)
            if accuracy < min_accuracy:
                self.conn.execute(
                    "UPDATE candidates SET test_accuracy=?,hidden_accuracy=?,status='rejected' WHERE id=?",
                    (accuracy, accuracy, candidate_id),
                )
                return False
            self.conn.execute(
                "UPDATE candidates SET test_accuracy=?,coordinator_accuracy=?,hidden_accuracy=?,status='verified' "
                "WHERE id=? AND status!='canonical'",
                (accuracy, accuracy, accuracy, candidate_id),
            )
            best = self.conn.execute(
                """SELECT * FROM candidates WHERE status IN ('verified','canonical')
                   ORDER BY test_accuracy DESC, created_at ASC LIMIT 1"""
            ).fetchone()
            if best:
                self.conn.execute("UPDATE candidates SET status='verified' WHERE status='canonical' AND id!=?", (best["id"],))
                self.conn.execute("UPDATE candidates SET status='canonical' WHERE id=?", (best["id"],))
            return True


    @synchronized
    def reject_candidate(self, candidate_id: str) -> None:
        self.conn.execute("UPDATE candidates SET status='rejected' WHERE id=?", (candidate_id,))

    @synchronized
    def fail_task(self, task_id: str, node_id: str, output: dict[str, Any]) -> None:
        now = time.time()
        self.conn.execute(
            "UPDATE tasks SET status='failed',completed_at=?,output=? WHERE id=?",
            (now, json.dumps(output, sort_keys=True), task_id),
        )
        self.conn.execute("UPDATE nodes SET failed=failed+1 WHERE id=?", (node_id,))

    @synchronized
    def canonical_candidate(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM candidates WHERE status='canonical' LIMIT 1").fetchone()

    def _refund_inference_if_terminal(self, task_id: str, reason: str) -> bool:
        row = self.conn.execute(
            "SELECT request_key,node_id,cost_units,refunded FROM inference_requests WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row or int(row["refunded"] or 0):
            return False
        self.conn.execute("UPDATE nodes SET credit_units=credit_units+? WHERE id=?", (int(row["cost_units"]), row["node_id"]))
        self.conn.execute("UPDATE inference_requests SET refunded=1 WHERE task_id=?", (task_id,))
        self.conn.execute(
            "INSERT OR IGNORE INTO ledger(entry_key,node_id,amount_units,reason,reference,created_at) VALUES(?,?,?,?,?,?)",
            (f"inference-refund:{row['request_key']}", row["node_id"], int(row["cost_units"]), f"inference failure refund: {reason}", task_id, time.time()),
        )
        return True

    @synchronized
    def create_inference_task(
        self,
        request_key: str,
        node_id: str,
        artifact_hash: str,
        features: list[float],
        cost_units: int,
        reward_units: int,
    ) -> tuple[str, bool]:
        with self.lock:
            self._transaction()
            try:
                existing = self.conn.execute(
                    "SELECT task_id FROM inference_requests WHERE request_key=?", (request_key,)
                ).fetchone()
                if existing:
                    self.conn.rollback()
                    return str(existing["task_id"]), False
                row = self.conn.execute("SELECT credit_units FROM nodes WHERE id=?", (node_id,)).fetchone()
                if not row:
                    raise ValueError("unknown credit account")
                if float(row["credit_units"]) < cost_units:
                    raise ValueError("insufficient credits")
                task_id = uuid.uuid4().hex
                now = time.time()
                self.conn.execute(
                    """INSERT INTO tasks(
                        id,dedupe_key,kind,payload,requirements,reward_units,priority,status,created_at,max_attempts,
                        parent_task,excluded_nodes
                    ) VALUES(?,?,?,?,?,?,100,'queued',?,3,NULL,?)""",
                    (
                        task_id,
                        f"inference:{request_key}",
                        TaskKind.INFERENCE.value,
                        json.dumps({"artifact_hash": artifact_hash, "features": features, "claim_bond_units": 4000}, sort_keys=True),
                        json.dumps(requirements_from_value(None, TaskKind.INFERENCE).model_dump(mode="json"), sort_keys=True),
                        reward_units,
                        now,
                        json.dumps([node_id]),
                    ),
                )
                self.conn.execute("UPDATE nodes SET credit_units=credit_units-? WHERE id=?", (cost_units, node_id))
                self.conn.execute(
                    "INSERT INTO ledger(entry_key,node_id,amount_units,reason,reference,created_at) VALUES(?,?,?,?,?,?)",
                    (f"debit:{request_key}", node_id, -cost_units, "model usage", task_id, now),
                )
                self.conn.execute(
                    "INSERT INTO inference_requests(request_key,node_id,task_id,cost_units,created_at) VALUES(?,?,?,?,?)",
                    (request_key, node_id, task_id, cost_units, now),
                )
                self.conn.commit()
                return task_id, True
            except Exception:
                self.conn.rollback()
                raise

    @synchronized
    def inference_request(self, request_key: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM inference_requests WHERE request_key=?", (request_key,)).fetchone()

    @synchronized
    def append_audit(self, event_type: str, body: dict[str, Any]) -> str:
        import hashlib
        from dendriswarm.core.crypto import canonical_json
        with self.lock:
            row = self.conn.execute("SELECT event_hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = str(row["event_hash"]) if row else "0" * 64
            created_at = time.time()
            material = {"event_type": event_type, "body": body, "prev_hash": prev_hash, "created_at": created_at}
            event_hash = hashlib.sha256(canonical_json(material)).hexdigest()
            self.conn.execute(
                "INSERT INTO audit(event_type,body,prev_hash,event_hash,created_at) VALUES(?,?,?,?,?)",
                (event_type, json.dumps(body, sort_keys=True), prev_hash, event_hash, created_at),
            )
            return event_hash

    @synchronized
    def validate_audit_chain(self) -> tuple[bool, int, str]:
        import hashlib
        from dendriswarm.core.crypto import canonical_json
        previous = "0" * 64
        count = 0
        for row in self.conn.execute("SELECT * FROM audit ORDER BY seq"):
            body = json.loads(row["body"])
            material = {"event_type": row["event_type"], "body": body, "prev_hash": row["prev_hash"], "created_at": row["created_at"]}
            expected = hashlib.sha256(canonical_json(material)).hexdigest()
            if row["prev_hash"] != previous or row["event_hash"] != expected:
                return False, count, previous
            previous = row["event_hash"]
            count += 1
        return True, count, previous

    @synchronized
    def set_metadata(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, sort_keys=True)),
        )

    @synchronized
    def get_metadata(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return None if not row else json.loads(row["value"])


    @synchronized
    def ledger_entries(self, node_id: str, limit: int = 50) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT amount_units,reason,reference,created_at FROM ledger WHERE node_id=? ORDER BY id DESC LIMIT ?",
            (node_id, int(limit)),
        ))

    @synchronized
    def stats(self) -> dict[str, Any]:
        self.requeue_expired_leases()
        q = lambda sql, args=(): self.conn.execute(sql, args).fetchone()[0]
        canonical = self.canonical_candidate()
        audit_valid, audit_events, audit_head = self.validate_audit_chain()
        return {
            "nodes": q("SELECT COUNT(*) FROM nodes"),
            "active_nodes": q("SELECT COUNT(*) FROM nodes WHERE last_seen > ?", (time.time() - 60,)),
            "paused_nodes": sum(
                1 for row in self.conn.execute("SELECT policy FROM nodes")
                if SeedPolicy.model_validate(json.loads(row["policy"] or "{}")).paused
            ),
            "architectures": {
                machine: count for machine, count in self.conn.execute(
                    "SELECT json_extract(capabilities, '$.machine') AS machine, COUNT(*) FROM nodes GROUP BY machine"
                ) if machine
            },
            "queued_tasks": q("SELECT COUNT(*) FROM tasks WHERE status='queued'"),
            "assigned_tasks": q("SELECT COUNT(*) FROM tasks WHERE status='assigned'"),
            "completed_tasks": q("SELECT COUNT(*) FROM tasks WHERE status='completed'"),
            "failed_tasks": q("SELECT COUNT(*) FROM tasks WHERE status='failed'"),
            "datasets_pending": q("SELECT COUNT(*) FROM datasets WHERE status='pending'"),
            "datasets_approved": q("SELECT COUNT(*) FROM datasets WHERE status='approved'"),
            "candidates": q("SELECT COUNT(*) FROM candidates"),
            "verified_candidates": q("SELECT COUNT(*) FROM candidates WHERE status IN ('verified','canonical')"),
            "verification_records": q("SELECT COUNT(*) FROM candidate_verifications"),
            "canonical": None
            if not canonical
            else {
                "artifact_hash": canonical["artifact_hash"],
                "test_accuracy": canonical["test_accuracy"],
                "hidden_accuracy": canonical["hidden_accuracy"],
                "train_accuracy": canonical["train_accuracy"],
                "config": json.loads(canonical["config"]),
                "verification_quorum": canonical["verification_quorum"],
                "verifications": len(self.candidate_verifications(str(canonical["id"]))),
            },
            "credit_supply_units": int(q("SELECT COALESCE(SUM(credit_units),0) FROM nodes")),
            "credit_supply": int(q("SELECT COALESCE(SUM(credit_units),0) FROM nodes")) / 1000.0,
            "benchmark": self.get_metadata("benchmark"),
            "verification_modes": {
                "exploration": "replicated-consensus",
                "training": "replicated-artifact-consensus",
                "verification": "independent-replicated-consensus",
                "inference": "proof-carrying-output-plus-secret-spot-audit",
            },
            "audit": {"valid": audit_valid, "events": audit_events, "head": audit_head},
        }
