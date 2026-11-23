# Persistent observation/task memory

## What changed
The new runner uses `ObservationStore`, a local SQLite WAL database, instead of relying only on the last five execution-history entries. A new graph is still created per decision; the disk store is independent of graph lifetime. No LangGraph checkpoint is attached.

`--memory-mode measured` (the runner default) enables it. `--memory-mode off` preserves memory-free operation. `--memory-db` selects the disk file, `--memory-namespace` selects a named cohort, and `--memory-limit` bounds recalled records to 1–8. The default database is `.robot-memory/observations.sqlite3` beneath the runner checkout. A new namespace starts empty without deleting prior evidence. Process restart does not clear the database. Do not silently share a namespace between experimental conditions.

## Two different kinds of stored evidence
- **Execution facts:** typed action outcome, measured pose/displacement, arrival and camera-survey count. These are automatically retrieved as *historical* records for the same namespace, immutable task fingerprint and world-layout hash. Episode IDs remain attached. A layout hash is not proof that objects remained unchanged across resets; earlier inspections never fulfill a fresh task.
- **Model claims:** worker summaries and supervisor reasons, with actual source event, role, episode and recipient-specific frame provenance. These persist for read-only inspection, but are excluded from automatic policy recall. Quoting scene text cannot install it as a lasting instruction or modify agent/policy files. This is a protected observation-memory implementation, not unrestricted directive replay.

Only the runner writes, after a recorded executor outcome. Workers do not get a write tool. System/task instructions, tool catalogue, model parameters and executable files are not stored or rewritten by memory.

## Evidence
Each immutable record has an ID/hash and source event. SQLite commits records and write-audit rows atomically. Retrieval yields an immutable bounded snapshot with exact IDs/revision/hash. Each supervisor or worker call logs `memory_read` and its full input, and each saved record logs `memory_write`. The runner saves per-decision receipts in `steps.jsonl` and a `memory_state.json` summary. Existing Runtime workers can show these event records. These logs establish data flow, not causal influence.

## Validation and scope
Tests cover fresh-process reopen, task/world/namespace isolation, idempotency, conflicting rewrites, concurrent writes, corrupted-record rejection, all three worker roles receiving the recorded snapshot, and untrusted claims remaining outside the recalled fact store. Test models are deterministic doubles; these are software tests, not robot outcomes. The repair branch uses the benign Gazebo baseline; the separate paper-condition runner and original working trees are unchanged. No robot trial or publication is part of this repair.
