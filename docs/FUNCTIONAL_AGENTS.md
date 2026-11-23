# Functional robot agents: contract (milestone priorities 1-4)

Branch `feature/functional-robot-agents`. Controller base 7dd16ec, Gazebo base 22a53a7.
Benign, simulator-only Station 1 inspection. No paper stimuli, propagation, directive/config rewriting,
hardware control or generated executable agents.

## Public interfaces

Controller (`robot_controller/functional.py`):
- `FunctionalAgent` subclasses `CameraAgent`, `NavigationAgent`, `ActionAgent`, `SafetyAgent`.
  `agent_id = "<role>_agent.<episode_id>"`, stable for every decision of one episode (one process).
- `FunctionalRuntime(task, model, ledger, agents, memory=None, budgets=FunctionalBudgets(), episode_id, world_id)`
  `.decide(decision, execution_context) -> DecisionResult` (proposal, gate, handoffs, per-role calls).
  `.record_outcome(decision, executor_record, executor_event_id)` writes role-owned memory.
- `ObservationStore.append_role_record(role, payload, **source)` / `.role_snapshot(role, recipient_agent_id, ...)`
  (additive record kinds `role_camera|role_navigation|role_action|role_safety`; schema 1 unchanged).
- `OllamaModel.complete(...)` unchanged signature; additionally sets `last_wire_sha256` = SHA-256 of the exact
  user-message content string sent.

Gazebo (`gazebo/`):
- `functional_tools.py`: role tool bindings over existing primitives (`sim.snapshot`, `sim.pose`,
  `planner.plan`, `mission_logic.advance_target`, `gazebo_adapter.envelope_check/target_state`, `schemas.gate`).
- `functional_mission.py`: episode runner (same exploration/setup as `langgraph_mission.py`; same `gazebo_adapter.Executor`).
- `run_functional_mission.py`: source-pinned launcher, asserts imported `robot_controller.__file__`.
- `watchstate.py`: canonical atomic JSON on the WSL native filesystem + bounded, nonfatal Windows mirror.
- `functional_rollout.py`: finite write/reopen pair driver (2 episodes, same DB, fresh processes).

## Agents, lifecycle and ownership

Lifecycle stages are events (`stage: observe|plan|check`) emitted by an agent; they are not agents.
observe = call the agent's own registered tools; plan = one role-specific model call; check = strict
validation of the model output (fail closed: invalid/missing output -> agent_failed, no action for the decision).

| Agent | Tools (only these) | Model input | Output (typed) | Role memory written |
|---|---|---|---|---|
| camera | `camera.capture_forward` (sim.snapshot), `camera.survey_sheet` (frames from executed survey) | frame image + meta, own history, own role memory | observation, markers_visible, obstacle_ahead, confidence | frame hash/source + own observation claim (untrusted) |
| navigation | `nav.pose`, `nav.astar_route` (planner.plan), `nav.segment_preview` (advance_target, 0.65 m), `nav.target_state` | measured pose, A* route facts, camera handoff text, own memory | target (catalogue), route_status, recommendation, note (+ code-computed claim_conflicts_measurement) | measured pose/route/arrival + own assessment |
| action | `action.capabilities` (read-only executor vocabulary/bounds) | frame, camera+navigation handoffs, safety feedback, own memory | `{kind: navigate|inspect|stop, target, reason, evidence_handoffs}` | choice + measured executor outcome |
| safety | `safety.envelope_check`, `safety.task_gate` (schemas.gate), `safety.preconditions` | frame, proposal, deterministic results, handoffs, own memory | verdict approve/reject, concerns, reason | verdict + deterministic results |

Each agent keeps private in-process state (`history` of its last outputs, `state_version`, `state_sha256`).
No agent sees another agent's history or memory; only explicit handoffs cross agents.
Admission = safety approve AND deterministic gate allowed AND envelope ok. Executor bounds (registered
kinds, catalogue targets, 0.65 m A* segment, territory, 0.28 m arrival tolerance, move timeout, inspect only
when measured at target) stay in `gazebo_adapter.Executor`, outside model control. No action is forced or
substituted; a failed agent or unapproved proposal executes nothing.

## Supervisor

Model call per routing round, schema `{next: enum(allowed), request, reason}`; `allowed` computed in code from
current evidence: camera/navigation always (<=2 each per decision), action needs camera+navigation handoffs this
decision, safety needs an unreviewed proposal, finish needs a reviewed proposal. Budgets per decision:
supervisor calls <= 8, agent invocations <= 7; per episode: model calls <= 260, decisions <= 16.

## Event schema (events.jsonl, one ledger per episode, schema_version 2)

Common: `schema_version, run_id, event_id, sequence, timestamp_ns, event_type, robot_id, agent_id, parent_event_ids, data`.
`data` adds: `role, stage, decision, round, state_before{version,sha256}, state_after{...}, handoff_ids,
memory_record_ids_read, frame{sha256,source,...}`.
Types: `run_started, decision_started, route_decision, agent_invoked, stage, tool_call, tool_result, tool_denied,
memory_read, model_input(payload, payload_sha256), model_output(output, wire_sha256), model_error, agent_failed,
handoff_created, handoff_delivered, safety_decision, decision_proposal, executor_dispatch_request, executor_result,
memory_write, decision_completed, run_completed`.
`memory_read`: `store_id, namespace, recipient_agent_id, owner_role, record_ids, records:[{id, sha256, episode_id,
source_event_id, prior_episode}], snapshot_sha256, revision`.

## Watch state

Canonical: `~/mana-trials/functional/<batch>/batch.json` (WSL ext4, unique tmp + fsync + os.replace).
Mirror: `/mnt/c/.../mana-monitor/functional/<batch>/batch.json`, <=3 attempts with backoff, failures recorded in
canonical `mirror` stats, never raised. A failed replace leaves the previous valid mirror intact.

## Resets (declared per episode)
controller process: fresh; agent private history: empty at process start; graph checkpoint: none;
memory DB: same file reopened; simulator world: fresh Gazebo; Ollama server/model cache: retained (no hidden-state reset claim).
