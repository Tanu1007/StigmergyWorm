# State repair contract: current-state precedence (fix/current-state-precedence)

Bases: controller 04c724d, Gazebo 7e136bb. Benign task-reliability repair only.

## Root causes (from the preserved failure functional_20261123T024024_reopen)
1. role_snapshot raw records (prior pose, route_length_m, arrived_at_target, previous recommendation) were sent
   verbatim to models; navigation emitted inspect_ready at a measured 4.2138 m.
2. NavigationAgent.validate only flagged claim_conflicts_measurement; check stage "passed" and the claim was handed off.
3. No recovery path for a semantic validation failure (only safety rejections re-opened routing).

## Repair
- `robot_controller/temporal.py` builds a derived, model-facing temporal view from a raw snapshot. Raw SQLite
  records and audit rows are never modified by projection. Sections: `previous_episodes` (role-specific
  experience summaries, no prior pose/route length/arrival/inspection-completion/recommendation values),
  `earlier_this_episode` (stale per-decision items labelled with decision and age; no pose/route length/
  recommendation), plus `source` lineage: store_id, namespace, raw_snapshot_sha256, raw record refs
  {id, raw_sha256, episode_id, source_event_id, decision}. The view carries its own `projection_sha256`;
  transformed items never carry a raw `sha256` field. Applied to all four roles and to the supervisor's
  task-fact memory. Payload key `role_memory` -> `memory_temporal_view`; supervisor `persistent_task_memory`
  -> `task_memory_temporal_view`. Current measured state stays only in `measured_execution_context` and tools.
- Event `memory_projection` per recipient: raw snapshot hash, record refs, projection hash, full projection.
- NavigationAgent check rejects (event `check_failed`, with proposed claim and measured fact) before handoff:
  inspect_ready or at_target when the measured target is not at target; advance or route_available when the
  measured A* route is unavailable; no_route when a measured route exists; advance when measured at target.
  hold / not_needed / uncertainty are never rejected for being at a station. Rejected output is not added to
  private history or handoffs.
- Bounded recovery: a semantic check failure re-opens that role once per decision (`semantic_recoveries_per_role`
  = 1, per decision total 2). The supervisor receives `failed_checks` (claim + measured fact) and may re-invoke
  the role or `finish` with no action. The re-invoked agent receives `previous_attempt_rejected`. A prior accepted
  handoff from the failed role in this decision is explicitly invalidated (`handoff_invalidated`), and a pending
  proposal depending on it is withdrawn. Exhausted recovery ends the decision with status
  `semantic_check_failed` (no execution). Runner stops the episode after 2 consecutive failed decisions or 16
  semantic check failures per episode (recovered or not) (`semantic_recovery_exhausted`).
- Runner publishes phase changes (legacy_mapper_exploration, setup_return_to_start, mission decision pending
  with decision number) before each decision resolves.

## Restart validation (one run)
SQLite online backup of the completed functional DB (read-only source URI) to a new path; meta `cloned_from`
row added to the copy only. Same namespace, new episode ID, fresh Gazebo and controller, same model/settings,
<=16 decisions, 1800 s, model-call budget 220. Canonical watch on WSL, nonfatal Windows mirror.
