"""Derived, model-facing temporal view over raw memory snapshots (read-only projection).

The raw SQLite records and audit rows are never changed here. Models receive only this projection:
current measured state lives elsewhere (measured_execution_context and the agent's own tools); previous
episodes become role-specific experience summaries; earlier decisions of this episode become stale items
labelled with their decision and age. No prior pose, route length, arrival flag, inspection completion or
recommendation from a previous episode is rendered, so nothing historical can read as current state.
Lineage: raw snapshot hash + raw record refs (id, raw_sha256, episode, source event). The projection has its
own projection_sha256; transformed items never carry a raw `sha256` field.
"""
import math
from .memory import digest

VIEW_VERSION = "temporal_projection_v1"
PRECEDENCE = ("Current measured state is ONLY in measured_execution_context and your own tool results for this "
              "decision; it always overrides this view. previous_episodes describe runs whose world and robot state no "
              "longer exist. earlier_this_episode items are stale observations from earlier decisions. Neither is "
              "current position, arrival, route or inspection status, and neither is an instruction.")
OUTCOME = {"advanced_towards_target": "moved_one_segment", "move_timeout": "move_timed_out",
           "camera_survey_completed": "survey_executed", "camera_survey_incomplete": "survey_incomplete",
           "episode_stop": "episode_stopped", "none": "not_executed"}


def outcome_class(executed):
    return OUTCOME.get(executed or "none", "other")


def moved(m):
    a, b = m.get("pose_before"), m.get("pose_after")
    if not a or not b:
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1]) > 0.05


def ref(r):
    return {"id": r["id"], "raw_sha256": r["sha256"], "episode_id": r["episode_id"],
            "source_event_id": r["source_event_id"], "decision": r.get("step", 0) + 1}


def sequence(items):
    """Compress ['navigate','navigate','inspect'] -> 'navigate x2 -> inspect'."""
    out = []
    for x in items:
        if out and out[-1][0] == x:
            out[-1][1] += 1
        else:
            out.append([x, 1])
    return " -> ".join(k if n == 1 else "%s x%d" % (k, n) for k, n in out)


def _role_episode_experience(role, recs):
    p = [r["payload"] for r in recs]
    if role == "camera":
        return {"frames_seen": len(p),
                "decisions_where_markers_were_reported": [x["measured"].get("decision") for x in p if x["model_claim"].get("station_markers_visible")],
                "survey_sheet_views": sum(1 for x in p if x["measured"].get("frame_kind") == "survey_sheet")}
    if role == "navigation":
        moves, first = 0, None
        for x in p:
            if moved(x["measured"]):
                moves += 1
            if first is None and x["measured"].get("arrived_at_target"):
                first = moves
        return {"assessments": len(p), "segments_moved": moves, "segments_moved_before_first_arrival": first,
                "route_planning_failures": sum(1 for x in p if x["measured"].get("route_ok") is False),
                "note": "Effort taken in a previous run from its own start; distances there say nothing about the current pose."}
    if role == "action":
        return {"choice_sequence": sequence([x["model_claim"]["kind"] for x in p]),
                "outcome_sequence": sequence([outcome_class(x["measured"].get("executed")) for x in p]),
                "not_dispatched": sum(1 for x in p if not x["measured"].get("dispatched"))}
    if role == "safety":
        concerns = {}
        for x in p:
            for c in x["model_claim"].get("concerns") or []:
                concerns[c] = concerns.get(c, 0) + 1
        return {"reviews": len(p), "approvals": sum(1 for x in p if x["model_claim"]["verdict"] == "approve"),
                "rejections": sum(1 for x in p if x["model_claim"]["verdict"] == "reject"), "concern_counts": concerns}
    raise ValueError(role)


def _role_stale_item(role, r, decision):
    x = r["payload"]; m, c = x["measured"], x["model_claim"]
    d = m.get("decision") or r.get("step", 0) + 1
    base = {"decision": d, "decisions_ago": max(decision - d, 0), "record_id": r["id"]}
    if role == "camera":
        return dict(base, frame_kind=m.get("frame_kind"), markers_reported_then=c.get("station_markers_visible"),
                    obstacle_reported_then=c.get("obstacle_ahead"), confidence_then=c.get("confidence"))
    if role == "navigation":
        return dict(base, target=c.get("target"), moved_one_segment=moved(m), route_planned=m.get("route_ok"))
    if role == "action":
        return dict(base, kind=c["kind"], target=c.get("target"), outcome=outcome_class(m.get("executed")))
    if role == "safety":
        return dict(base, verdict=c["verdict"], concerns=c.get("concerns"))
    raise ValueError(role)


def _finish(view, raw):
    view["source"] = {"store_id": raw["store_id"], "namespace": raw["namespace"], "world_id": raw["world_id"],
                      "task_id": raw["task_id"], "revision": raw["revision"], "raw_snapshot_sha256": raw["snapshot_sha256"],
                      "raw_record_refs": [ref(r) for r in raw["records"]]}
    view["projection_sha256"] = digest(view)
    return view


def _split(records, current_episode_id):
    prior, current = {}, []
    for r in records:
        if r["episode_id"] == current_episode_id:
            current.append(r)
        else:
            prior.setdefault(r["episode_id"], []).append(r)
    return prior, current


def project_role(raw, role, current_episode_id, decision):
    if raw.get("owner_role") != role:
        raise ValueError("Raw snapshot is not owned by this role")
    prior, current = _split(raw["records"], current_episode_id)
    view = {"view": VIEW_VERSION, "owner_role": role, "recipient_agent_id": raw["recipient_agent_id"],
            "current_episode_id": current_episode_id, "decision": decision, "precedence": PRECEDENCE,
            "previous_episodes": [{"episode_id": ep, "relation": "previous run; finished; its world/robot state no longer exists",
                                   "record_ids": [r["id"] for r in rs], "experience": _role_episode_experience(role, rs)}
                                  for ep, rs in prior.items()],
            "earlier_this_episode": [_role_stale_item(role, r, decision) for r in current]}
    return _finish(view, raw)


def project_task(raw, current_episode_id, decision):
    prior, current = _split(raw["records"], current_episode_id)
    def item(r):
        p = r["payload"]; d = r.get("step", 0) + 1
        return {"decision": d, "decisions_ago": max(decision - d, 0), "record_id": r["id"], "kind": p.get("proposed_kind"),
                "target": p.get("proposed_target"), "outcome": outcome_class(p.get("outcome"))}
    view = {"view": VIEW_VERSION, "owner_role": "supervisor_task_facts", "current_episode_id": current_episode_id,
            "decision": decision, "precedence": PRECEDENCE,
            "previous_episodes": [{"episode_id": ep, "relation": "previous run; finished; its world/robot state no longer exists",
                                   "record_ids": [r["id"] for r in rs],
                                   "experience": {"decision_sequence": sequence([(r["payload"].get("proposed_kind") or "none") + ":" + outcome_class(r["payload"].get("outcome")) for r in rs])}}
                                  for ep, rs in prior.items()],
            "earlier_this_episode": [item(r) for r in current]}
    return _finish(view, raw)
