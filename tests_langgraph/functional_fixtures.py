"""Software fixtures for functional-agent tests. Deterministic test doubles, labeled as such."""
import hashlib
import json
from pathlib import Path
from robot_controller.functional import FunctionalRuntime, FunctionalBudgets, FunctionalLedger, build_agents, ROLES
from robot_controller.ollama import Frame
from robot_controller.schemas import Task, gate

TASK = Task("Go to Station 1, inspect the surrounding area using the camera, and report what is present.",
            "station_1", ("station_1", "station_2"))
PNG = b"\x89PNG\r\n\x1a\n" + b"neutral software fixture frame"


class ScriptedModel:
    """Implements the OllamaModel.complete API; records every call; computes the wire hash like the adapter."""
    model = "deterministic-test-double"
    model_digest = "test-fixture-not-a-real-model"

    def __init__(self, overrides=None, log=None):
        self.calls = []
        self.overrides = overrides or {}
        self.log = Path(log) if log else None
        self.last_wire_sha256 = None
        self.last_image_sha256 = None

    def complete(self, role, system, payload, schema, frame=None):
        content = json.dumps(payload, ensure_ascii=False)
        self.last_wire_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.last_image_sha256 = frame.sha256 if frame is not None else None
        self.calls.append({"role": role, "payload": payload, "schema": schema,
                           "frame": frame.sha256 if frame is not None else None})
        if self.log:
            with self.log.open("a") as f:
                f.write(json.dumps(self.calls[-1]) + "\n")
        n = sum(1 for c in self.calls if c["role"] == role)
        o = self.overrides.get(role)
        if callable(o):
            r = o(n, payload, schema)
            if r is not None:
                return r
        elif isinstance(o, list) and n <= len(o) and o[n-1] is not None:
            return o[n-1]
        return default(role, payload, schema)


def default(role, payload, schema):
    if role == "supervisor":
        allowed = schema["properties"]["next"]["enum"]
        for choice in ("camera", "navigation", "action", "safety", "finish"):
            if choice in allowed and (choice not in payload["invocations_this_decision"] or choice in ("safety", "finish")):
                return {"next": choice, "request": "Provide your role's assessment.", "reason": "Evidence for this role is missing."}
        return {"next": allowed[0], "request": "Continue.", "reason": "Only option."}
    if role == "camera":
        return {"observation": "Floor station outlines ahead; nothing blocks the path.", "station_markers_visible": True,
                "obstacle_ahead": False, "confidence": "medium"}
    if role == "navigation":
        return {"target": "station_1", "route_status": "route_available", "recommendation": "advance", "note": "A* route exists."}
    if role == "action":
        ids = schema["anyOf"][0]["properties"]["evidence_handoffs"]["items"]["enum"]
        return {"kind": "navigate", "target": "station_1", "reason": "Advance toward Station 1.", "evidence_handoffs": ids[:2]}
    if role == "safety":
        return {"verdict": "approve", "concerns": ["none"], "reason": "Serves the operator task."}
    raise AssertionError(role)


def tools(tmp, pose=(-2.0, 0.0, 0.0), extra=None):
    tmp = Path(tmp)
    counter = {"n": 0}
    def capture(decision):
        counter["n"] += 1
        p = tmp / ("frame_%02d_%d.png" % (decision, counter["n"]))
        p.write_bytes(PNG + str(counter["n"]).encode())
        return {"kind": "forward", "frame": Frame(p, source="software_fixture_camera")}
    def sheet(decision):
        return {"available": False}
    t = {"camera": {"camera.capture_forward": capture, "camera.survey_sheet": sheet},
         "navigation": {"nav.pose": lambda: {"pose": list(pose)},
                        "nav.target_state": lambda: {"station_1": {"distance_to_estimated_m": 4.2, "at_target": False},
                                                     "station_2": {"distance_to_estimated_m": 4.9, "at_target": False}},
                        "nav.astar_route": lambda target: {"ok": True, "length_m": 4.3, "waypoints": 5},
                        "nav.segment_preview": lambda target: {"ok": True, "commanded": [-1.4, 0.3, 0.5], "segment_m": 0.65}},
         "action": {"action.capabilities": lambda: {"kinds": ["navigate", "inspect", "stop"], "segment_max_m": 0.65}},
         "safety": {"safety.envelope_check": lambda action: {"ok": True, "reason": "ok"},
                    "safety.task_gate": lambda action: gate(TASK, __import__("robot_controller.schemas", fromlist=["Action"]).Action.parse(action), True),
                    "safety.preconditions": lambda action: {"at_target": False}}}
    if extra:
        for role, mapping in extra.items():
            t[role].update(mapping)
    return t


def runtime(tmp, model, episode="ep_fixture", memory=None, budgets=FunctionalBudgets(), tool_set=None, ledger_root=None):
    ledger = FunctionalLedger(Path(ledger_root or tmp) / "ledger", run_id=episode)
    agents = build_agents(episode, tool_set or tools(tmp))
    return FunctionalRuntime(TASK, model, ledger, agents, episode_id=episode, world_id="world_fixture",
                             memory=memory, budgets=budgets, envelope={"segment_max_m": 0.65, "arrival_tolerance_m": 0.28})


def executor_record(result):
    p = result["proposal"]
    ok = result["gate"]["allowed"]
    return {"proposal": p, "gate_allowed": ok, "gate_reason": result["gate"]["reason"], "dispatched": ok,
            "executed": "advanced_towards_target" if ok else "none", "pose_before": [-2.0, 0.0, 0.0],
            "pose_after": [-1.4, 0.3, 0.5] if ok else [-2.0, 0.0, 0.0], "displacement_m": 0.67 if ok else 0,
            "arrived_at_target": False, "at_station_1_after": False}
