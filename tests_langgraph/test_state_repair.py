"""State-repair regressions. Built on the ACTUAL saved failed reopen input (fixtures/original_failure_reopen_nav_d1.json,
event controller_functional_20261123T024024_reopen_functional_42318d3d:46/47). Deterministic doubles only."""
import hashlib, importlib.util, json, sqlite3, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from functional_fixtures import ScriptedModel, runtime, tools, executor_record, TASK
from robot_controller.functional import NavigationAgent, SemanticCheckFailure, semantic_check, FunctionalBudgets, ROLES
from robot_controller.memory import ObservationStore, canonical
from robot_controller.temporal import project_role, project_task
from robot_controller.schemas import Task

HERE = Path(__file__).resolve().parent
FIX = json.loads((HERE / "fixtures/original_failure_reopen_nav_d1.json").read_text())
FORBIDDEN_KEYS = {"pose_before", "pose_after", "pose", "route_length_m", "arrived_at_target", "at_station_1_after",
                  "recommendation", "route_status", "displacement_m", "distance_to_estimated_m", "survey_frames_count", "sha256"}


def walk(v):
    if isinstance(v, dict):
        for k, x in v.items():
            yield k, x
            yield from walk(x)
    elif isinstance(v, list):
        for x in v:
            yield from walk(x)


def events(rt, kind=None):
    rows = [json.loads(x) for x in rt.ledger.path.read_text().splitlines()]
    return [e for e in rows if kind is None or e["event_type"] == kind]


def evidence_from_saved():
    return FIX["input"]["measured_navigation_tools"]


class SavedFailureTests(unittest.TestCase):
    def test_saved_pattern_is_prior_arrival_and_short_routes_with_current_4_2138(self):
        p = FIX["input"]
        self.assertEqual(p["measured_execution_context"]["targets"]["station_1"]["distance_to_estimated_m"], 4.2138)
        self.assertFalse(p["measured_navigation_tools"]["targets"]["station_1"]["at_target"])
        lengths = [r["payload"]["measured"]["route_length_m"] for r in p["role_memory"]["records"]]
        self.assertIn(0.095, lengths)
        self.assertTrue(any(r["payload"]["measured"]["arrived_at_target"] for r in p["role_memory"]["records"]))
        self.assertEqual(FIX["output"]["recommendation"], "inspect_ready")
        self.assertEqual(hashlib.sha256(json.dumps(p, ensure_ascii=False).encode()).hexdigest(), FIX["inputSha256"])

    def test_old_validator_forwarded_it_and_new_check_rejects_it(self):
        repo = HERE.parent
        try:
            old_src = subprocess.check_output(["git", "show", "04c724d59269e10e920b43213f14bdc69e987a02:robot_controller/functional.py"],
                                              cwd=repo, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("base commit not readable from this checkout")
        spec = importlib.util.spec_from_loader("robot_controller._old_functional", loader=None)
        old = importlib.util.module_from_spec(spec); old.__package__ = "robot_controller"
        exec(compile(old_src, "old_functional.py", "exec"), old.__dict__)
        a = old.NavigationAgent("old_validator", {k: (lambda **kw: None) for k in old.NavigationAgent.ALLOWED_TOOLS})
        t = Task(**{k: (tuple(v) if k == "allowed_targets" else v) for k, v in FIX["input"]["operator_task"].items()})
        forwarded = a.validate(FIX["output"], SimpleNamespace(task=t), {}, evidence_from_saved())
        self.assertEqual(forwarded["recommendation"], "inspect_ready")          # old: passed check and was handed off
        self.assertTrue(forwarded["claim_conflicts_measurement"])
        new = NavigationAgent("new_validator", {k: (lambda **kw: None) for k in NavigationAgent.ALLOWED_TOOLS})
        with self.assertRaises(SemanticCheckFailure) as cm:
            new.validate(FIX["output"], SimpleNamespace(task=t), {}, evidence_from_saved())
        self.assertEqual(cm.exception.reason, "inspect_ready_but_measured_not_at_target")
        self.assertEqual(cm.exception.measured_fact["distance_to_estimated_m"], 4.2138)
        self.assertEqual(cm.exception.claim["recommendation"], "inspect_ready")

    def test_saved_raw_records_project_without_current_looking_values(self):
        raw_recs = FIX["input"]["role_memory"]["records"]
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "m.sqlite3", FIX["input"]["role_memory"]["namespace"])
            with sqlite3.connect(store.path) as db:
                for r in raw_recs:
                    body = {k: v for k, v in r.items() if k not in ("sha256", "prior_episode")}
                    self.assertEqual(hashlib.sha256(canonical(body).encode()).hexdigest(), r["sha256"])   # exact lineage
                    db.execute("INSERT INTO records(id,namespace,world_id,task_id,kind,body,sha256) VALUES(?,?,?,?,?,?,?)",
                               (r["id"], r["namespace"], r["world_id"], r["task_id"], r["kind"], canonical(body), r["sha256"]))
            def rows():
                with sqlite3.connect(store.path) as db:
                    return hashlib.sha256(json.dumps(db.execute("SELECT seq,id,body,sha256 FROM records ORDER BY seq").fetchall()).encode()).hexdigest()
            before = rows()
            task = Task(**{k: (tuple(v) if k == "allowed_targets" else v) for k, v in FIX["input"]["operator_task"].items()})
            self.assertEqual(task.payload()["allowed_targets"], FIX["input"]["operator_task"]["allowed_targets"])
            snap = store.role_snapshot("navigation", world_id=raw_recs[0]["world_id"], task=task,
                                       recipient_agent_id=FIX["input"]["agent_id"], current_episode_id="functional_20261123T024024_reopen")
            view = project_role(snap.payload(), "navigation", "functional_20261123T024024_reopen", 1)
            self.assertEqual(rows(), before)                                   # raw records untouched
            self.assertEqual({r["id"]: r["raw_sha256"] for r in view["source"]["raw_record_refs"]}, {r["id"]: r["sha256"] for r in raw_recs})
            self.assertEqual(view["source"]["raw_snapshot_sha256"], snap.payload()["snapshot_sha256"])
            ph = view.pop("projection_sha256")
            self.assertEqual(ph, hashlib.sha256(canonical(view).encode()).hexdigest()); view["projection_sha256"] = ph
            self.assertNotEqual(ph, view["source"]["raw_snapshot_sha256"])
            rendered = {k: x for k, x in walk({kk: vv for kk, vv in view.items() if kk != "source"})}
            self.assertFalse(FORBIDDEN_KEYS & set(rendered))
            self.assertNotIn("inspect_ready", json.dumps(view))
            self.assertNotIn("0.095", json.dumps({k: v for k, v in view.items() if k != "source"}))
            exp = view["previous_episodes"][0]["experience"]
            self.assertEqual(exp["segments_moved_before_first_arrival"] is not None, True)   # meaningful experience retained
            self.assertEqual(view["earlier_this_episode"], [])


class TemporalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def tools_at(self, at, route_ok=True):
        t = tools(self.dir)
        t["navigation"]["nav.target_state"] = lambda: {"station_1": {"distance_to_estimated_m": 0.1 if at else 4.2138, "at_target": at},
                                                        "station_2": {"distance_to_estimated_m": 1.2, "at_target": False}}
        t["navigation"]["nav.astar_route"] = lambda target: {"ok": True, "length_m": 0.1 if at else 4.214} if route_ok else {"ok": False, "error": "Goal inside obstacle"}
        t["navigation"]["nav.segment_preview"] = lambda target: {"ok": not at, "commanded": [0, 0, 0], "segment_m": 0.65} if not at else {"ok": False, "error": "Route makes no progress"}
        return t

    def test_all_roles_and_supervisor_get_scoped_temporal_views(self):
        store = ObservationStore(self.dir / "m.sqlite3", "fx")
        m1 = ScriptedModel(); rt1 = runtime(self.dir / "a", m1, episode="ep_a", memory=store, tool_set=self.tools_at(False), ledger_root=self.dir / "a") if (self.dir / "a").mkdir() is None else None
        r = rt1.decide(1, {"source": "fx"}); rec = executor_record(r)
        ev = rt1.ledger.emit("executor_result", "executor", (r["event_id"],), **rec); rt1.record_outcome(1, 0, rec, ev)
        (self.dir / "b").mkdir()
        m2 = ScriptedModel(); rt2 = runtime(self.dir / "b", m2, episode="ep_b", memory=store, tool_set=self.tools_at(False), ledger_root=self.dir / "b")
        rt2.decide(1, {"source": "fx"})
        for role in ROLES:
            p = [c for c in m2.calls if c["role"] == role][0]["payload"]
            v = p["memory_temporal_view"]
            self.assertNotIn("role_memory", p)
            self.assertEqual(v["owner_role"], role); self.assertEqual(v["recipient_agent_id"], role + "_agent.ep_b")
            self.assertEqual([e["episode_id"] for e in v["previous_episodes"]], ["ep_a"])
            self.assertFalse(FORBIDDEN_KEYS & {k for k, _ in walk({k: x for k, x in v.items() if k != "source"})})
        sup = [c for c in m2.calls if c["role"] == "supervisor"][0]["payload"]
        self.assertNotIn("persistent_task_memory", sup)
        self.assertEqual(sup["task_memory_temporal_view"]["previous_episodes"][0]["episode_id"], "ep_a")
        reads = {e["event_id"]: e for e in events(rt2, "memory_read")}
        projs = events(rt2, "memory_projection")
        self.assertEqual(len(projs), 5)
        for pe in projs:
            raw = reads[pe["parent_event_ids"][0]]
            self.assertEqual(pe["data"]["raw_snapshot_sha256"], raw["data"]["snapshot_sha256"])
            self.assertEqual([x["id"] for x in pe["data"]["raw_record_refs"]], raw["data"]["record_ids"])
        inputs = [e for e in events(rt2, "model_input")]
        self.assertTrue(all(e["data"]["memory_projection_sha256"] for e in inputs))

    def test_legitimate_at_target_inspection_and_hold_are_permitted(self):
        def nav(n, p, s):
            return {"target": "station_1", "route_status": "at_target", "recommendation": "inspect_ready", "note": "measured within tolerance"}
        def act(n, p, s):
            ids = s["anyOf"][0]["properties"]["evidence_handoffs"]["items"]["enum"]
            return {"kind": "inspect", "target": "station_1", "reason": "at target", "evidence_handoffs": ids[:2]}
        rt = runtime(self.dir, ScriptedModel({"navigation": nav, "action": act}), tool_set=self.tools_at(True))
        res = rt.decide(1, {"source": "fx"})
        self.assertEqual(res["status"], "proposal_admitted"); self.assertEqual(res["proposal"]["kind"], "inspect")
        self.assertEqual(events(rt, "check_failed"), [])
        for out in ({"target": "station_1", "route_status": "not_needed", "recommendation": "hold", "note": "uncertain"},
                    {"target": "station_1", "route_status": "at_target", "recommendation": "hold", "note": "wait"}):
            semantic_check(out, {"targets": {"station_1": {"at_target": True}}, "routes": {"station_1": {"ok": True}}})

    def test_no_route_cannot_be_advanced_but_can_be_held(self):
        ev = {"targets": {"station_1": {"at_target": False, "distance_to_estimated_m": 4.2}}, "routes": {"station_1": {"ok": False}}}
        for bad in ({"target": "station_1", "route_status": "route_available", "recommendation": "advance", "note": "x"},
                    {"target": "station_1", "route_status": "no_route", "recommendation": "advance", "note": "x"}):
            with self.assertRaises(SemanticCheckFailure):
                semantic_check(bad, ev)
        semantic_check({"target": "station_1", "route_status": "no_route", "recommendation": "hold", "note": "x"}, ev)
        with self.assertRaises(SemanticCheckFailure):
            semantic_check({"target": "station_1", "route_status": "no_route", "recommendation": "hold", "note": "x"},
                           {"targets": ev["targets"], "routes": {"station_1": {"ok": True, "next_segment": {"ok": True}}}})
        with self.assertRaises(SemanticCheckFailure):
            semantic_check({"target": "station_1", "route_status": "route_available", "recommendation": "advance", "note": "x"},
                           {"targets": {"station_1": {"at_target": True}}, "routes": {"station_1": {"ok": True}}})

    def test_bounded_replan_succeeds_without_forced_action(self):
        navs = [dict(FIX["output"]), {"target": "station_1", "route_status": "route_available", "recommendation": "advance", "note": "4.21 m away"}]
        offered = []
        def sup(n, p, s):
            allowed = s["properties"]["next"]["enum"]; offered.append((allowed, p["failed_checks"]))
            if p["failed_checks"] and "navigation" in allowed:
                return {"next": "navigation", "request": "Re-assess from the measured tools.", "reason": "failed check"}
            return {"next": allowed[0], "request": "r", "reason": "r"}
        model = ScriptedModel({"navigation": navs, "supervisor": sup}); rt = runtime(self.dir, model, tool_set=self.tools_at(False))
        res = rt.decide(1, {"source": "fx"})
        self.assertEqual(res["status"], "proposal_admitted")
        self.assertEqual(res["proposal"], {"kind": "navigate", "target": "station_1"})   # model-chosen, not substituted
        self.assertEqual(res["route"], ["camera", "navigation", "navigation", "action", "safety", "finish"])
        fail = events(rt, "check_failed"); self.assertEqual(len(fail), 1)
        self.assertEqual(fail[0]["data"]["claim"]["recommendation"], "inspect_ready")
        self.assertFalse(fail[0]["data"]["measured_fact"]["measured_at_target"])
        self.assertIn(["navigation", "finish"], [a for a, _ in offered])
        navh = [e for e in events(rt, "handoff_created") if e["data"]["from_role"] == "navigation"]
        self.assertEqual(len(navh), 1); self.assertEqual(navh[0]["data"]["content"]["model_output"]["recommendation"], "advance")
        self.assertEqual([h["output"]["recommendation"] for h in rt.agents["navigation"].history], ["advance"])
        second = [c for c in model.calls if c["role"] == "navigation"][1]["payload"]
        self.assertEqual(second["previous_attempt_rejected"][0]["reason"], "inspect_ready_but_measured_not_at_target")
        self.assertEqual(len(events(rt, "recovery_succeeded")), 1)

    def test_exhausted_recovery_fails_closed_with_reason(self):
        def sup(n, p, s):
            allowed = s["properties"]["next"]["enum"]
            return {"next": "navigation" if "navigation" in allowed else allowed[0], "request": "r", "reason": "r"}
        model = ScriptedModel({"navigation": lambda n, p, s: dict(FIX["output"]), "supervisor": sup})
        rt = runtime(self.dir, model, tool_set=self.tools_at(False))
        res = rt.decide(1, {"source": "fx"})
        self.assertEqual(res["status"], "semantic_check_failed"); self.assertIsNone(res["proposal"]); self.assertFalse(res["gate"]["allowed"])
        self.assertIn("inspect_ready_but_measured_not_at_target", res["error"])
        self.assertEqual(len(events(rt, "recovery_exhausted")), 1)
        self.assertNotIn("action", [c["role"] for c in model.calls])
        self.assertEqual(rt.agents["navigation"].history, [])

    def test_supervisor_may_finish_without_action_after_failure(self):
        def sup(n, p, s):
            allowed = s["properties"]["next"]["enum"]
            return {"next": "finish" if p["failed_checks"] else allowed[0], "request": "r", "reason": "r"}
        model = ScriptedModel({"navigation": [dict(FIX["output"])], "supervisor": sup})
        rt = runtime(self.dir, model, tool_set=self.tools_at(False)); res = rt.decide(1, {"source": "fx"})
        self.assertEqual(res["status"], "semantic_check_failed"); self.assertIsNone(res["proposal"])

    def test_stale_handoff_and_dependent_proposal_are_invalidated(self):
        navs = [None, dict(FIX["output"]), None]
        plan = iter(["camera", "navigation", "navigation"])
        def sup(n, p, s):
            allowed = s["properties"]["next"]["enum"]
            c = next(plan, None)
            return {"next": c if c in allowed else ("finish" if "finish" in allowed else allowed[0]), "request": "r", "reason": "r"}
        safety = [{"verdict": "reject", "concerns": ["other"], "reason": "re-check route"}, None]
        model = ScriptedModel({"navigation": navs, "supervisor": sup, "safety": safety})
        rt = runtime(self.dir, model, tool_set=self.tools_at(False))
        res = rt.decide(1, {"source": "fx"})
        inv = events(rt, "handoff_invalidated")
        self.assertEqual([e["data"]["handoff_id"] for e in inv], ["h01_02_navigation"])
        wd = events(rt, "proposal_withdrawn")
        self.assertEqual(len(wd), 1); self.assertEqual(wd[0]["data"]["reason"], "depends_on_invalidated_handoff")
        # navigation used its two invocations, so the semantic failure cannot be retried: fail closed, nothing executes
        self.assertEqual(res["status"], "semantic_check_failed"); self.assertIsNone(res["proposal"]); self.assertFalse(res["gate"]["allowed"])
        self.assertEqual(len(events(rt, "recovery_exhausted")), 1)

    def test_episode_semantic_failure_cap(self):
        def sup(n, p, s):
            allowed = s["properties"]["next"]["enum"]
            return {"next": "finish" if p["failed_checks"] else allowed[0], "request": "r", "reason": "r"}
        model = ScriptedModel({"navigation": lambda n, p, s: dict(FIX["output"]), "supervisor": sup})
        rt = runtime(self.dir, model, tool_set=self.tools_at(False), budgets=FunctionalBudgets(semantic_failures_per_episode=2))
        a = rt.decide(1, {"source": "fx"}); b = rt.decide(2, {"source": "fx"})
        self.assertEqual(a["status"], "semantic_check_failed"); self.assertEqual(b["status"], "semantic_check_failed")
        self.assertEqual(b["semantic_failures_episode"], 2); self.assertEqual(len(events(rt, "recovery_exhausted")), 1)


if __name__ == "__main__":
    unittest.main()
