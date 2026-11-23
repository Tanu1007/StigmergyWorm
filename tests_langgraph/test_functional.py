import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from functional_fixtures import ScriptedModel, runtime, tools, executor_record, TASK
from robot_controller.functional import (FunctionalBudgets, build_agents, CameraAgent, ROLES, AgentFailure)
from robot_controller.memory import ObservationStore
from robot_controller.ollama import OllamaModel, Frame


def events(rt, kind=None):
    rows = [json.loads(x) for x in rt.ledger.path.read_text().splitlines()]
    return [e for e in rows if kind is None or e["event_type"] == kind]


class FunctionalAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_four_roles_call_injected_model_and_frames_reach_relevant_recipients(self):
        model = ScriptedModel(); rt = runtime(self.dir, model)
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "proposal_admitted")
        self.assertEqual(res["proposal"], {"kind": "navigate", "target": "station_1"})
        roles = [c["role"] for c in model.calls]
        for role in ROLES:
            self.assertIn(role, roles)
        frames = {c["role"]: c["frame"] for c in model.calls if c["role"] in ROLES}
        self.assertIsNotNone(frames["camera"])
        self.assertEqual(frames["camera"], frames["action"]); self.assertEqual(frames["camera"], frames["safety"])
        self.assertIsNone(frames["navigation"])
        self.assertTrue(all(c["frame"] is None for c in model.calls if c["role"] == "supervisor"))
        nav = [c for c in model.calls if c["role"] == "navigation"][0]["payload"]
        self.assertIn("station_1", nav["measured_navigation_tools"]["routes"])
        self.assertEqual([h["from_role"] for h in nav["received_handoffs"]], ["camera"])
        safety = [c for c in model.calls if c["role"] == "safety"][0]["payload"]
        self.assertEqual({h["from_role"] for h in safety["received_handoffs"]}, {"camera", "navigation", "action"})
        outs = events(rt, "model_output")
        self.assertTrue(outs and all(e["data"]["wire_matches_payload"] for e in outs))
        inputs = [e for e in events(rt, "model_input") if e["data"]["role"] in ("action", "safety")]
        self.assertTrue(all(e["data"]["frame"]["sha256"] == frames["camera"] for e in inputs))

    def test_stable_ids_private_history_and_distinct_stages(self):
        model = ScriptedModel(); rt = runtime(self.dir, model)
        a = rt.decide(1, {"source": "fixture"}); b = rt.decide(2, {"source": "fixture"})
        self.assertEqual(a["agents"], b["agents"])
        self.assertEqual(len(set(a["agents"].values())), 4)
        second_cam = [c for c in model.calls if c["role"] == "camera"][1]["payload"]
        self.assertEqual(len(second_cam["private_history"]), 1)
        self.assertIn("observation", second_cam["private_history"][0]["output"])
        self.assertNotIn("private_history", json.dumps(second_cam["received_handoffs"]))
        second_nav = [c for c in model.calls if c["role"] == "navigation"][1]["payload"]
        self.assertTrue(all("route_status" in h["output"] for h in second_nav["private_history"]))
        stages = events(rt, "stage")
        for role in ROLES:
            seen = {e["data"]["stage"] for e in stages if e["data"]["role"] == role}
            self.assertEqual(seen, {"observe", "plan", "check"})
        agents = {e["agent_id"] for e in stages}
        self.assertEqual(agents, set(a["agents"].values()))
        self.assertFalse(any(e["data"].get("role") in ("observe", "plan", "check") for e in events(rt)))

    def test_role_memory_is_owned_and_scoped(self):
        store = ObservationStore(self.dir / "m.sqlite3", "fixture")
        model = ScriptedModel(); rt = runtime(self.dir, model, memory=store)
        res = rt.decide(1, {"source": "fixture"})
        rec = executor_record(res); ev = rt.ledger.emit("executor_result", "executor", (res["event_id"],), **rec)
        receipts = rt.record_outcome(1, 0, rec, ev)
        owned = {r["owner_role"]: r["id"] for r in receipts}
        self.assertEqual(set(owned), set(ROLES) | {"supervisor_task_facts"})
        rt.decide(2, {"source": "fixture"})
        for role in ROLES:
            p = [c for c in model.calls if c["role"] == role][1]["payload"]
            mem = p["role_memory"]
            self.assertEqual(mem["owner_role"], role)
            self.assertEqual(mem["record_ids"], [owned[role]])
            self.assertTrue(all(r["payload"]["owner_role"] == role for r in mem["records"]))
            others = {owned[r] for r in ROLES if r != role}
            self.assertFalse(others & set(json.dumps(p).split('"')))
        reads = [e for e in events(rt, "memory_read") if e["data"].get("owner_role") in ROLES]
        self.assertTrue(all(e["data"]["recipient_agent_id"] == e["agent_id"] for e in reads))

    def test_invalid_output_fails_closed_without_guessing(self):
        model = ScriptedModel({"camera": [{"observation": "x"}]}); rt = runtime(self.dir, model)
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "agent_failed")
        self.assertIsNone(res["proposal"]); self.assertFalse(res["gate"]["allowed"])
        self.assertNotIn("action", [c["role"] for c in model.calls])
        self.assertEqual(len(events(rt, "agent_failed")), 1)

    def test_action_must_cite_received_handoffs(self):
        bad = {"kind": "navigate", "target": "station_1", "reason": "r", "evidence_handoffs": ["h99_99_invented"]}
        model = ScriptedModel({"action": [bad]}); rt = runtime(self.dir, model)
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "agent_failed"); self.assertIsNone(res["proposal"])
        self.assertNotIn("safety", [c["role"] for c in model.calls])

    def test_supervisor_cannot_skip_to_action_or_finish(self):
        model = ScriptedModel({"supervisor": [{"next": "action", "request": "act", "reason": "skip"}]}); rt = runtime(self.dir, model)
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "agent_failed")
        first = model.calls[0]["schema"]["properties"]["next"]["enum"]
        self.assertEqual(first, ["camera", "navigation"])
        self.assertEqual(len(model.calls), 1)

    def test_safety_rejection_routes_back_dynamically(self):
        plan = iter(["camera", "camera", "action"])
        offered = []
        def sup(n, payload, schema):
            allowed = schema["properties"]["next"]["enum"]; offered.append(allowed)
            choice = next(plan); self.assertIn(choice, allowed)
            return {"next": choice, "request": "routing note", "reason": "feedback-driven"}
        safety = [{"verdict": "reject", "concerns": ["camera_disagreement"], "reason": "Re-check the view."}, None]
        model = ScriptedModel({"supervisor": sup, "safety": safety}); rt = runtime(self.dir, model)
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["route"], ["camera", "navigation", "action", "safety", "camera", "action", "safety", "finish"])
        self.assertEqual(res["route_sources"].count("supervisor_model"), 3)
        self.assertEqual(offered[0], ["camera", "navigation"])
        self.assertEqual(offered[1], ["camera", "navigation", "action", "finish"])   # feedback opens re-planning
        self.assertEqual(offered[2], ["navigation", "action", "finish"])
        self.assertEqual(res["status"], "proposal_admitted")
        act2 = [c for c in model.calls if c["role"] == "action"][1]["payload"]
        self.assertIn("safety", [h["from_role"] for h in act2["received_handoffs"]])
        cams = [c["frame"] for c in model.calls if c["role"] == "camera"]
        self.assertNotEqual(cams[0], cams[1])
        self.assertEqual([c for c in model.calls if c["role"] == "safety"][1]["frame"], cams[1])

    def test_approved_proposal_cannot_be_replanned_and_single_options_skip_model(self):
        model = ScriptedModel(); rt = runtime(self.dir, model)
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["route"], ["camera", "navigation", "action", "safety", "finish"])
        self.assertEqual(res["route_sources"], ["supervisor_model"] + ["only_allowed_option"] * 4)
        self.assertEqual([c["role"] for c in model.calls], ["supervisor", "camera", "navigation", "action", "safety"])

    def test_rejected_proposal_executes_nothing(self):
        def sup(n, payload, schema):
            allowed = schema["properties"]["next"]["enum"]
            return {"next": "finish" if "finish" in allowed else [a for a in ("camera", "navigation", "action", "safety") if a in allowed and a not in payload["invocations_this_decision"]][0],
                    "request": "r", "reason": "r"}
        model = ScriptedModel({"supervisor": sup, "safety": [{"verdict": "reject", "concerns": ["other"], "reason": "no"}]})
        rt = runtime(self.dir, model); res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "proposal_rejected_by_safety"); self.assertFalse(res["gate"]["allowed"])

    def test_deterministic_gate_overrides_safety_approval(self):
        off = {"kind": "navigate", "target": "station_2", "reason": "elsewhere", "evidence_handoffs": None}
        def act(n, payload, schema):
            ids = schema["anyOf"][0]["properties"]["evidence_handoffs"]["items"]["enum"]
            return dict(off, evidence_handoffs=ids[:1])
        model = ScriptedModel({"action": act}); rt = runtime(self.dir, model)
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "proposal_rejected_by_safety")
        self.assertEqual(res["safety"]["override"], "deterministic_check_rejected")
        self.assertFalse(res["gate"]["allowed"])

    def test_tools_are_role_owned(self):
        t = tools(self.dir)
        t["camera"]["nav.pose"] = lambda: {}
        with self.assertRaises(ValueError):
            build_agents("ep", t)
        agents = build_agents("ep", tools(self.dir))
        with self.assertRaises(PermissionError):
            agents["action"].tool("nav.pose")

    def test_tool_denial_is_logged_and_fails_closed(self):
        model = ScriptedModel(); rt = runtime(self.dir, model)
        original = CameraAgent.observe
        def rogue(agent, rt_, d):
            return rt_.use_tool(agent, "nav.astar_route", target="station_1")
        with patch.object(CameraAgent, "observe", rogue):
            res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "agent_failed")
        self.assertEqual(len(events(rt, "tool_denied")), 1)
        self.assertEqual(len([c for c in model.calls if c["role"] == "camera"]), 0)

    def test_episode_model_budget_fails_closed(self):
        model = ScriptedModel(); rt = runtime(self.dir, model, budgets=FunctionalBudgets(model_calls_per_episode=3))
        res = rt.decide(1, {"source": "fixture"})
        self.assertEqual(res["status"], "agent_failed"); self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(events(rt, "budget_stop")), 1)

    def test_handoff_lineage_events(self):
        model = ScriptedModel(); rt = runtime(self.dir, model)
        rt.decide(1, {"source": "fixture"})
        created = {e["data"]["handoff_id"]: e for e in events(rt, "handoff_created")}
        delivered = events(rt, "handoff_delivered")
        self.assertTrue(delivered)
        for e in delivered:
            self.assertIn(e["data"]["handoff_id"], created)
            self.assertEqual(e["parent_event_ids"], [created[e["data"]["handoff_id"]]["event_id"]])
            self.assertIn(e["agent_id"].split("_agent.")[0], created[e["data"]["handoff_id"]]["data"]["to_roles"])

    def test_ollama_adapter_wire_hash_is_exact_content(self):
        m = OllamaModel("qwen3-vl:8b-instruct", "http://127.0.0.1:11434"); m.model_digest = "d"
        sent = {}
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "{\"ok\": true}"}, "done_reason": "stop"}
        def post(url, json=None, **kw):
            sent.update(json); return R()
        p = self.dir / "f.png"; p.write_bytes(b"\x89PNG\r\n\x1a\nabc")
        with patch.object(m.session, "post", post):
            m.complete("camera", "sys", {"a": "é"}, {"type": "object"}, Frame(p))
        content = sent["messages"][1]["content"]
        self.assertEqual(m.last_wire_sha256, hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(m.last_image_sha256, Frame(p).sha256)


class DistinctProcessReopenTest(unittest.TestCase):
    def test_episode_one_role_records_reach_episode_two_model_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); db = tmp / "shared.sqlite3"; script = Path(__file__).with_name("reopen_process.py")
            runs = []
            for ep in ("ep_write", "ep_reopen"):
                (tmp / ep).mkdir()
                p = subprocess.run([sys.executable, str(script), str(db), ep, str(tmp / ep), str(tmp / ep / "calls.jsonl")],
                                   capture_output=True, text=True, timeout=120)
                self.assertEqual(p.returncode, 0, p.stderr)
                runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
            self.assertEqual(runs[0]["store_id"], runs[1]["store_id"])
            written = {r["owner_role"]: r["id"] for r in runs[0]["receipts"] if r.get("owner_role") in ROLES}
            self.assertEqual(set(written), set(ROLES))
            calls = [json.loads(x) for x in (tmp / "ep_reopen" / "calls.jsonl").read_text().splitlines()]
            ev2 = [json.loads(x) for x in Path(runs[1]["ledger"]).read_text().splitlines()]
            for role in ROLES:
                first = [c for c in calls if c["role"] == role][0]["payload"]
                mem = first["role_memory"]
                self.assertIn(written[role], mem["record_ids"])
                rec = [r for r in mem["records"] if r["id"] == written[role]][0]
                self.assertEqual(rec["episode_id"], "ep_write"); self.assertTrue(rec["prior_episode"])
                self.assertEqual(first["agent_id"], role + "_agent.ep_reopen")
                read = [e for e in ev2 if e["event_type"] == "memory_read" and e["data"].get("owner_role") == role][0]
                self.assertEqual(read["data"]["recipient_agent_id"], role + "_agent.ep_reopen")
                self.assertEqual(read["data"]["store_id"], runs[0]["store_id"])
                self.assertEqual(read["data"]["records"][0]["episode_id"], "ep_write")
                inp = [e for e in ev2 if e["event_type"] == "model_input" and e["data"]["role"] == role][0]
                self.assertIn(written[role], inp["data"]["memory_record_ids_read"])
                self.assertEqual(inp["data"]["payload_sha256"], hashlib.sha256(json.dumps(first, ensure_ascii=False).encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
