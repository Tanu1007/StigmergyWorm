import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from robot_controller.schemas import Task, Limits, Action, gate
from robot_controller.telemetry import Ledger
from robot_controller.orchestrator import Controller
from robot_controller.ollama import local_url, OllamaModel


def finish(target="station_1", kind="inspect"):
    return {"decision":"finish", "jobs":[], "action":{"kind":kind,"target":target},"reason":"Bounded task proposal."}


def delegate(*roles):
    return {"decision":"delegate","jobs":[{"role":r,"question":"Assess the neutral station evidence."} for r in roles],"action":None,"reason":"Gather task evidence."}


class FakeModel:
    model="deterministic-test-double"
    model_digest="test-fixture-not-a-real-model"
    def __init__(self, decisions):
        self.decisions=list(decisions);self.calls=[]
    def complete(self, role, system, payload, schema, frame=None):
        self.calls.append((role,system,payload,frame))
        if role=="supervisor":
            return self.decisions.pop(0)
        return {"summary":"Neutral fixture: station marker observed. No action executed."}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.task=Task("Inspect Station 1.","station_1",("station_1","station_2"))
    def make(self, responses, **kwargs):
        model=FakeModel(responses); ledger=Ledger(self.temp.name)
        controller=Controller(self.task,model,ledger,observations=("A neutral station marker is visible.",),**kwargs)
        return controller,model,ledger
    def test_dynamic_worker_counts_and_roles(self):
        c,m,l=self.make([delegate("observe","check"),delegate("plan"),finish()])
        r=c.run()
        self.assertEqual(r["workers_created"],3)
        self.assertEqual(r["workers_completed"],3)
        self.assertEqual([x[0] for x in m.calls],["supervisor","observe","check","supervisor","plan","supervisor"])
        self.assertEqual(r["execution"],"not_executed_shadow")
        self.assertIsNone(r["robot_task_success"])
    def test_no_mandatory_worker_chain(self):
        c,m,l=self.make([finish(None,"stop")])
        self.assertEqual(c.run()["workers_created"],0)
        self.assertEqual(len(m.calls),1)
    def test_worker_does_not_receive_sibling_results(self):
        c,m,l=self.make([delegate("observe","check"),finish()]);c.run()
        for role,system,payload,frame in m.calls:
            if role!="supervisor":
                self.assertNotIn("worker_reports",payload)
                self.assertNotIn("results",payload)
                self.assertEqual(payload["operator_task"],self.task.payload())
    def test_observer_and_checker_get_assigned_frame_not_planner(self):
        frame=Mock();frame.metadata.return_value={"sha256":"fixture-frame"}
        c,m,l=self.make([delegate("observe","check","plan"),finish()],frame=frame)
        c.run()
        roles={role:f for role,system,payload,f in m.calls if role!="supervisor"}
        self.assertIs(roles["observe"],frame)
        self.assertIs(roles["check"],frame)
        self.assertIsNone(roles["plan"])
    def test_supervisor_receives_remaining_call_budget(self):
        c,m,l=self.make([delegate("observe"),delegate("check"),finish()]);c.run()
        self.assertEqual([p["remaining_supervisor_calls"] for r,s,p,f in m.calls if r=="supervisor"],[3,2,1])
    def test_max_rounds_stops_before_additional_model_call(self):
        c,m,l=self.make([delegate("observe")],limits=Limits(max_rounds=1))
        r=c.run();self.assertEqual(r["stop_reason"],"no_followup_budget")
        self.assertEqual(len(m.calls),1)
        self.assertEqual(r["workers_created"],0)
    def test_total_worker_budget(self):
        c,m,l=self.make([delegate("observe","check")],limits=Limits(max_total_workers=1))
        r=c.run();self.assertEqual(r["stop_reason"],"max_total_workers")
        self.assertEqual(r["workers_created"],0)
    def test_unregistered_worker_fails_closed(self):
        c,m,l=self.make([delegate("shell")])
        with self.assertRaises(ValueError):c.run()
        self.assertFalse(any(e["event_type"]=="worker_spawned" for e in l.events))
        self.assertEqual(json.loads((l.directory/"result.json").read_text())["controller_status"],"failed")
    def test_unknown_supervisor_fields_rejected(self):
        d=finish();d["system_prompt"]="not permitted"
        c,m,l=self.make([d])
        with self.assertRaises(ValueError):c.run()
    def test_arbitrary_action_rejected(self):
        c,m,l=self.make([finish(None,"execute_command")])
        with self.assertRaises(ValueError):c.run()
    def test_verifier_on_blocks_wrong_task_target(self):
        c,m,l=self.make([finish("station_2")],semantic_verification=True)
        r=c.run();self.assertFalse(r["gate"]["allowed"]);self.assertTrue(r["gate"]["task_conflict"])
    def test_verifier_off_does_not_remove_capability_envelope(self):
        c,m,l=self.make([finish("outside_world")],semantic_verification=False)
        r=c.run();self.assertFalse(r["gate"]["allowed"])
        self.assertEqual(r["gate"]["reason"],"outside_target_catalogue")
    def test_verifier_off_is_still_shadow_only(self):
        c,m,l=self.make([finish("station_2")],semantic_verification=False)
        r=c.run();self.assertTrue(r["gate"]["allowed"])
        self.assertTrue(r["gate"]["task_conflict"])
        self.assertEqual(r["execution"],"not_executed_shadow")
    def test_controller_cannot_leak_state_to_second_episode(self):
        c,m,l=self.make([finish()]);c.run()
        with self.assertRaises(RuntimeError):c.run()
    def test_event_parents_and_identifiers(self):
        c,m,l=self.make([delegate("observe","check"),finish()]);c.run()
        lines=[json.loads(x) for x in l.path.read_text().splitlines()]
        prior=set()
        for n,e in enumerate(lines,1):
            self.assertEqual(e["sequence"],n)
            self.assertTrue(set(e["parent_event_ids"])<=prior)
            self.assertNotIn(e["event_id"],prior)
            prior.add(e["event_id"])
        starts=[e["agent_id"] for e in lines if e["event_type"]=="worker_spawned"]
        self.assertEqual(len(starts),len(set(starts)))
    def test_task_is_frozen(self):
        with self.assertRaises(Exception):self.task.instruction="Changed task"
    def test_invalid_limits(self):
        for value in [0,-1,True,100]:
            with self.assertRaises(ValueError):Limits(max_rounds=value)
    def test_worker_output_unknown_field_fails(self):
        c,m,l=self.make([delegate("observe")])
        original=m.complete
        m.complete=lambda role,*args: ({"summary":"Neutral","code":"not allowed"} if role!="supervisor" else original(role,*args))
        with self.assertRaises(ValueError):c.run()
    def test_real_model_error_is_not_success(self):
        c,m,l=self.make([])
        m.complete=Mock(side_effect=TimeoutError("fixture"))
        with self.assertRaises(TimeoutError):c.run()
        self.assertEqual(l.events[-1]["event_type"],"run_failed")


class LocalAdapterTests(unittest.TestCase):
    def test_local_endpoints_only(self):
        for endpoint in ["http://127.0.0.1:11434","http://localhost:11434","http://172.24.112.1:11434"]:
            self.assertEqual(local_url(endpoint),endpoint)
        for endpoint in ["https://example.com","http://user:password@127.0.0.1","http://127.0.0.1/path","http://8.8.8.8","http://127.0.0.1?q=x"]:
            with self.assertRaises(ValueError):local_url(endpoint)
    def test_preflight_requires_exact_installed_model(self):
        m=OllamaModel();r=Mock();r.json.return_value={"models":[]};m.session.get=Mock(return_value=r)
        with self.assertRaises(ValueError):m.preflight()
    def test_preflight_records_digest(self):
        m=OllamaModel();r=Mock();r.json.return_value={"models":[{"name":m.model,"digest":"fixture-digest"}]};m.session.get=Mock(return_value=r)
        self.assertEqual(m.preflight()["digest"],"fixture-digest")
    def test_malformed_or_truncated_model_output(self):
        m=OllamaModel();m.model_digest="fixture";r=Mock(status_code=200)
        r.json.return_value={"message":{"content":"not JSON"}};m.session.post=Mock(return_value=r)
        with self.assertRaises(ValueError):m.complete("observe","system",{}, {})
        r.json.return_value={"done_reason":"length","message":{"content":"{}"}}
        with self.assertRaises(ValueError):m.complete("observe","system",{}, {})
    def test_redirects_not_followed(self):
        m=OllamaModel();m.model_digest="fixture";m.session.post=Mock(return_value=Mock(status_code=302))
        with self.assertRaises(RuntimeError):m.complete("observe","system",{}, {})
        self.assertFalse(m.session.post.call_args.kwargs["allow_redirects"])
    def test_proxy_environment_disabled(self):
        self.assertFalse(OllamaModel().session.trust_env)


if __name__=="__main__":unittest.main()
