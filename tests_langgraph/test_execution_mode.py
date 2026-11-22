import json
import tempfile
import unittest
from pathlib import Path
from robot_controller.schemas import Task
from robot_controller.telemetry import Ledger
from robot_controller.orchestrator import Controller, SUPERVISOR, SUPERVISOR_EXECUTION
from robot_controller.ollama import Frame
from test_controller import FakeModel, finish, delegate

CTX = {"decision": 3, "executed_history": [{"step": 2, "dispatched": True, "displacement_m": 0.61}]}


class ExecutionModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.task = Task("Inspect Station 1.", "station_1", ("station_1", "station_2"))

    def make(self, responses, **kw):
        model = FakeModel(responses); ledger = Ledger(self.temp.name, run_id=kw.pop("run_id", None))
        return Controller(self.task, model, ledger, observations=("Measured fixture.",), **kw), model, ledger

    def test_default_mode_is_unchanged_shadow_prompt(self):
        c, m, l = self.make([finish()]); r = c.run()
        self.assertIs(m.calls[0][1], SUPERVISOR)
        self.assertEqual(r["execution"], "not_executed_shadow")
        self.assertNotIn("measured_execution_context", m.calls[0][2])

    def test_execution_prompt_and_labels_do_not_claim_shadow(self):
        c, m, l = self.make([finish()], mode="gazebo_execution", execution_context=CTX); r = c.run()
        self.assertIs(m.calls[0][1], SUPERVISOR_EXECUTION)
        self.assertNotIn("shadow", SUPERVISOR_EXECUTION.lower())
        self.assertEqual(r["execution"], "pending_runner_executor")
        self.assertEqual(r["gate"]["execution"], "pending_runner_executor")
        started = [e for e in l.events if e["event_type"] == "run_started"][0]
        self.assertEqual(started["data"]["mode"], "gazebo_execution")

    def test_measured_context_reaches_supervisor_and_workers(self):
        c, m, l = self.make([delegate("plan", "observe"), finish()], mode="gazebo_execution", execution_context=CTX); c.run()
        for role, system, payload, frame in m.calls:
            self.assertEqual(payload["measured_execution_context"], CTX)

    def test_shadow_rejects_executor_context(self):
        with self.assertRaises(ValueError):
            self.make([finish()], execution_context=CTX)

    def test_unknown_mode_and_oversized_context_rejected(self):
        with self.assertRaises(ValueError):
            self.make([finish()], mode="hardware")
        with self.assertRaises(ValueError):
            self.make([finish()], mode="gazebo_execution", execution_context={"x": "y" * 9000})

    def test_failed_execution_controller_is_not_pending(self):
        c, m, l = self.make([delegate("shell")], mode="gazebo_execution", execution_context=CTX)
        with self.assertRaises(ValueError):
            c.run()
        self.assertEqual(json.loads((l.directory / "result.json").read_text())["execution"], "not_dispatched_controller_failed")

    def test_namespaced_run_ids_are_unique_and_validated(self):
        a = Ledger(self.temp.name, run_id="ep01_d03"); b = Ledger(self.temp.name, run_id="ep01_d03")
        self.assertTrue(a.run_id.startswith("controller_ep01_d03_"))
        self.assertNotEqual(a.run_id, b.run_id)
        for bad in ["../x", "a b", "", "x" * 101]:
            with self.assertRaises(ValueError):
                Ledger(self.temp.name, run_id=bad)

    def test_frame_provenance_label(self):
        png = Path(self.temp.name) / "fixture.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)  # header-only fixture; not a camera image
        f = Frame(png, source="gazebo_forward_camera", description="step 2 forward frame")
        self.assertEqual(f.metadata()["source"], "gazebo_forward_camera")
        self.assertEqual(f.metadata()["description"], "step 2 forward frame")


if __name__ == "__main__":
    unittest.main()
