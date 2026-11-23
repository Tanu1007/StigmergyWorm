"""Distinct process: one decision with the shared SQLite store. Used by the reopen test."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_fixtures import ScriptedModel, runtime, executor_record, TASK
from robot_controller.memory import ObservationStore

db, episode, tmp, log = sys.argv[1:5]
store = ObservationStore(db, "reopen_fixture")
model = ScriptedModel(log=log)
rt = runtime(tmp, model, episode=episode, memory=store)
res = rt.decide(1, {"source": "software_fixture"})
rec = executor_record(res)
ev = rt.ledger.emit("executor_result", "executor", (res["event_id"],), **rec)
receipts = rt.record_outcome(1, 0, rec, ev)
print(json.dumps({"status": res["status"], "ledger": str(rt.ledger.path), "receipts": receipts, "store_id": store.store_id,
                  "agents": res["agents"]}))
