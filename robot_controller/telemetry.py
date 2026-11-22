"""Local append-only tracing. No hosted telemetry, model keys, or image bytes."""
import json
import re
import threading
import time
import uuid
from pathlib import Path


class Ledger:
    def __init__(self, root, run_id=None):
        # Optional caller namespace (for example episode + decision); a random suffix
        # keeps IDs unique. Fresh per-decision graphs are not cross-episode memory.
        if run_id is None:
            self.run_id = "controller_" + uuid.uuid4().hex
        else:
            if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", run_id):
                raise ValueError("run_id namespace must be 1-100 safe characters")
            self.run_id = "controller_" + run_id + "_" + uuid.uuid4().hex[:8]
        self.directory = Path(root) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "events.jsonl"
        self._lock = threading.Lock()
        self._seq = 0
        self.events = []

    def emit(self, kind, agent_id, parents=(), **data):
        with self._lock:
            self._seq += 1
            event = {"schema_version": 1, "run_id": self.run_id,
                     "event_id": f"{self.run_id}:{self._seq}", "sequence": self._seq,
                     "timestamp_ns": time.time_ns(), "event_type": kind,
                     "robot_id": "robot_a", "agent_id": agent_id,
                     "parent_event_ids": list(parents), "data": data}
            serialized = json.dumps(event, ensure_ascii=False, allow_nan=False)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(serialized + "\n")
                f.flush()
            self.events.append(event)
            return event["event_id"]

    def save_result(self, result):
        target = self.directory / "result.json"
        with target.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
