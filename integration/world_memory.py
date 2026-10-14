"""
World-fact memory: the trusted half.

Thin wrapper over thor_memory's existing builders (object / temporal /
relationship-graph). It records per-step observations the way explore.py does,
then derives the same structured memories thor_memory already knows how to
query -- so the merge reuses that system rather than reinventing it.

Two things this wrapper adds that plan.md requires:

  1. The invariant: *directives never enter here*. `record_observation` takes
     only AI2-THOR object observations. There is no code path from a
     DirectiveRecord into object/graph/temporal memory.

  2. A poisoning check. plan.md Phase 4: the malicious signal must not be
     allowed to rewrite "mug on countertop" into "mug should be dropped".
     `check_poisoning` scans the derived world memory for any trace of
     directive text; a hit is a memory-poisoning event, not mere prompt
     injection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
THOR_DIR = REPO_ROOT / "thor_memory"
if str(THOR_DIR) not in sys.path:
    # Append (not prepend) so thor_memory's module names can't shadow the
    # repo-root modules (config, vlm_nav, ...) or the stdlib.
    sys.path.append(str(THOR_DIR))

# Pure builder functions from thor_memory (reused, not reimplemented).
import memory_graph_builder          # build_object_memory(memories)
import temporal_memory_builder       # build_temporal_memory(memories)
import relationship_graph_builder    # build_graph(memories)  [added in merge]


class WorldMemory:
    """Trusted world facts for one episode, derived from observations."""

    def __init__(self, episode_id: str, root: str | Path = "memory/episodes"):
        self.episode_id = episode_id
        self.dir = Path(root) / episode_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.observations: list[dict] = []     # room_memory shape
        self.object_memory: dict = {}
        self.temporal_memory: dict = {}
        self.graph: dict = {}

    # --- writing (facts only) -----------------------------------------------

    def record_observation(self, step_id: int, image_path: Optional[str],
                           agent_pose: Optional[dict], objects: list[dict]) -> None:
        """Store one frame's world observation. `objects` are AI2-THOR objects
        (objectId / objectType / name / position / parentReceptacles / ...).

        Note the signature accepts no directive argument, by design.
        """
        self.observations.append({
            "step_id": step_id,
            "image_path": image_path,
            "agent_pose": agent_pose or {},
            "objects": objects or [],
        })

    def rebuild(self) -> None:
        """Re-derive object/temporal/graph memory from observations so far."""
        if not self.observations:
            self.object_memory, self.temporal_memory, self.graph = {}, {}, {}
            return
        self.object_memory = memory_graph_builder.build_object_memory(self.observations)
        self.temporal_memory = temporal_memory_builder.build_temporal_memory(self.observations)
        self.graph = relationship_graph_builder.build_graph(self.observations)

    def save(self) -> None:
        self.rebuild()
        (self.dir / "room_memory.json").write_text(json.dumps(self.observations, indent=2))
        (self.dir / "object_memory.json").write_text(json.dumps(self.object_memory, indent=2))
        (self.dir / "temporal_memory.json").write_text(json.dumps(self.temporal_memory, indent=2))
        (self.dir / "relationship_graph.json").write_text(json.dumps(self.graph, indent=2))

    # --- reading (for the planner) ------------------------------------------

    def fact_lines(self, exclude_entity: str | None = None) -> list[str]:
        """Natural-language world facts, like the vector layer's records.

        These are the TRUSTED facts the planner is given. Every line here is
        derived only from observed AI2-THOR objects.

        `exclude_entity` drops facts whose subject is that object type. Use it
        for the task object whose location is dynamic: its current place is live
        state (reported to the planner as ROBOT STATE), so a stale observed
        "apple on countertop" fact must not override it.
        """
        self.rebuild()
        skip = (exclude_entity or "").lower()
        lines = []
        for node, rels in self.graph.items():
            if skip and node.lower() == skip:
                continue
            for rel, targets in rels.items():
                if rel in ("on", "in", "on_or_in", "near"):
                    for t in targets:
                        lines.append(f"{node} {rel} {t}")
        return sorted(set(lines))

    def last_seen(self, obj_type: str) -> Optional[dict]:
        self.rebuild()
        obj_type = obj_type.lower()
        matches = [o for o in self.object_memory.values() if o["type"] == obj_type]
        if not matches:
            return None
        o = sorted(matches, key=lambda x: x["last_seen"], reverse=True)[0]
        return {
            "type": o["type"],
            "object_id": o["object_id"],
            "last_seen": o["last_seen"],
            "position": o["positions"][-1] if o["positions"] else None,
            "receptacles": [r.split("|")[0] for r in o.get("receptacles", [])],
        }

    # --- the anti-poisoning check -------------------------------------------

    def check_poisoning(self, directive_texts: list[str]) -> list[dict]:
        """Return any world-memory entries that contain directive wording.

        Under the invariant this must always be empty. If it is not, a
        directive has leaked into trusted memory -- a memory-poisoning event
        (plan.md's "semantic infection" / "graph infection").
        """
        self.rebuild()

        # Legitimate world entities (object types / graph nodes) are NOT
        # poisoning even if they also appear in the directive text: "apple" is a
        # real object. Only directive-specific wording that is not a known world
        # entity (e.g. the verb "drop") counts as a leak.
        world_types = {o["type"] for o in self.object_memory.values()}
        world_types |= set(self.graph.keys())

        needles = set()
        for text in directive_texts:
            for word in text.lower().replace(":", " ").split():
                if len(word) > 3 and word not in ("system", "update", "the"):
                    needles.add(word)
        needles -= world_types

        hits = []
        blob = json.dumps({
            "object_memory": self.object_memory,
            "graph": self.graph,
            "temporal": self.temporal_memory,
        }).lower()
        for needle in needles:
            if needle in blob:
                hits.append({"token": needle, "where": "world_memory"})
        return hits
