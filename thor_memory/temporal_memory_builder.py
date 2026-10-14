import json
from pathlib import Path
from collections import defaultdict

ROOM_MEMORY = "memory/room_memory.json"
OUTPUT_FILE = "memory/temporal_memory.json"


def build_temporal_memory(memories):
    temporal = {}

    for memory in memories:
        step_id = memory["step_id"]
        image_path = memory["image_path"]
        robot_pose = memory["agent_pose"]

        for obj in memory["objects"]:
            oid = obj["objectId"]

            if oid not in temporal:
                temporal[oid] = {
                    "object_id": oid,
                    "type": obj["objectType"].lower(),
                    "name": obj["name"],
                    "first_seen_step": step_id,
                    "last_seen_step": step_id,
                    "seen_steps": [],
                    "trajectory": [],
                    "receptacle_history": [],
                    "visibility_events": []
                }

                temporal[oid]["visibility_events"].append({
                    "event": "first_seen",
                    "step": step_id,
                    "image_path": image_path
                })

            entry = temporal[oid]
            entry["last_seen_step"] = step_id
            entry["seen_steps"].append(step_id)

            entry["trajectory"].append({
                "step": step_id,
                "object_position": obj["position"],
                "robot_pose": robot_pose,
                "image_path": image_path
            })

            parents = obj.get("parentReceptacles")
            if parents:
                for parent in parents:
                    parent_name = parent.split("|")[0].lower()

                    if not entry["receptacle_history"] or entry["receptacle_history"][-1]["receptacle"] != parent_name:
                        entry["receptacle_history"].append({
                            "step": step_id,
                            "receptacle": parent_name,
                            "raw_receptacle_id": parent
                        })

    max_step = max(m["step_id"] for m in memories)

    for entry in temporal.values():
        seen = set(entry["seen_steps"])

        missing_after_seen = [
            step for step in range(entry["first_seen_step"], max_step + 1)
            if step not in seen
        ]

        if missing_after_seen:
            entry["visibility_events"].append({
                "event": "not_visible_after_seen",
                "steps": missing_after_seen
            })

        entry["total_observations"] = len(entry["seen_steps"])
        entry["lifespan_steps"] = entry["last_seen_step"] - entry["first_seen_step"] + 1

    return temporal


def main():
    memories = json.loads(Path(ROOM_MEMORY).read_text())
    temporal = build_temporal_memory(memories)

    Path(OUTPUT_FILE).write_text(json.dumps(temporal, indent=2))

    print(f"Created temporal memory for {len(temporal)} objects")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()