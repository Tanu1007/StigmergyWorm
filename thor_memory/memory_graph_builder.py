import json
from pathlib import Path
from collections import defaultdict


INPUT_FILE = "memory/room_memory.json"
OUTPUT_FILE = "memory/object_memory.json"


def build_object_memory(memories):
    object_memory = {}

    for memory in memories:

        step_id = memory["step_id"]
        image_path = memory["image_path"]

        for obj in memory["objects"]:

            oid = obj["objectId"]

            if oid not in object_memory:

                object_memory[oid] = {
                    "object_id": oid,
                    "type": obj["objectType"].lower(),
                    "name": obj["name"],

                    "first_seen": step_id,
                    "last_seen": step_id,

                    "observed_count": 0,

                    "positions": [],
                    "distances": [],
                    "frames": [],

                    "receptacles": set()
                }

            entry = object_memory[oid]

            entry["last_seen"] = step_id
            entry["observed_count"] += 1

            entry["positions"].append(obj["position"])

            if obj.get("distance") is not None:
                entry["distances"].append(obj["distance"])

            entry["frames"].append({
                "step_id": step_id,
                "image_path": image_path
            })

            parents = obj.get("parentReceptacles")

            if parents:
                for p in parents:
                    entry["receptacles"].add(p)

    for obj in object_memory.values():
        obj["receptacles"] = list(obj["receptacles"])

    return object_memory


def main():

    memories = json.loads(
        Path(INPUT_FILE).read_text()
    )

    object_memory = build_object_memory(memories)

    Path(OUTPUT_FILE).write_text(
        json.dumps(object_memory, indent=2)
    )

    print(
        f"Created memory graph with "
        f"{len(object_memory)} unique objects"
    )

    print(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()