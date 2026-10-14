import json
from pathlib import Path
from collections import defaultdict
from math import sqrt


ROOM_MEMORY = "memory/room_memory.json"
OUTPUT_GRAPH = "memory/relationship_graph.json"

NEAR_THRESHOLD = 0.75

IGNORE_NEAR_TYPES = {
    "floor",
    "wall",
    "window",
    "countertop",
    "cabinet",
    "drawer",
}

STATIC_RECEPTACLES = {
    "countertop",
    "fridge",
    "cabinet",
    "drawer",
    "floor",
}


def distance(a, b):
    return sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def add_edge(graph, source, relation, target):
    if source != target:
        graph[source][relation].add(target)


def parent_relation(parent_name):
    parent_name = parent_name.lower()

    if parent_name in {"countertop", "table", "desk", "shelf", "floor"}:
        return "on"

    if parent_name in {"fridge", "cabinet", "drawer", "garbagecan"}:
        return "in"

    return "on_or_in"


def build_graph(memories):
    """Build the relationship graph from room-memory observations.

    Factored out of main() so other modules (the integration layer) can build a
    graph for an arbitrary list of observations without going through files.
    Returns a plain {node: {relation: sorted[list]}} dict.
    """
    graph = defaultdict(lambda: defaultdict(set))

    for memory in memories:
        objects = memory["objects"]

        # ON / IN relations from parentReceptacles
        for obj in objects:
            obj_type = obj["objectType"].lower()
            parents = obj.get("parentReceptacles")

            if parents:
                for parent in parents:
                    parent_type = parent.split("|")[0].lower()
                    relation = parent_relation(parent_type)

                    add_edge(graph, obj_type, relation, parent_type)
                    add_edge(graph, parent_type, "supports_or_contains", obj_type)

        # NEAR relations only between meaningful movable/semantic objects
        filtered_objects = [
            obj for obj in objects
            if obj["objectType"].lower() not in IGNORE_NEAR_TYPES
        ]

        for i in range(len(filtered_objects)):
            a = filtered_objects[i]
            a_type = a["objectType"].lower()

            for j in range(i + 1, len(filtered_objects)):
                b = filtered_objects[j]
                b_type = b["objectType"].lower()

                d = distance(a["position"], b["position"])

                if d < NEAR_THRESHOLD:
                    add_edge(graph, a_type, "near", b_type)
                    add_edge(graph, b_type, "near", a_type)

        # seen_with relation, also filtered
        visible_types = sorted(set(
            obj["objectType"].lower()
            for obj in filtered_objects
        ))

        for a in visible_types:
            for b in visible_types:
                if a != b:
                    add_edge(graph, a, "seen_with", b)

    serializable_graph = {}

    for node, rels in graph.items():
        serializable_graph[node] = {}

        for rel, targets in rels.items():
            serializable_graph[node][rel] = sorted(list(targets))

    return serializable_graph


def main():
    memories = json.loads(Path(ROOM_MEMORY).read_text())

    serializable_graph = build_graph(memories)

    Path(OUTPUT_GRAPH).write_text(
        json.dumps(serializable_graph, indent=2)
    )

    print(f"Created filtered relationship graph with {len(serializable_graph)} nodes")
    print(f"Saved to {OUTPUT_GRAPH}")


if __name__ == "__main__":
    main()