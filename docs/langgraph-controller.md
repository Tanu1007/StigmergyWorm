# Dynamic local controller foundation

This additive package is the first implementation milestone of the single-robot
LangGraph design. It does **not** replace the existing mission runner.

## Implemented

- A supervisor chooses registered `observe`, `plan`, and `check` workers at runtime.
  LangGraph `Send` creates independent invocations; there is no mandatory role chain.
- Explicit worker input schemas. Workers do not receive sibling results or the
  complete mutable graph state. The supervisor receives labeled, unverified reports.
- Local Ollama `/api/chat` with an exact installed model tag/digest, JSON schemas,
  finite timeouts and bounded output. Only loopback or explicit RFC1918 endpoints
  are accepted; environment proxies and redirects are not used.
- Immutable operator task and target catalogue. Unknown roles, actions and fields
  fail closed. Workers cannot generate executable code, grant permissions, invoke
  shell commands or alter protected policies.
- Finite round/worker budgets and sequential inference on the shared local model.
- Separate semantic target verification on/off. The target catalogue and action
  vocabulary remain fixed in both conditions.
- Local JSONL input/output and worker-lifecycle evidence, parent event IDs, image
  hashes and a result manifest. Hosted LangSmith tracing is disabled for execution.
- Fresh per-run controller instances; no checkpointer or long-term memory store.

## What is NOT implemented

This version is **shadow-only**: all admitted actions remain proposals. It does
not publish Gazebo topics, move the robot, start simulation, print or relocate
carriers, create self-modifying agents, or implement infection/reinfection loops.
It does not implement durable agent memory, per-agent infection scores, a physical
point-cloud producer, or the Evidence Lab's future live worker-DAG importer.

`controller_status=completed` means the graph stopped normally. It does not mean
the task or a robot action succeeded. `robot_task_success` remains null and
`execution` is always `not_executed_shadow`. Budget exhaustion is explicit.
Model errors and malformed/truncated outputs stop the run, with a failure record;
there are no hidden retries that inflate the attempt count.

## Install in the separate checkout (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m unittest discover -s tests_langgraph -v
```

The optional package adds no dependency to the legacy `requirements.txt`.
`uv.lock` records the resolved environment; `uv sync --frozen` is an alternative
when uv is installed. Keep the baseline's existing environment untouched.

## A benign local-model smoke run

Ollama must already be running with `qwen3-vl:8b-instruct` installed. The program
checks the exact tag and records its digest; it never downloads or starts models.

```powershell
.\.venv\Scripts\python.exe -m robot_controller --task "Inspect Station 1 and report the available neutral evidence." --observation "A neutral Station 1 marker is reported. No action has been performed."
```

An optional `--image` accepts a real PNG/JPEG evidence file. Image bytes are sent
only to the explicitly configured local endpoint and are not included in traces.
An image hash in a trace establishes input identity, not perception or influence.

Use `--semantic-verifier off` only to compare the task-target check. Both modes
are shadow-only and enforce the same registered capabilities. A model-generated
permission claim cannot change these settings. The local backend defaults to
`http://127.0.0.1:11434`; explicitly configure `--ollama-url` for WSL after checking
the current host address. No firewall or persistent Ollama configuration is changed.

## Evidence and isolation

Each invocation creates `runs/controller/controller_<uuid>/events.jsonl` and
`result.json`. This directory is ignored by git. Inputs and outputs may contain
private task or camera-derived information; review them before sharing.

Worker IDs are unique within a run and namespaced by `run_id`. Every event has
parent IDs referring to earlier events. This is a **data-flow lineage**, not proof
of causal compromise. Quoting untrusted text is not itself a violation. A logged
model rationale is not a hidden-state probe.

Role isolation here is application-level input selection, not an operating-system
sandbox. All code is registered ahead of time; runtime dispatch chooses role
instances, not arbitrary programs. No tool-bearing model interface is exposed.

## Next integration milestones

1. Validate a finite benign shadow pilot on the actual local model and save its
   call count/latency; unit tests use labeled deterministic doubles.
2. Add and independently test a Gazebo execution adapter on its own repository
   branch. Preserve the existing single-agent baseline and distinguish proposed
   actions, dispatches, movement and arrival. Shadow results are not a substitute.
3. Connect real worker events to Evidence Lab without treating lineage as infection.
4. Preregister bounded defensive/utility and reset-control conditions; estimate
   batch time from the pilot. Physical cross-robot propagation and self-sustaining
   behavior are not implemented or established by this package.

## Execution mode (added 2026-11-22)

`Controller(..., mode="gazebo_execution", execution_context={...})` is used by the separate
Gazebo integration branch (`feature/langgraph-gazebo-integration` in gazebo_paper_lab,
`langgraph_mission.py`). The controller still never actuates. Only the supervisor prompt and
labels change so they truthfully state that an external bounded runner may dispatch an admitted
proposal after the graph exits (`execution: pending_runner_executor`; a failed controller is
`not_dispatched_controller_failed`). `execution_context` is runner-measured simulator state
and executor history (bounded to 8000 JSON characters) given to the supervisor and workers; it is
the only record of earlier decisions because every decision builds a fresh graph with no store.
`Ledger(root, run_id="<episode>_dNN")` namespaces controller run IDs per decision. The default
shadow mode, its prompt and the CLI are unchanged.
