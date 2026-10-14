"""
StigmergyWorm x thor_memory integration.

Physical prompt injection against an embodied agent that has persistent memory.

The two halves this package joins:

  - StigmergyWorm  (repo root: vlm_nav.py, config.py, p_inf_trial_loop.py)
    perception + VLM decision + infection statistics (p_inf, Wilson CI).

  - thor_memory    (repo root: thor_memory/)
    AI2-THOR environment + persistent world memory
    (object / temporal / graph / vector).

The new layer this package adds is the one plan.md says is missing from both:
a *directive* memory type with provenance and trust, kept strictly separate
from world facts, plus the planner / safety-validator / infection-evaluator
spine that turns "the model read a label" into a measured infection.

See README_integration.md in the repo root for the architecture and how to run.
"""

__all__ = [
    "directive_memory",
    "world_memory",
    "carriers",
    "perception",
    "planner",
    "validator",
    "executor",
    "thor_task",
    "infection",
    "trial",
]
