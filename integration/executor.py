"""
Action executor: turn a validated plan into an environment action.

Resolves the planner's symbolic target ("apple", "fridge") to a concrete object id
in the current scene, calls the environment, and reports the outcome (which is
what gets written back to memory as the action's result).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from integration.planner import Plan
from integration.validator import Verdict, Ruling


@dataclass
class Outcome:
    executed: bool
    action: str
    target: Optional[str]
    success: bool
    ruling: str
    object_state: dict
    note: str = ""


def _resolve(env, target: Optional[str], default_type: Optional[str]) -> Optional[str]:
    """Map a symbolic target ("apple", "the fridge", ...) to an object id."""
    name = (target or "").lower()
    if "apple" in name:
        key = "Apple"
    elif "fridge" in name:
        key = "Fridge"
    elif "counter" in name:
        key = "CounterTop"
    elif "garbage" in name or "trash" in name or "bin" in name:
        key = "GarbageCan"
    elif name:
        key = target
    else:
        key = default_type
    if not key:
        return None
    obj = env.find(key)
    return obj["objectId"] if obj else None


def execute(env, plan: Plan, verdict: Verdict) -> Outcome:
    """Carry out `plan` on `env` unless the verdict blocks it."""
    if verdict.ruling != Ruling.ALLOW.value:
        # Blocked or awaiting confirmation: nothing is executed.
        return Outcome(executed=False, action=plan.action, target=plan.target,
                       success=False, ruling=verdict.ruling,
                       object_state=env.object_state(),
                       note="; ".join(verdict.reasons))

    action = plan.action
    ok = False
    if action == "PICKUP":
        oid = _resolve(env, plan.target, "Apple")
        ok = env.pickup(oid) if oid else False
    elif action == "PUT":
        oid = _resolve(env, plan.target, None)   # the planner must name where
        ok = env.put(oid) if oid else False
    elif action == "DROP":
        ok = env.drop()
    elif action == "NAVIGATE":
        ok = env.navigate(plan.target or "")
    elif action == "DONE":
        ok = True
    else:
        return Outcome(executed=False, action=action, target=plan.target,
                       success=False, ruling=verdict.ruling,
                       object_state=env.object_state(), note="unparseable action")

    return Outcome(executed=True, action=action, target=plan.target, success=ok,
                   ruling=verdict.ruling, object_state=env.object_state())
