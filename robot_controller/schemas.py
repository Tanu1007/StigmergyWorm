"""Strict data contracts. Model output never defines code or tool permissions."""
from dataclasses import dataclass

ROLES = ("observe", "plan", "check")
ACTIONS = ("navigate", "inspect", "stop")


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected or missing fields")
    return value


def text(value, limit=1200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Expected a nonempty bounded string")
    return value


@dataclass(frozen=True)
class Task:
    instruction: str
    goal_target: str
    allowed_targets: tuple[str, ...]

    def __post_init__(self):
        text(self.instruction, 2000)
        if not isinstance(self.allowed_targets, (tuple, list)):
            raise ValueError("Target catalogue must be a tuple or list")
        object.__setattr__(self, "allowed_targets", tuple(self.allowed_targets))
        if not self.allowed_targets or len(self.allowed_targets) > 12:
            raise ValueError("Provide 1–12 operator-defined targets")
        for target in self.allowed_targets:
            text(target, 80)
        if len(set(self.allowed_targets)) != len(self.allowed_targets):
            raise ValueError("Duplicate target IDs")
        if self.goal_target not in self.allowed_targets:
            raise ValueError("Goal target must be in the trusted catalogue")

    def payload(self):
        return {"instruction": self.instruction, "goal_target": self.goal_target,
                "allowed_targets": list(self.allowed_targets)}


@dataclass(frozen=True)
class Limits:
    max_rounds: int = 3
    max_workers_per_round: int = 3
    max_total_workers: int = 6

    def __post_init__(self):
        for value, upper in [(self.max_rounds, 8), (self.max_workers_per_round, 4),
                             (self.max_total_workers, 12)]:
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("Controller limits outside bounded range")


@dataclass(frozen=True)
class Action:
    kind: str
    target: str | None

    @classmethod
    def parse(cls, value):
        exact(value, ("kind", "target"))
        if value["kind"] not in ACTIONS:
            raise ValueError("Unregistered action")
        target = value["target"]
        if value["kind"] == "stop":
            if target is not None:
                raise ValueError("Stop must have a null target")
        else:
            text(target, 80)
        return cls(value["kind"], target)

    def payload(self):
        return {"kind": self.kind, "target": self.target}


def gate(task: Task, action: Action, semantic_verification: bool):
    """Off omits only the task-target check, never the capability envelope.

    This package does not execute either accepted or rejected proposals.
    """
    in_envelope = action.kind == "stop" or action.target in task.allowed_targets
    task_conflict = action.kind != "stop" and action.target != task.goal_target
    allowed = in_envelope and not (semantic_verification and task_conflict)
    reason = "outside_target_catalogue" if not in_envelope else (
        "task_target_mismatch" if semantic_verification and task_conflict else "proposal_admitted")
    return {"allowed": allowed, "reason": reason, "task_conflict": task_conflict,
            "semantic_verification": semantic_verification,
            "execution": "not_executed_shadow"}


def object_schema(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


ACTION_SCHEMA = {"anyOf": [
    object_schema({"kind": {"enum": ["navigate", "inspect"]},
                   "target": {"type": "string", "minLength": 1, "maxLength": 80}}),
    object_schema({"kind": {"enum": ["stop"]}, "target": {"type": "null"}})
]}
JOB_SCHEMA = object_schema({"role": {"enum": list(ROLES)},
                            "question": {"type": "string", "minLength": 1, "maxLength": 1200}})
SUPERVISOR_SCHEMA = {"anyOf": [
    object_schema({"decision": {"enum": ["delegate"]},
                   "jobs": {"type": "array", "minItems": 1, "maxItems": 4, "items": JOB_SCHEMA},
                   "action": {"type": "null"},
                   "reason": {"type": "string", "minLength": 1, "maxLength": 1200}}),
    object_schema({"decision": {"enum": ["finish"]},
                   "jobs": {"type": "array", "maxItems": 0, "items": JOB_SCHEMA},
                   "action": ACTION_SCHEMA,
                   "reason": {"type": "string", "minLength": 1, "maxLength": 1200}})
]}
WORKER_SCHEMA = {"type": "object", "additionalProperties": False,
                 "properties": {"summary": {"type": "string", "maxLength": 2000}},
                 "required": ["summary"]}


def parse_supervisor(value, limits):
    exact(value, ("decision", "jobs", "action", "reason"))
    text(value["reason"])
    jobs = value["jobs"]
    if not isinstance(jobs, list) or len(jobs) > limits.max_workers_per_round:
        raise ValueError("Invalid worker count")
    if value["decision"] == "delegate":
        if not jobs or value["action"] is not None:
            raise ValueError("Delegation needs jobs and no action")
        for job in jobs:
            exact(job, ("role", "question"))
            if job["role"] not in ROLES:
                raise ValueError("Unregistered worker role")
            text(job["question"])
    elif value["decision"] == "finish":
        if jobs or value["action"] is None:
            raise ValueError("Finish needs one typed action and no jobs")
        Action.parse(value["action"])
    else:
        raise ValueError("Invalid routing decision")
    return value
