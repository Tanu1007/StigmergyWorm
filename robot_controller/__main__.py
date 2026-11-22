"""Run the controller foundation. This entry point never starts Gazebo or actuates."""
import argparse
import json
from pathlib import Path
from .ollama import OllamaModel, Frame
from .schemas import Task, Limits
from .telemetry import Ledger
from .orchestrator import Controller


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Inspect Station 1 and report what is present.")
    p.add_argument("--goal-target", default="station_1")
    p.add_argument("--allowed-target", action="append", default=None)
    p.add_argument("--image", type=Path)
    p.add_argument("--observation", action="append", default=[])
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--trace-root", type=Path, default=Path("runs/controller"))
    p.add_argument("--semantic-verifier", choices=["on", "off"], default="on")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=6)
    args = p.parse_args()
    try:
        task = Task(args.task, args.goal_target, tuple(args.allowed_target or [args.goal_target]))
        limits = Limits(max_rounds=args.max_rounds, max_total_workers=args.max_workers)
        frame = Frame(args.image) if args.image else None
        model = OllamaModel(args.model, args.ollama_url)
        manifest = model.preflight()
        ledger = Ledger(args.trace_root)
        ledger.emit("backend_manifest", "controller", **manifest)
        result = Controller(task, model, ledger, frame, args.observation, limits,
                            args.semantic_verifier == "on").run()
    except (ValueError, RuntimeError, OSError) as exc:
        p.exit(2, f"Controller failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2))
    print("Local evidence:", ledger.directory)
    print("No robot actions executed. This is controller shadow validation only.")


if __name__ == "__main__":
    main()
