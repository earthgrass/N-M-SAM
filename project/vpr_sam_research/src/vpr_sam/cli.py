from __future__ import annotations
import argparse
import json
from pathlib import Path

from .config import load_config, validate, recovery_config
from .repro import environment, project_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent VPR-SAM infrastructure; formal execution opt-in")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Standalone project root (all IO confined here)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--allow-formal", action="store_true")
        if name == "train":
            p.add_argument("--resume")
            p.add_argument("--stop-after", type=int)
    p = sub.add_parser("prune")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--protocol", help="JSON file containing explicit pruning configuration")
    p = sub.add_parser("recover")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--allow-formal", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", default="sam")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--checkpoint", choices=("final", "best_validation"), default="final")
    p.add_argument("--recovery-checkpoint", choices=("final", "best_validation"), default="final")
    p.add_argument("--pruning-protocol", help="Select one explicit common N:M protocol when multiple exist")
    p = sub.add_parser("plot")
    p.add_argument("--summary", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-data")
    p.add_argument("--data-root", default="data/cifar10")
    sub.add_parser("environment")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src/vpr_sam").is_dir():
        parser.error("--root must point to the standalone vpr_sam_research project")
    if args.command == "environment":
        result = environment()
    elif args.command == "prepare-data":
        from torchvision.datasets import CIFAR10
        location = project_path(root, args.data_root)
        CIFAR10(str(location), train=True, download=True)
        CIFAR10(str(location), train=False, download=True)
        result = {"verified_real_cifar10": str(location)}
    elif args.command in {"train", "validate"}:
        c = load_config(project_path(root, args.config))
        validate(c, formal_allowed=args.allow_formal)
        if args.command == "validate":
            result = {"valid": True, "purpose": c["purpose"]}
        else:
            from .training import train
            result = train(c, root, formal_allowed=args.allow_formal, resume=args.resume, stop_after=args.stop_after)
    elif args.command == "prune":
        from .training import prune
        protocol = load_config(project_path(root, args.protocol)) if args.protocol else None
        result = prune(args.checkpoint, root, protocol)
    elif args.command == "recover":
        from .training.engine import read_checkpoint, train
        state, _ = read_checkpoint(args.checkpoint, root)
        result = train(state["config"], root, formal_allowed=args.allow_formal, resume=args.resume,
                       recover_from=args.checkpoint, stop_after=args.stop_after)
    elif args.command == "evaluate":
        from .training import evaluate_checkpoint
        result = evaluate_checkpoint(args.checkpoint, root, args.split)
    elif args.command == "summarize":
        from .analysis.report import summarize
        pruning_protocol = load_config(project_path(root, args.pruning_protocol)) if args.pruning_protocol else None
        result = summarize(root, args.runs, args.output, args.baseline, args.split, args.checkpoint,
                           pruning_protocol, args.recovery_checkpoint)
    elif args.command == "plot":
        from .analysis.report import plot
        result = plot(root, args.summary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
