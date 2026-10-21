import argparse
import importlib.util
import json
import sys

from .model import MODEL_SPECS, LocalModel
from .profile import Profile
from .service import LyricService


def parser():
    p = argparse.ArgumentParser(description="Local English lyrics with configurable soft rhyme weights")
    p.add_argument("--profile", help="YAML profile (per-request JSON profile overrides this)")
    p.add_argument("--quantization", choices=["nf4", "none"], default="nf4")
    p.add_argument("--model", choices=list(MODEL_SPECS), default="9b", help="Pinned Qwen3.5 size (default: 9b)")
    sub = p.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare", help="Explain a rhyme, no model needed")
    compare.add_argument("candidate")
    compare.add_argument("target")
    find = sub.add_parser("find", help="Search CMUdict with soft phonetic weights")
    find.add_argument("target")
    find.add_argument("--limit", type=int, default=20)
    for command in ["rank", "suggest"]:
        child = sub.add_parser(command)
        child.add_argument("--request", required=True, help="JSON request file, or - for stdin")
    sub.add_parser("serve", help="Run MCP over stdio")
    sub.add_parser("status", help="Show pinned model configuration without loading")
    sub.add_parser("download", help="Download the selected pinned model and tokenizer")
    return p


def main():
    args = parser().parse_args()
    try:
        profile = Profile.load(args.profile) if args.profile else Profile()
        service = LyricService(profile, LocalModel(args.quantization, model=args.model))
        if args.command == "serve":
            # NumPy's Windows extension initialization can stall once stdio reader
            # threads are active. Import it first, without loading torch or weights.
            if sys.platform == "win32" and importlib.util.find_spec("numpy") is not None:
                importlib.import_module("numpy")
            from .server import create_server

            create_server(service).run(transport="stdio")
            return
        if args.command == "compare":
            result = service.compare(args.candidate, args.target)
        elif args.command == "find":
            result = service.find(args.target, args.limit)
        elif args.command == "status":
            result = service.model.status()
        elif args.command == "download":
            result = {"snapshot_path": service.model.download(), **service.model.status()}
        else:
            if args.request == "-":
                request = json.load(sys.stdin)
            else:
                with open(args.request, encoding="utf-8-sig") as source:
                    request = json.load(source)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            result = getattr(service, args.command)(**request)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError, RuntimeError, ImportError, MemoryError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from None
