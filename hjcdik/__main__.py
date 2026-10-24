"""Command line: `hjcdik build <urdf> --ee-link <fixed joint> [--spheres preset|file]`, `hjcdik list`, `hjcdik info`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hjcdik", description="HJCD-IK runtime robot builds")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build (or reuse) the solver for a URDF through GRiD's artifact store")
    b.add_argument("urdf")
    b.add_argument("--ee-link", required=True, help="fixed joint naming the end-effector target frame")
    b.add_argument("--spheres", default=None, help="collision sphere model: a preset name or a spherized URDF")
    b.add_argument("--name", default=None, help="artifact name (default <robot>-<ee_link>[-cc])")
    b.add_argument("--cache-dir", default=None)
    b.add_argument("--cuda-arch", type=int, default=None)
    b.add_argument("--force", action="store_true", help="recompile every unit")
    b.add_argument("--plan", action="store_true", help="report what a build would use without building")
    sub.add_parser("list", help="prepared builds in the store").add_argument("--cache-dir", default=None)
    sub.add_parser("info", help="the packaged robot's build metadata")
    args = parser.parse_args(argv)

    from . import _build
    if args.command == "info":
        from ._hjcdik import build_info
        print(json.dumps(build_info(), indent=2))
        return 0
    try:
        if args.command == "list":
            for entry in _build.list_builds(cache_dir=args.cache_dir):
                print(json.dumps(entry, sort_keys=True))
            return 0
        result = _build.build(args.urdf, ee_link=args.ee_link, spheres=args.spheres, name=args.name,
                              cache_dir=args.cache_dir, cuda_arch=args.cuda_arch,
                              force_rebuild=args.force, plan_only=args.plan)
    except _build.BuildError as error:
        print(f"hjcdik build: {error}", file=sys.stderr)
        return 1
    if args.plan:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"{'built' if result.compiled else 'reused'} {result.name}: {result.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
