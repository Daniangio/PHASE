from __future__ import annotations

"""Thin CLI wrapper around the simulation pipeline.

The full pipeline implementation lives in phase.simulation.pipeline.
"""

from phase.simulation.pipeline import _build_arg_parser, parse_args, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_pipeline(args, parser=parser)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
