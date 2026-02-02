from __future__ import annotations

from phase.potts import pipeline as sim_main


def main(argv: list[str] | None = None) -> int:
    parser = sim_main._build_arg_parser()
    args = parser.parse_args(argv)
    args.fit_only = False
    try:
        sim_main.run_pipeline(args, parser=parser)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
