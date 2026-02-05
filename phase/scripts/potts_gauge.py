from __future__ import annotations

import argparse
from pathlib import Path

from phase.potts.potts_model import (
    load_potts_model,
    load_potts_model_metadata,
    save_potts_model,
    zero_sum_gauge_model,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert a Potts model NPZ into zero-sum gauge and save as a new NPZ.")
    ap.add_argument("--input", required=True, help="Input Potts model NPZ path.")
    ap.add_argument("--output", default="", help="Output NPZ path (default: <input>.gauge0.npz).")
    args = ap.parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input model not found: {in_path}")

    out_path = Path(args.output).expanduser()
    if not str(out_path).strip():
        out_path = in_path.with_suffix("").with_suffix(".gauge0.npz")
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_potts_model(str(in_path))
    meta = load_potts_model_metadata(str(in_path)) or {}
    meta = dict(meta)
    meta["gauge"] = "zero_sum"
    meta["gauge_source"] = str(in_path.name)

    gauged = zero_sum_gauge_model(model)
    save_potts_model(gauged, str(out_path), metadata=meta)
    print(f"[potts_gauge] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

