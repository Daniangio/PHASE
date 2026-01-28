from __future__ import annotations

from typing import Optional, Tuple


def parse_slice_spec(raw_value: Optional[str]) -> Tuple[Optional[str], int]:
    """
    Parse a slice spec string in the form start:stop:step (each optional).
    Returns (slice_spec_str_or_none, stride_step).
    Accepts a single integer as a shorthand for step.
    """
    if raw_value is None:
        return None, 1
    value = str(raw_value).strip()
    if not value:
        return None, 1

    if ":" not in value:
        step = int(value)
        if step <= 0:
            raise ValueError("Slice step must be >= 1.")
        return f"::{step}" if step != 1 else None, step

    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError("Slice spec must be start:stop:step (at most 3 parts).")

    def _parse_int(item: str, label: str) -> Optional[int]:
        if item == "":
            return None
        val = int(item)
        if val < 0:
            raise ValueError(f"{label} must be >= 0.")
        return val

    start = _parse_int(parts[0], "start")
    stop = _parse_int(parts[1], "stop") if len(parts) > 1 else None
    step = _parse_int(parts[2], "step") if len(parts) > 2 else None
    if step is not None and step <= 0:
        raise ValueError("Step must be >= 1.")
    step_val = step or 1
    slice_spec = f"{'' if start is None else start}:{'' if stop is None else stop}:{'' if step is None else step}"
    if slice_spec == "::":
        return None, 1
    return slice_spec, step_val
