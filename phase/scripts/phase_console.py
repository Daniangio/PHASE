from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _find_repo_root(start: Path) -> Path | None:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "scripts" / "phase_console.sh").exists():
            return cur
        if (cur / ".git").exists():
            # Prefer the repo root, but still require the script to exist.
            candidate = cur
            if (candidate / "scripts" / "phase_console.sh").exists():
                return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    start = Path(os.getenv("PHASE_REPO_ROOT", "")).expanduser()
    repo = None
    if str(start).strip():
        repo = start.resolve() if (start / "scripts" / "phase_console.sh").exists() else None
    if repo is None:
        repo = _find_repo_root(Path.cwd())
    if repo is None:
        raise SystemExit("Could not locate scripts/phase_console.sh. Set $PHASE_REPO_ROOT or run from the repo.")

    script = repo / "scripts" / "phase_console.sh"
    proc = subprocess.run(["bash", str(script), *argv], check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

