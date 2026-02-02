from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    """Execution policy shared by offline + webserver entry points."""

    allow_multiprocessing: bool = True

    def apply_to_potts_args(self, args) -> None:
        """Mutate a Namespace in-place to honor multiprocessing constraints."""
        if not self.allow_multiprocessing:
            if hasattr(args, "gibbs_chains"):
                args.gibbs_chains = 1
            if hasattr(args, "rex_chains"):
                args.rex_chains = 1
