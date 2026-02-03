"""
Compatibility shim for backend imports.

Core storage logic lives in phase.services.project_store so offline scripts and
the backend share the same metadata behavior.
"""

from phase.services.project_store import *  # noqa: F401,F403
