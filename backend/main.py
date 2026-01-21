"""
FastAPI Main Application
Initializes the app and includes the API routers.
"""

import os
import inspect

import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from rq import Queue

try:
    from starlette.exceptions import HTTPException as StarletteHTTPException
except Exception as exc:  # pragma: no cover - defensive
    StarletteHTTPException = Exception
    print(f"[startup] Starlette exception import failed: {exc}")

try:  # Guard to avoid import errors in minimal environments
    import starlette.formparsers as formparsers
    import starlette.requests as starlette_requests
except Exception as exc:  # pragma: no cover - defensive
    formparsers = None
    starlette_requests = None
    print(f"[startup] Starlette multipart patch skipped: {exc}")

from backend.api.v1 import routers as v1_routers

# --- Upload limits ---
# Allow large trajectory uploads (up to 5 GB). Starlette/python-multipart
# enforce default per-file/body limits; we patch the multipart parser to raise
# those ceilings during app import and we return a clear 413 if the limit is hit.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


def _configure_multipart_limits(max_bytes: int) -> None:
    """
    Monkey-patch Starlette's Request.form to use a multipart parser configured
    with a higher max file/body size. This keeps FastAPI's UploadFile handling
    but avoids the default ~1 GB cap in python-multipart.
    """
    if not formparsers or not starlette_requests:
        return

    try:
        # Bump any module-level limits exposed by the installed Starlette version.
        for attr in (
            "MAX_FILE_SIZE",
            "MAX_BODY_SIZE",
            "DEFAULT_MAX_FILE_SIZE",
            "DEFAULT_MAX_MULTIPART_BODY_SIZE",
            "DEFAULT_MAX_MEMORY_SIZE",
        ):
            if hasattr(formparsers, attr):
                setattr(formparsers, attr, max_bytes)

        parser_sig = inspect.signature(formparsers.MultiPartParser)
        parser_kwargs = {}
        # Set any size-related arguments the installed version supports.
        for key in (
            "max_file_size",
            "max_body_size",
            "max_multipart_body_size",
            "max_form_memory_size",
        ):
            if key in parser_sig.parameters:
                parser_kwargs[key] = max_bytes

        async def large_form(self, *args, **kwargs):
            # Mirror Starlette's cached form parsing, but use our tuned parser.
            if getattr(self, "_form", None) is None:
                content_type = self.headers.get("content-type", "")
                if content_type and "multipart/form-data" in content_type:
                    parser = formparsers.MultiPartParser(self.headers, self.stream(), **parser_kwargs)
                else:
                    parser = formparsers.FormParser(self.headers, self.stream())
                self._form = await parser.parse()
            return self._form

        starlette_requests.Request.form = large_form
        print(f"[startup] Multipart upload limits set to {max_bytes} bytes.")
    except Exception as exc:  # pragma: no cover - defensive; avoid startup crash
        print(f"[startup] Failed to configure multipart limits: {exc}")


_configure_multipart_limits(MAX_UPLOAD_BYTES)

# --- RQ Setup ---
# Use the environment variable set in docker-compose.yaml
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Redis connection and RQ Queue
try:
    redis_conn = redis.from_url(REDIS_URL)
    # The queue name must match the worker's queue name (phase-jobs)
    task_queue = Queue('phase-jobs', connection=redis_conn)
except Exception as e:
    # If Redis connection fails, we can't enqueue jobs, which will be caught in the health check
    print(f"FATAL: Could not connect to Redis: {e}")
    redis_conn = None
    task_queue = None
# --- End RQ Setup ---


app = FastAPI(
    title="PHASE API",
    description="API for the PHASE analysis and sampling pipeline.",
    version="0.1.0",
)


@app.middleware("http")
async def enforce_upload_ceiling(request: Request, call_next):
    """
    Return a clear 413 early if Content-Length advertises an oversize upload.
    """
    if request.method in {"POST", "PUT", "PATCH"}:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            try:
                content_length = int(request.headers.get("content-length") or 0)
            except ValueError:
                content_length = 0
            if content_length and content_length > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Upload exceeds 5 GB limit ({content_length} bytes received)."},
                )
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def oversize_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Convert Starlette's 400/413 multipart size errors into a consistent message.
    """
    detail = exc.detail or ""
    detail_str = detail if isinstance(detail, str) else str(detail)
    too_large = any(
        keyword in detail_str.lower()
        for keyword in ("too large", "exceed", "exceeded", "over limit", "max file size", "body too large")
    )
    if exc.status_code in (400, 413) and too_large:
        return JSONResponse(
            status_code=413,
            content={"detail": "Trajectory upload exceeds 5 GB limit. Reduce the file size or stride/split the trajectory."},
        )
    return await http_exception_handler(request, exc)


# Pass the queue object to the router so it can enqueue jobs
app.state.task_queue = task_queue
app.state.redis_conn = redis_conn 

# Include the v1 API routes
app.include_router(v1_routers.api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint for health check.
    """
    return {"message": "Welcome to the PHASE API. Go to /docs for details."}
