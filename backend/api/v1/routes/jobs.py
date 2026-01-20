from fastapi import APIRouter, HTTPException, Request
from redis import RedisError
from rq.job import Job

from backend.api.v1.common import get_queue


router = APIRouter()


@router.get("/health/check", summary="End-to-end system health check")
async def health_check(request: Request):
    """
    Performs an end-to-end health check of the API, Redis connection,
    and RQ worker availability.
    """
    report = {
        "api_status": "ok",
        "redis_status": {"status": "unknown"},
        "worker_status": {"status": "unknown"},
    }

    redis_conn = request.app.state.redis_conn
    if redis_conn:
        try:
            redis_conn.ping()
            report["redis_status"] = {"status": "ok", "info": "Connected and ping successful."}
        except RedisError as exc:
            report["redis_status"] = {"status": "error", "error": f"Redis connection failed: {str(exc)}"}
    else:
        report["redis_status"] = {"status": "error", "error": "Redis client failed to initialize."}

    if report["redis_status"]["status"] == "ok":
        task_queue = request.app.state.task_queue
        try:
            report["worker_status"] = {
                "status": "ok",
                "queue_name": task_queue.name,
                "queue_length": task_queue.count,
            }
        except Exception as exc:
            report["worker_status"] = {"status": "error", "error": f"Error interacting with RQ queue: {str(exc)}"}

    if report["redis_status"]["status"] != "ok" or report["worker_status"]["status"] != "ok":
        raise HTTPException(status_code=503, detail=report)
    return report


@router.get("/job/status/{job_id}", summary="Get the live status of a running job")
async def get_job_status(job_id: str, request: Request):
    """
    Polls RQ for the live status of an enqueued job.
    The job_id here is the RQ job ID, not our UUID.
    """
    task_queue = get_queue(request)
    try:
        job = Job.fetch(job_id, connection=task_queue.connection)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found in RQ.")

    status = job.get_status()
    response = {
        "job_id": job_id,
        "status": status,
        "meta": job.meta,
    }

    if status in {"finished", "failed"}:
        response["result"] = job.result

    return response
