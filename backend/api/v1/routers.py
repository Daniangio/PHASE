"""
API router aggregator for v1.
"""

from fastapi import APIRouter

from backend.api.v1.routes import (
    analysis,
    clusters,
    descriptors,
    jobs,
    metastable,
    projects,
    results,
    states,
    systems,
)


api_router = APIRouter()
api_router.include_router(projects.router)
api_router.include_router(systems.router)
api_router.include_router(states.router)
api_router.include_router(descriptors.router)
api_router.include_router(metastable.router)
api_router.include_router(clusters.router)
api_router.include_router(analysis.router)
api_router.include_router(jobs.router)
api_router.include_router(results.router)
