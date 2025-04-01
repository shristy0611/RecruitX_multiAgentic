from fastapi import APIRouter

from recruitx_app.api.v1.endpoints import jobs, scores, candidates

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(scores.router, prefix="/scores", tags=["scores"])
api_router.include_router(candidates.router, prefix="/candidates", tags=["candidates"])

# Add more endpoint routers here as they are created
# api_router.include_router(candidates.router, prefix="/candidates", tags=["candidates"]) 