from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from recruitx_app.core.config import settings
from recruitx_app.api.v1.api import api_router

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you'll want to restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
def root():
    """
    Redirect to API documentation
    """
    return RedirectResponse(url="/docs")

@app.get("/ping", tags=["Health Check"])
def pong():
    """
    Simple health check endpoint.
    """
    return {"ping": "pong!"}

# To run the app (from the project root directory):
# uvicorn recruitx_app.main:app --reload 