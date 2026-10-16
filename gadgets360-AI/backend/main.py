"""
Gadgets360 AI Assistant Backend.

FastAPI application entrypoint. Wires up routers, CORS, and the
background session-cleanup task. Run with:

    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.services.session_service import start_session_cleanup_task
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start background tasks on startup, cancel on shutdown."""
    logger.info("Starting %s in %s mode", settings.APP_NAME, settings.APP_ENV)
    cleanup_task = asyncio.create_task(start_session_cleanup_task())

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Backend API for the Gadgets360 AI Assistant. Parses user queries, "
        "validates domain scope, routes to Gadgets360 content APIs, and "
        "generates grounded, data-backed responses."
    ),
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so unexpected errors never leak stack traces to clients."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


@app.get("/")
async def root():
    """Root endpoint with basic API info."""
    return {
        "name": settings.APP_NAME,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "chat": "/api/chat",
    }
