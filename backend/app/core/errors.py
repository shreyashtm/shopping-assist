"""Error types and handlers.

The assistant's failure policy is "never show an empty screen": recoverable
problems degrade to weaker-but-real results inside the service layer. Only
unrecoverable problems reach these handlers.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base for errors that carry a user-safe message."""

    status_code = 500
    message = "Something went wrong."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        if message:
            self.message = message


class CatalogueNotReady(AppError):
    status_code = 503
    message = "Product catalogue is not loaded. Run scripts/build_index.py."


class ProductNotFound(AppError):
    status_code = 404
    message = "Product not found."


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
