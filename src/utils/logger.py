"""Structured logging setup."""

from __future__ import annotations

import logging
import sys

import structlog


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure structlog for the whole process. Call once at startup."""
    upper = level.upper()
    if upper not in _VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level!r}. Must be one of {_VALID_LOG_LEVELS}")
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, upper),
    )
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, upper)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "faceswap") -> structlog.BoundLogger:
    return structlog.get_logger(name)
