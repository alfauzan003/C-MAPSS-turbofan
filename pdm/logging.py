"""Structured JSON logging via structlog.

Each service calls `configure_logging(service="<name>")` once at startup.
After that, modules use `get_logger(__name__)` to get a bound logger that
emits JSON to stdout, including the service name on every line.
"""

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", service: str = "pdm") -> None:
    """Configure structlog + stdlib logging to emit JSON to stdout.

    Idempotent: safe to call multiple times.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bind the service name so every log line carries it.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
