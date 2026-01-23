"""
Structured logging setup using structlog.
Outputs JSON logs for production, colorized console for development.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from tsn_common.config import LoggingSettings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    event_dict["app"] = "tsn"
    return event_dict


def setup_logging(settings: LoggingSettings | None = None) -> None:
    """
    Configure structlog for the application.

    Args:
        settings: Logging settings. If None, uses defaults.
    """
    if settings is None:
        settings = LoggingSettings()

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.level),
        stream=sys.stdout if settings.output == "stdout" else None,
    )

    # File handler if needed
    if settings.output == "file" and settings.file_path:
        settings.file_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            settings.file_path,
            maxBytes=settings.rotate_mb * 1024 * 1024,
            backupCount=5,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(handler)

    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Final renderer based on format
    if settings.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.plain_traceback,
                )
            ]
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)
