"""Centralized logging configuration.

Call :func:`setup_logging` once at application startup (it is idempotent).
Every module then uses ``logging.getLogger(__name__)`` and the messages
propagate to the root logger configured here.

Logs are written to both the console (stdout) and a rotating file
(``settings.log_file``, default ``agent.log``).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import settings

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are very chatty at INFO level.
_NOISY_LOGGERS = (
    "httpx", "httpcore", "openai",
    "uvicorn.access",
    "watchfiles", "numexpr", "numexpr.utils",
    "psycopg", "psycopg_pool",
)

_configured = False


def setup_logging() -> None:
    """Configure the root logger once."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Add a handler only if the root logger does not already have one (for
    # example, when running under uvicorn a handler may already exist).
    if not root.handlers:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Keep noisy third-party libraries from flooding the logs.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # psycopg_pool's background reaper retries connections and spams warnings
    # during normal operation. Our own init_pool handles startup failures.
    logging.getLogger("psycopg_pool").setLevel(logging.ERROR)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin convenience wrapper)."""
    return logging.getLogger(name)
