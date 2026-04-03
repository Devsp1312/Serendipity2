"""
Centralized logging setup.

Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")

Call setup_logging() once at startup (app.py and main.py both do this).
It is idempotent — safe to call multiple times.
"""

import logging
import logging.handlers
import sys

from src.config import LOGS_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def setup_logging() -> None:
    """
    Attaches a rotating file handler (INFO+) and a console handler (WARNING+) to the root logger.
    File: logs/serendipity.log, rotated at 5 MB, 5 backups.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already initialized

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "serendipity.log",
        maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    # Console only shows warnings to avoid cluttering Streamlit's output
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.WARNING)

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
