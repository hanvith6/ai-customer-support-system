"""
Logging configuration for AI Customer Support System.
"""

import logging
import sys


def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("ai_support")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()
