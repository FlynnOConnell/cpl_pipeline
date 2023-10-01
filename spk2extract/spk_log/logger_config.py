from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pytz

def configure_logger(name: str, log_file: Path | str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger with file and stream handlers.

    Parameters
    ----------
    name : str
        Name of the logger.
    log_file : Path | str
        Path to the log file.
    level : int, optional
        Logging level, by default spk_log.INFO

    Returns
    -------
    logging.Logger
        Configured logger object.
    """

    if 'sphinx' in sys.modules:
        return logging.getLogger(name)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
