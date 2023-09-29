from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytz


class ESTFormatter(logging.Formatter):
    """
    Formatter for logging timestamps in the US Eastern Time zone.

    Methods
    -------
    formatTime(record, datefmt=None)
        Format the time for the log record.
    """

    def formatTime(self, record, datefmt=None):
        """
        Format the time for the log record.

        Parameters
        ----------
        record : logging.LogRecord
            The log record.
        datefmt : str, optional
            The date format, by default None

        Returns
        -------
        str
            The formatted time string.
        """
        dt = datetime.fromtimestamp(record.created, pytz.timezone('US/Eastern'))
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')


class ColoredFormatter(logging.Formatter):
    """
    Formatter for logging with colored output.

    Attributes
    ----------
    COLORS : dict
        Mapping of log levels to terminal colors.

    Methods
    -------
    format(record)
        Format the log record with colors.
    """
    COLORS = {
        'DEBUG': '\033[33m',  # Orange
        'INFO': '\033[34m',  # Blue
        'WARNING': '\033[31m',  # Red
        'ERROR': '\033[41m',  # Red background
        'CRITICAL': '\033[45m'  # Magenta background
    }

    def format(self, record):
        """
        Format the log record with colors.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted.

        Returns
        -------
        str
            The formatted log message.
        """
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}\033[0m"


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
        Logging level, by default logging.INFO

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

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Create a file handler with rotation
    file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=int(1e6))
    file_handler.setFormatter(
        ESTFormatter('%(filename)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p'))
    file_handler.setLevel(level)

    # Create a stream handler with color
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        ColoredFormatter('%(filename)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p'))
    stream_handler.setLevel(level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
