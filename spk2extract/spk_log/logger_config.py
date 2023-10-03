from __future__ import annotations

import logging
import sys
from pathlib import Path
import termcolor


def colored_formatter(message: str, log_method: str):
    color = "white"
    if log_method == "info":
        color = "green"
    elif log_method == "debug":
        color = "yellow"
    elif log_method in ["warning", "error", "critical"]:
        color = "red"
    return termcolor.colored(message, color)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)
        return colored_formatter(log_message, record.levelname.lower())


def configure_logger(name: str, log_file: Path | str, level: int = logging.INFO) -> logging.Logger | None:
    if 'sphinx' in sys.modules:
        return None

    if not Path(log_file).exists():
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).touch()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = CustomFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler with custom colored formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger