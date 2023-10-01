"""
======
Logger
======
Initialization of the module logger.

Colorized output, file and stream handlers are configured in :func:`configure_logger`.

"""
from spk2extract.spk_log.logger_config import configure_logger

__all__ = ["configure_logger"]