"""
======
Logger
======
Initialization of the module logger.

Colorized output, file and stream handlers are configured in :func:`configure_logger`.

"""
from .logs import logger, log_exception, use_log_level, set_log_level

__all__ = ["logger", "log_exception", "use_log_level", "set_log_level"]
