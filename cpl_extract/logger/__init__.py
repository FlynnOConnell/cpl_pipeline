"""
======
Logger
======
Initialization of the module logger.

Colorized output, file and stream handlers are configured in :func:`configure_logger`.

"""
from .logs import cpl_logger, log_exception, use_log_level, set_log_level, setup_file_logging


__all__ = ["cpl_logger", "log_exception", "use_log_level", "set_log_level", "setup_file_logging"]
