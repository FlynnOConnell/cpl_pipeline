"""
======
Logger
======
Initialization of the module logger.

Colorized output, file and stream handlers are configured in :func:`configure_logger`.

"""
# from .logs import cpl_logger, log_exception, use_log_level, set_log_level
from .logs import setUpLogging


# __all__ = ["cpl_logger", "log_exception", "use_log_level", "set_log_level"]
__all__ = ["setUpLogging"]
