"""
======
Logger
======
Initialization of the module logger.

Colorized output, file and stream handlers are configured in :func:`configure_logger`.

"""
from spk2extract.logging.logger_config import *

__all__ = ["configure_logger", "ColoredFormatter", "ESTFormatter"]
