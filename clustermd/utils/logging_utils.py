"""Configure logging and console output for analysis runs."""

import logging
from clustermd.constants import ANSI_ESCAPE

class PlainFileFormatter(logging.Formatter):
    """Strips ANSI color codes for clean file logging."""
    def format(self, record):
        message = super().format(record)
        return ANSI_ESCAPE.sub('', message)


def get_clusterMD_file_handler():
    """Safely retrieves the designated file handler by name."""
    for handler in logging.getLogger().handlers:
        if getattr(handler, "name", None) == "clusterMD_file":
            return handler
    return None

