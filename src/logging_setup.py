"""
one logger for both scripts, so output carries a timestamp and a level.

print statements are fine until something runs unattended, at which point there
is no way to tell a result from a warning, or to know when a step started.
"""

import logging
import sys

FORMAT = "%(asctime)s %(levelname)-7s %(name)-16s %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level=logging.INFO, log_file=None):
    """configures the root logger once. safe to call from either entry point."""
    root = logging.getLogger()
    if root.handlers:
        return root

    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
        root.addHandler(handler)

    return root


def get_logger(name):
    return logging.getLogger(name)
