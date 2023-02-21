# -*- coding: utf-8 -*-
import logging
from typing import Union

FORMAT = "%(asctime)s:%(pathname)s:%(levelname)s:%(lineno)d:%(message)s"


def get_logger(name: str, log_level: Union[int, str] = logging.INFO) -> logging.Logger:
    """
    Builds a `Logger` instance with provided name and log level.
    Args:
        name: The name for the logger.
        log_level: The default log level.
    Returns: The logger.
    """

    logger = logging.getLogger(name=name)
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=FORMAT)
    stream_handler.setFormatter(fmt=formatter)
    logger.addHandler(stream_handler)

    return logger
