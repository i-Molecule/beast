import datetime
import logging
import os
from typing import Optional, Union


logging_env_var = "LOGGING_LEVEL"


def get_logging_level() -> str:
    """Retrieves logging level string from environmental variable."""
    return os.getenv(logging_env_var)


if os.getenv(logging_env_var) is None:
    os.environ[logging_env_var] = "INFO"


# from https://stackoverflow.com/questions/44691558/suppress-multiple-messages-with-same-content-in-python-logging-module-aka-log-co
class DuplicateFilter(logging.Filter):
    """Filter for removal of repeated messages in logger."""

    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False


class LogFormatter(logging.Formatter):
    def __init__(
        self,
        *args,
        time_from_start: bool = True,
        log_name: bool = False,
        **kwargs,
    ):
        if time_from_start:
            s = "%(delta)s"
        else:
            s = "%(asctime)s"
            kwargs["datefmt"] = "%Y-%m-%d:%H:%M:%S"
        s += " {%(filename)-22s:%(lineno)4d}"
        if log_name:
            s += " %(name)-30s"
        s += " %(levelname)-8s: %(message)s"
        super().__init__(s, *args, **kwargs)
        self.time_from_start = time_from_start

    def format(self, record):
        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S")
        return super().format(record)


def init_logger(
    name: Optional[str] = None,
    level: Optional[str] = None,
    filename: Union[os.PathLike, None] = None,
    level_file: str = "INFO",
    level_stream: str = "DEBUG",
    **kwargs,
):
    """Creates logger."""

    logger = logging.getLogger(name if name else "tetrad")

    if getattr(logger, "_tetrad_configured", False):
        return logger

    if level is None:
        # getting logging level from environmental variable
        #   set it with:
        #   export LOGGING_LEVEL=DEBUG
        #   DEBUG / INFO / WARNING / ERROR
        level = os.getenv(logging_env_var)
        if not level:
            level = None
    if level is None:
        level = "INFO"

    level = logging.getLevelName(level)
    level_file = logging.getLevelName(level_file)
    level_stream = logging.getLevelName(level_stream)

    logger.setLevel(level)
    logger.propagate = False

    formatter = LogFormatter(**kwargs)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level_stream)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        dirname = filename.parent
        dirname.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(filename, "w", "utf-8")
        file_handler.setLevel(level_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.addFilter(DuplicateFilter())

    logger._tetrad_configured = True
    return logger
