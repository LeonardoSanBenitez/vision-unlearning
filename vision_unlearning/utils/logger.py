import logging
import sys
from typing import List


FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] %(name)-8s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
)


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger('libs.' + name)
    # log to sys.stdout for backward compatibility.
    # TODO: May need to be removed in the future, after local/blob file stream are fully supported.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(FORMATTER)
    logger.addHandler(stdout_handler)
    logger.setLevel(level)
    return logger


def setup_loggers(
    root_level: int = logging.WARNING,
    modules_debug: List[str] = [],
    modules_info: List[str] = [],
    modules_warning: List[str] = [],
    modules_error: List[str] = [],
) -> None:
    '''
    Module names are matched with "contains"
    for example: `azure.` will match all azure modules

    All non-specified modules go to WARNING

    ---

    You should called it in the main file of your application, after all imports
    caveat: imports done after this call will not be affected
    '''
    logging.root.setLevel(root_level)

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if any([m in logger.name for m in modules_debug]):
            logger.setLevel(logging.DEBUG)
        elif any([m in logger.name for m in modules_info]):
            logger.setLevel(logging.INFO)
        elif any([m in logger.name for m in modules_warning]):
            logger.setLevel(logging.WARNING)
        elif any([m in logger.name for m in modules_error]):
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.WARNING)
