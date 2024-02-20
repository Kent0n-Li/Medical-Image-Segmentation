import logging

from config import global_logging_txt


def setup_logger(logger_name, output_file=None):
    # Create a logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # Set the logging level to INFO
    logger.propagate = False  # Prevent log messages from being duplicated in the root logger

    # Common formatter for all handlers
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    # Always log to the default "logging.txt" file
    default_fh = logging.FileHandler(global_logging_txt)
    default_fh.setLevel(logging.INFO)
    default_fh.setFormatter(formatter)
    logger.addHandler(default_fh)

    # Additionally, log to another file if output_file is specified
    if output_file is not None:
        fh = logging.FileHandler(output_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
