import logging
from pathlib import Path
from os import getppid, path
import pandas as pd
from datetime import datetime
from functools import wraps
from inspect import stack
from sys import stderr, stdout, exit as sys_exit

# ==================================================
# CONTRIBUTION START: Logger Setup and Decorator
# Contributor: Leslie Horace
# ==================================================

class LoggerHandler:
    def __init__(self, logger):
        self.logger = logger

    def _log_message(self, log_func, msg, stack_level=2, exc_info=False):

        frame = stack()[stack_level]
        filename = Path(frame.filename).name
        function_name = frame.function
        lineno = frame.lineno
        log_func(f"[{filename}:{function_name}:{lineno}] || {msg}", exc_info=exc_info)

    def critical(self, msg: str = "unknown critical msg"):
        self._log_message(self.logger.critical, msg, stack_level=2, exc_info=True)
        sys_exit(-1)

    def error(self, msg: str = "unknown error msg"):
        self._log_message(self.logger.error, msg, stack_level=2, exc_info=True)
        sys_exit(-1)

    def warning(self, msg: str = "unknown warning msg"):
        self._log_message(self.logger.warning, msg, stack_level=2)

    def info(self, msg: str = "unknown info msg"):
        self._log_message(self.logger.info, msg, stack_level=2)

    def debug(self, msg: str = "unknown debug msg"):
        self._log_message(self.logger.debug, msg, stack_level=2)


class SingletonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance.logger = None
            cls._instance.handlers = {}
            cls.base_logger = None
            cls.default_name = None
        return cls._instance
    
    def clean_logger_name(self, logger_name: str):
        if not isinstance(logger_name, str) or not logger_name.strip():
            raise Exception("logger_name must be a none empty string")
        clean_name = path.basename(logger_name) if path.exists(logger_name) else logger_name
        return clean_name.replace(" ", "_").replace(".py", "")
    
    def setup_logger(self, module_name, log_stdout=False, log_stderr=True):
        logger_name = self.clean_logger_name(module_name)

        if self.base_logger is None:
            unique_id = f"{logger_name}_ppid{getppid()}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
            log_file = f"{unique_id}.log"

            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            logs_path = logs_dir / log_file

            base_logger = logging.getLogger("shared_logger")
            base_logger.setLevel(logging.DEBUG)

            if not base_logger.handlers:
                formatter = logging.Formatter(
                    "[%(asctime)s] || [%(levelname)s] || %(message)s"
                )
                file_handler = logging.FileHandler(logs_path)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                base_logger.addHandler(file_handler)

                if log_stdout:
                    stdout_handler = logging.StreamHandler(stdout)
                    stdout_handler.setLevel(logging.INFO)
                    stdout_handler.setFormatter(formatter)
                    base_logger.addHandler(stdout_handler)

                if log_stderr:
                    stderr_handler = logging.StreamHandler(stderr)
                    stderr_handler.setLevel(logging.ERROR)
                    stderr_handler.setFormatter(formatter)
                    base_logger.addHandler(stderr_handler)

            self.base_logger = base_logger

        child = self.base_logger.getChild(logger_name)
        self.handlers[logger_name] = LoggerHandler(child)
        self.default_name =logger_name
        return child

    def get_handler(self, module_name=None):
        logger_name = self.default_name if module_name is None else self.clean_logger_name(module_name)
        if not logger_name or logger_name not in self.handlers:
            raise ValueError("Logger is not initialized. Call init_shared_logger() first.")
        return self.handlers[logger_name]


def get_logger(module_name=None):
    return SingletonLogger().get_handler(module_name)


def init_shared_logger(module_name, log_stdout=False, log_stderr=True):
    module_str = str(module_name)
    module_clean = Path(module_str).stem if path.splitext(module_str)[1] else module_str
    logger_instance = SingletonLogger()
    logger = logger_instance.setup_logger(module_clean, log_stdout, log_stderr)
    handler = logger_instance.get_handler()
    logger.info(f"init_shared_logger: {module_clean}")
    return handler

def set_logger_level(level: int = 20):
    if level not in logging._levelToName:
        return

    inst = SingletonLogger()
    base = inst.base_logger
    if base is None:
        return

    base.setLevel(level)
    if inst.logger is not None:
        inst.logger.setLevel(level)

    for h in base.handlers:
        h.setLevel(logging.WARNING) if getattr(h, "stream", None) is stderr else h.setLevel(level)


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.now()
        logger.info(f"Starting clock for {func.__name__} at {start_time}")
        result = func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"Ending clock for {func.__name__} at {end_time}")
        elapsed_time = (end_time - start_time).total_seconds()
        logger.info(f"Elapsed time for {func.__name__} is {elapsed_time:.6f} seconds")
        return result
    return wrapper

# ==================================================
# CONTRIBUTION END: Logger Setup and Decorator
# ==================================================
