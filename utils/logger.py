import logging
import os
from logging import FileHandler, Formatter
from dotenv import load_dotenv


class Logger:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self._logger_name = 'APP_LOG'
        self._logger = logging.getLogger('APP_LOG')
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_path = os.getenv('LOG_FILE')
        directory_path = os.getenv('LOG_DIR')
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        if debug_mode:
            logHandler = logging.StreamHandler()
            logHandler.setLevel(logging.DEBUG)
        else:
            logHandler = FileHandler(file_path)
        logHandler.setFormatter(formatter)
        self._logger.addHandler(logHandler)

    def info(self, message: dict):

        try:
            self._logger.info(message)
        except Exception as e:
            self._logger.info("CommonLoggerError: %s" % e, exc_info=True)

    def debug(self, message: dict):
        try:
            self._logger.debug(message)
        except Exception as e:
            self._logger.info("CommonLoggerError: %s" % e, exc_info=True)

    def error(self, message: dict):
        try:
            self._logger.error(message)
        except Exception as e:
            self._logger.info("CommonLoggerError: %s" % e, exc_info=True)