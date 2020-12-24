import os
import logging
from abc import ABC
import logging.handlers


class LoggingModule(ABC):
    def __init__(self,
                 prefix: str,
                 logger_name: str = 'VVC',
                 stream_loglevel: str = 'INFO',
                 file_loglevel: str = 'DEBUG',
                 logfile: str = None):
        self.logger = logging.getLogger(logger_name)
        if self.logger.handlers:
            return
        self.logger.setLevel(logging.DEBUG)
        # Format
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] <{}> %(message)s'.format(prefix)
        )
        # Add stream handler
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, stream_loglevel.upper()))
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        # Add file handler if filename is given
        if logfile is not None:
            if not os.path.exists(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))
            fh = logging.handlers.RotatingFileHandler(
                filename=logfile,
                mode='a',
                maxBytes=1 * 1024 * 1024,
                backupCount=2
            )
            fh.setLevel(getattr(logging, file_loglevel.upper()))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)