import logging
import os
import time

default_log_path = './logs'


def formatted_logger(label, level=None, format=None, date_format=None, file_path=None):
    log = logging.getLogger(label)
    if level is None:
        level = logging.INFO
    elif level.lower() == 'debug':
        level = logging.DEBUG
    elif level.lower() == 'info':
        level = logging.INFO
    elif level.lower() == 'warn':
        level = logging.WARN
    elif level.lower() == 'error':
        level = logging.ERROR
    elif level.lower() == 'critical':
        level = logging.CRITICAL
    log.setLevel(level)

    if format is None:
        format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
    if date_format is None:
        date_format = '%Y-%m-%d %H:%M:%S'
    if file_path is None:
        if not os.path.exists(default_log_path):
            os.makedirs(default_log_path)
        file_path = '%s/%s.%s.log.txt' % (default_log_path, label, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    formatter = logging.Formatter(format, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    return log
