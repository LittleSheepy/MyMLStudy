#!/usr/bin/env python
# -*- coding:utf-8 -*-


'''
@CreateTime: 2020/12/29 14:08
@Author : shouke
'''

import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            'format': '%(asctime)s %(filename)s %(lineno)s %(levelname)s %(message)s',
        },
        "plain": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "console_plain": {
            "class": "logging.StreamHandler",
            "level": logging.INFO,
            "formatter": "plain"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": 20,
            "filename": "./log.txt",
            "formatter": "default",
        }
    },
    "loggers": {
        "console_logger": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "console_plain_logger": {
            "handlers": ["console_plain"],
            "level": "DEBUG",
            "propagate": False,
        },
        "file_logger": {
            "handlers": ["file"],
            "level": "INFO",
            "propagate": False,
        }
    },
    "disable_existing_loggers": True,
}
if __name__ == '__main__':

    # 运行测试
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("console_logger")
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    logger.critical('critical message')
