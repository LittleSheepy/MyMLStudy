import logging

LOGGING_NAME = "my_log"

print("")


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

def set_logging(name=LOGGING_NAME):
    # sets up logging for the given name
    level = logging.INFO
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            name: {
                "format": "%(message)s"},
            "default": {
                # 'format': '%(asctime)s %(filename)s %(lineno)s %(levelname)s %(message)s',
                'format': '%(asctime)s - %(levelname)-10s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s',
                "datefmt": "%m/%d/%Y %H:%M:%S %p"
            },
            "plain": {
                "format": "%(message)s",
            },
        },
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level,
            },
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
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,},

            "console_logger": {
                "handlers": ["console"],
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
        }
    })


if __name__ == '__main__':
    set_logging()
    # logging.config.dictConfig(LOGGING_CONFIG)
    # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_FORMAT = '%(asctime)s - %(levelname)-10s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s'
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"  # 日期格式
    # logging.basicConfig(filename='my.log', format=LOG_FORMAT, datefmt=DATE_FORMAT, level=logging.DEBUG)

    logger = logging.getLogger("console_logger")
    logger.debug("This is a debug log.")
    logger.info("This is a info log.")
    logger.warning("This is a warning log.")
    logger.error("This is a error log.")
    logger.critical("This is a critical log.")





