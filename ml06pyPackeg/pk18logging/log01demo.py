import logging

print("")
# logging.basicConfig(level=logging.DEBUG)        # 配置日志级别
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FORMAT = '%(asctime)s - %(levelname)-10s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s'
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"        # 日期格式
logging.basicConfig(filename='my.log', format=LOG_FORMAT, datefmt=DATE_FORMAT)
logging.debug("This is a debug log.")
logging.info("This is a info log.")
logging.warning("This is a warning log.")
logging.error("This is a error log.")
logging.critical("This is a critical log.")
