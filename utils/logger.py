#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 14:09 
# ide： PyCharm

import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler

from configs.path_config import LOG_PATH, LOG_NAME

LOGGER_LEVEL = logging.INFO
# 日志保存个数
BACKUP_COUNT = 30

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH, exist_ok=True)


def setup_log(log_path, log_name):
    logging.basicConfig(level=logging.ERROR)
    # 创建logger对象。传入logger名字
    logger = logging.getLogger(log_name)
    log_path = os.path.join(log_path, log_name)
    # 设置日志记录等级
    logger.setLevel(LOGGER_LEVEL)
    # interval 滚动周期，
    # when="MIDNIGHT", interval=1 表示每天0点为更新点，每天生成一个文件
    file_handler = TimedRotatingFileHandler(
        filename=log_path, when="MIDNIGHT", interval=1, backupCount=BACKUP_COUNT
    )
    # 设置时间
    file_handler.suffix = "%Y-%m-%d.log"
    # extMatch是编译好正则表达式，用于匹配日志文件名后缀
    # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
    # 定义日志输出格式
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
        )
    )
    logger.addHandler(file_handler)
    return logger


logger = setup_log(LOG_PATH, LOG_NAME)
