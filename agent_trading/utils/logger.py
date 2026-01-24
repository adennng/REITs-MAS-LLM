#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志配置模块
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.config import LOG_DIR, LOG_FILE_PREFIX


def setup_logger(name: str = "trading_agent", log_to_file: bool = True) -> logging.Logger:
    """
    配置日志器

    Args:
        name: 日志器名称
        log_to_file: 是否输出到文件

    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 控制台Handler - INFO级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件Handler - DEBUG级别
    if log_to_file:
        # 确保日志目录存在
        os.makedirs(LOG_DIR, exist_ok=True)

        # 生成日志文件名（包含日期）
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}_{date_str}.log")

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"日志文件: {log_file}")

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    获取已配置的日志器

    Args:
        name: 日志器名称，如果为None则使用默认名称

    Returns:
        logging.Logger: 日志器
    """
    if name is None:
        name = "trading_agent"
    return logging.getLogger(name)
