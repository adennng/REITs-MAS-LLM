#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库写入模块
将预测结果写入数据库
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
import pymysql

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from config.db_config import get_db_reits_config  # noqa: E402

# 获取logger
logger = logging.getLogger(__name__)


def ensure_table_exists(connection):
    """
    确保price_predictions表存在，如果不存在则创建

    Args:
        connection: 数据库连接
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS price_predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        analysis_date DATE NOT NULL COMMENT '分析日期',
        fund_code VARCHAR(20) NOT NULL COMMENT '基金代码',
        direction_prediction TEXT NOT NULL COMMENT '方向预测JSON',
        direction_prediction_cot TEXT COMMENT 'LLM推理过程(Chain of Thought)',
        direction_prediction_input TEXT COMMENT '传递给LLM的完整输入(prompt)',
        volatility_threshold DECIMAL(10, 6) COMMENT '动态波动阈值（来自价格动量专家）',
        output_announcement TEXT COMMENT '公告专家输出JSON',
        output_market TEXT COMMENT '市场专家输出JSON',
        output_price TEXT COMMENT '价格动量专家输出JSON',
        output_event TEXT COMMENT '事件专家输出JSON',
        price_context TEXT COMMENT '价格与交易上下文JSON',
        t1_prob_up DECIMAL(5,4) COMMENT 'T+1上涨概率',
        t1_prob_down DECIMAL(5,4) COMMENT 'T+1下跌概率',
        t1_prob_side DECIMAL(5,4) COMMENT 'T+1横盘概率',
        t5_prob_up DECIMAL(5,4) COMMENT 'T+5上涨概率',
        t5_prob_down DECIMAL(5,4) COMMENT 'T+5下跌概率',
        t5_prob_side DECIMAL(5,4) COMMENT 'T+5横盘概率',
        t20_prob_up DECIMAL(5,4) COMMENT 'T+20上涨概率',
        t20_prob_down DECIMAL(5,4) COMMENT 'T+20下跌概率',
        t20_prob_side DECIMAL(5,4) COMMENT 'T+20横盘概率',
        UNIQUE KEY unique_fund_date (fund_code, analysis_date),
        INDEX idx_analysis_date (analysis_date),
        INDEX idx_fund_code (fund_code)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='方向预测结果表';
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(create_table_sql)
            connection.commit()
            logger.debug("price_predictions表检查/创建完成")
    except Exception as e:
        logger.error(f"创建price_predictions表失败: {e}")
        raise


def write_prediction_to_db(
    fund_code: str,
    analysis_date: str,
    prediction: Dict[str, Any],
    reasoning_content: str = "",
    llm_input: str = "",
    volatility_threshold: float = None,
    output_announcement: Dict[str, Any] = None,
    output_market: Dict[str, Any] = None,
    output_price: Dict[str, Any] = None,
    output_event: Dict[str, Any] = None,
    price_context: Dict[str, Any] = None,
    t1_prob_up: float = None,
    t1_prob_down: float = None,
    t1_prob_side: float = None,
    t5_prob_up: float = None,
    t5_prob_down: float = None,
    t5_prob_side: float = None,
    t20_prob_up: float = None,
    t20_prob_down: float = None,
    t20_prob_side: float = None
) -> bool:
    """
    将预测结果写入数据库

    Args:
        fund_code: 基金代码
        analysis_date: 分析日期（格式：YYYY-MM-DD）
        prediction: 预测结果JSON（dict格式）
        reasoning_content: LLM推理过程（Chain of Thought）
        llm_input: 传递给LLM的完整输入（prompt）
        volatility_threshold: 动态波动阈值（来自价格动量专家）
        output_announcement: 公告专家输出（dict格式）
        output_market: 市场专家输出（dict格式）
        output_price: 价格动量专家输出（dict格式）
        output_event: 事件专家输出（dict格式）
        price_context: 价格与交易上下文（dict格式）
        t1_prob_up: T+1上涨概率
        t1_prob_down: T+1下跌概率
        t1_prob_side: T+1横盘概率
        t5_prob_up: T+5上涨概率
        t5_prob_down: T+5下跌概率
        t5_prob_side: T+5横盘概率
        t20_prob_up: T+20上涨概率
        t20_prob_down: T+20下跌概率
        t20_prob_side: T+20横盘概率

    Returns:
        bool: 是否写入成功
    """
    logger.info(f"准备写入数据库 - 基金: {fund_code}, 日期: {analysis_date}")

    # 获取数据库配置
    db_config = get_db_reits_config()

    # 连接数据库
    connection = pymysql.connect(**db_config)

    try:
        # 确保表存在
        ensure_table_exists(connection)

        # 将预测结果转换为JSON字符串
        prediction_json_str = json.dumps(prediction, ensure_ascii=False)

        # 将四个专家的输出转换为JSON字符串
        output_announcement_str = json.dumps(output_announcement, ensure_ascii=False) if output_announcement else None
        output_market_str = json.dumps(output_market, ensure_ascii=False) if output_market else None
        output_price_str = json.dumps(output_price, ensure_ascii=False) if output_price else None
        output_event_str = json.dumps(output_event, ensure_ascii=False) if output_event else None
        price_context_str = json.dumps(price_context, ensure_ascii=False) if price_context else None

        with connection.cursor() as cursor:
            # 使用INSERT ... ON DUPLICATE KEY UPDATE实现插入或更新
            sql = """
                INSERT INTO price_predictions
                    (analysis_date, fund_code, direction_prediction, direction_prediction_cot,
                     direction_prediction_input, volatility_threshold,
                     output_announcement, output_market, output_price, output_event, price_context,
                     t1_prob_up, t1_prob_down, t1_prob_side,
                     t5_prob_up, t5_prob_down, t5_prob_side,
                     t20_prob_up, t20_prob_down, t20_prob_side)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    direction_prediction = VALUES(direction_prediction),
                    direction_prediction_cot = VALUES(direction_prediction_cot),
                    direction_prediction_input = VALUES(direction_prediction_input),
                    volatility_threshold = VALUES(volatility_threshold),
                    output_announcement = VALUES(output_announcement),
                    output_market = VALUES(output_market),
                    output_price = VALUES(output_price),
                    output_event = VALUES(output_event),
                    price_context = VALUES(price_context),
                    t1_prob_up = VALUES(t1_prob_up),
                    t1_prob_down = VALUES(t1_prob_down),
                    t1_prob_side = VALUES(t1_prob_side),
                    t5_prob_up = VALUES(t5_prob_up),
                    t5_prob_down = VALUES(t5_prob_down),
                    t5_prob_side = VALUES(t5_prob_side),
                    t20_prob_up = VALUES(t20_prob_up),
                    t20_prob_down = VALUES(t20_prob_down),
                    t20_prob_side = VALUES(t20_prob_side)
            """

            cursor.execute(sql, (
                analysis_date, fund_code, prediction_json_str, reasoning_content, llm_input,
                volatility_threshold,
                output_announcement_str, output_market_str, output_price_str,
                output_event_str, price_context_str,
                t1_prob_up, t1_prob_down, t1_prob_side,
                t5_prob_up, t5_prob_down, t5_prob_side,
                t20_prob_up, t20_prob_down, t20_prob_side
            ))
            connection.commit()

            # 获取影响的行数
            rows_affected = cursor.rowcount

            if rows_affected > 0:
                if rows_affected == 1:
                    logger.info(f"✓ 新增记录成功 - 基金: {fund_code}, 日期: {analysis_date}")
                else:
                    logger.info(f"✓ 更新记录成功 - 基金: {fund_code}, 日期: {analysis_date}")
                return True
            else:
                logger.warning(f"✗ 写入数据库无影响 - 基金: {fund_code}, 日期: {analysis_date}")
                return False

    except Exception as e:
        logger.error(f"✗ 写入数据库失败 - 基金: {fund_code}, 日期: {analysis_date}, 错误: {e}")
        return False

    finally:
        connection.close()


def read_prediction_from_db(
    fund_code: str,
    analysis_date: str
) -> Dict[str, Any]:
    """
    从数据库读取预测结果（用于测试）

    Args:
        fund_code: 基金代码
        analysis_date: 分析日期

    Returns:
        Dict: 预测结果，如果不存在返回None
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT analysis_date, fund_code, direction_prediction,
                       direction_prediction_cot, direction_prediction_input
                FROM price_predictions
                WHERE fund_code = %s AND analysis_date = %s
            """
            cursor.execute(sql, (fund_code, analysis_date))
            result = cursor.fetchone()

            if result:
                # 解析direction_prediction JSON字符串
                result['direction_prediction'] = json.loads(result['direction_prediction'])

                # 转换日期为字符串
                if result['analysis_date']:
                    result['analysis_date'] = result['analysis_date'].strftime('%Y-%m-%d')

                return result
            return None

    finally:
        connection.close()
