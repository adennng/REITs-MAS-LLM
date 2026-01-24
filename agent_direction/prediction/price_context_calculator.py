#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价格上下文计算模块
计算方向预测所需的价格与交易上下文数据
"""

import sys
import os
from typing import Dict, Any, Optional, List
import pymysql
import pandas as pd
from datetime import datetime, timedelta
import math

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from config.db_config import get_db_reits_config  # noqa: E402
from agent_direction import config  # noqa: E402


# ============================================================================
# 盘整状态计算函数（独立于 agent_trading）
# ============================================================================

def calculate_daily_regime(
    price_change_1d: float,
    volatility_threshold: float
) -> str:
    """
    计算当日盘整/突破状态

    Args:
        price_change_1d: 当日涨跌幅
        volatility_threshold: 动态波动阈值

    Returns:
        str: 状态标识（未突破/上涨突破/下跌突破）
    """
    if volatility_threshold == 0:
        return "未突破"

    # 判断是否突破阈值
    if abs(price_change_1d) < volatility_threshold:
        return "未突破"
    elif price_change_1d >= volatility_threshold:
        return "上涨突破"
    else:
        return "下跌突破"


def calculate_5d_regime(
    price_change_5d: float,
    volatility_threshold: float,
    sqrt_multiplier: float = math.sqrt(5)
) -> str:
    """
    计算最近5日盘整/突破状态

    Args:
        price_change_5d: 最近5日涨跌幅
        volatility_threshold: 单日动态波动阈值
        sqrt_multiplier: 5日阈值系数（默认为√5）

    Returns:
        str: 状态标识（未突破/上涨突破/下跌突破）
    """
    if volatility_threshold == 0:
        return "未突破"

    # 计算5日阈值 = √5 × 单日阈值
    threshold_5d = sqrt_multiplier * volatility_threshold

    # 判断是否突破阈值
    if abs(price_change_5d) < threshold_5d:
        return "未突破"
    elif price_change_5d >= threshold_5d:
        return "上涨突破"
    else:
        return "下跌突破"


def calculate_20d_regime(
    price_change_20d: float,
    volatility_threshold: float,
    sqrt_multiplier: float = math.sqrt(20)
) -> str:
    """
    计算最近20日盘整/突破状态

    Args:
        price_change_20d: 最近20日涨跌幅
        volatility_threshold: 单日动态波动阈值
        sqrt_multiplier: 20日阈值系数（默认为√20）

    Returns:
        str: 状态标识（未突破/上涨突破/下跌突破）
    """
    if volatility_threshold == 0:
        return "未突破"

    # 计算20日阈值 = √20 × 单日阈值
    threshold_20d = sqrt_multiplier * volatility_threshold

    # 判断是否突破阈值
    if abs(price_change_20d) < threshold_20d:
        return "未突破"
    elif price_change_20d >= threshold_20d:
        return "上涨突破"
    else:
        return "下跌突破"



def get_fund_price_data(fund_code: str, end_date: str, days: int = 150) -> pd.DataFrame:
    """
    获取指定基金的历史价格数据（用于计算动态阈值）

    Args:
        fund_code: 基金代码
        end_date: 截止日期 (格式: YYYY-MM-DD)
        days: 往前获取的自然日天数

    Returns:
        DataFrame: 包含 trade_date, close, vol 的数据框，按日期升序排列
    """
    config = get_db_reits_config()
    conn = pymysql.connect(**config)

    try:
        # 计算起始日期
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days)
        start_date = start_dt.strftime('%Y-%m-%d')

        sql = """
            SELECT trade_date, close, vol
            FROM price_data
            WHERE fund_code = %s
              AND trade_date >= %s
              AND trade_date <= %s
            ORDER BY trade_date ASC
        """

        df = pd.read_sql(sql, conn, params=(fund_code, start_date, end_date))
        return df

    finally:
        conn.close()


def calculate_price_context(
    fund_code: str,
    current_date: str,
    volatility_threshold: float,
    num_days: int = 20
) -> Dict[str, Any]:
    """
    计算价格与交易上下文

    根据技术方案要求，计算：
    - 最近20个交易日收盘价列表
    - 最近20个交易日日期列表
    - 最新价格
    - 1日、5日、20日涨跌幅

    Args:
        fund_code: 基金代码
        current_date: 当前日期（格式：YYYY-MM-DD）
        volatility_threshold: 动态波动阈值（用于构建提示词，不存储在返回数据中）
        num_days: 获取最近N个交易日（默认20个）

    Returns:
        Dict: 包含价格上下文信息的字典（英文字段）
        {
            "recent_close": [最近20个交易日收盘价列表],
            "recent_dates": [最近20个交易日日期列表],
            "latest_price": 最新价格,
            "price_change_1d": 1日涨跌幅,
            "price_change_5d": 5日涨跌幅,
            "price_change_20d": 20日涨跌幅
        }
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # 查询最近21个交易日的收盘价（20个用于展示，1个额外用于计算第1天的涨跌幅）
            sql = """
                SELECT trade_date, close
                FROM price_data
                WHERE fund_code = %s
                  AND trade_date <= %s
                ORDER BY trade_date DESC
                LIMIT %s
            """
            # 查询21个交易日，确保能计算20日涨跌幅
            query_limit = 21
            cursor.execute(sql, (fund_code, current_date, query_limit))
            results = cursor.fetchall()

            if not results:
                raise ValueError(f"未找到基金 {fund_code} 在 {current_date} 及之前的价格数据")

            # 结果是倒序的（最新的在前），需要反转为正序（从旧到新）
            results_asc = list(reversed(results))

            # 提取收盘价列表和日期列表
            all_prices = [float(row['close']) for row in results_asc]
            all_dates = [row['trade_date'].strftime('%Y-%m-%d') for row in results_asc]

            # 取最近20个交易日的收盘价和日期
            recent_close = all_prices[-20:] if len(all_prices) >= 20 else all_prices
            recent_dates = all_dates[-20:] if len(all_dates) >= 20 else all_dates

            # 最新价格（列表中的最后一个）
            latest_price = all_prices[-1]

            # 计算1日涨跌幅
            price_change_1d = None
            if len(all_prices) >= 2:
                prev_1d_price = all_prices[-2]
                price_change_1d = (latest_price - prev_1d_price) / prev_1d_price

            # 计算5日涨跌幅
            price_change_5d = None
            if len(all_prices) >= 6:
                prev_5d_price = all_prices[-6]
                price_change_5d = (latest_price - prev_5d_price) / prev_5d_price

            # 计算20日涨跌幅（基于第21个交易日的价格）
            price_change_20d = None
            if len(all_prices) >= 21:
                prev_20d_price = all_prices[-21]  # 第21个交易日的价格（即20日前）
                price_change_20d = (latest_price - prev_20d_price) / prev_20d_price

            # 构建返回数据（英文字段）
            price_context = {
                "recent_close": recent_close,
                "recent_dates": recent_dates,
                "latest_price": latest_price,
                "price_change_1d": price_change_1d,
                "price_change_5d": price_change_5d,
                "price_change_20d": price_change_20d
            }

            return price_context

    finally:
        connection.close()


def format_price_context_for_llm(price_context: Dict[str, Any]) -> str:
    """
    ��价格上下文格式化为适合LLM输入的字符串

    Args:
        price_context: 价格上下文数据

    Returns:
        str: 格式化的字符串
    """
    lines = []

    lines.append("价格与交易上下文：")
    lines.append(f"- 最新价格: {price_context['latest_price']:.3f}元")

    if price_context.get('price_change_1d') is not None:
        change_1d_pct = price_context['price_change_1d'] * 100
        lines.append(f"- 1日涨跌幅: {change_1d_pct:+.2f}%")

    if price_context.get('price_change_5d') is not None:
        change_5d_pct = price_context['price_change_5d'] * 100
        lines.append(f"- 5日涨跌幅: {change_5d_pct:+.2f}%")

    if price_context.get('price_change_20d') is not None:
        change_20d_pct = price_context['price_change_20d'] * 100
        lines.append(f"- 20日涨跌幅: {change_20d_pct:+.2f}%")

    lines.append(f"- 最近{len(price_context['recent_close'])}个交易日收盘价（从旧到新）:")
    lines.append(f"  {price_context['recent_close']}")

    return "\n".join(lines)
