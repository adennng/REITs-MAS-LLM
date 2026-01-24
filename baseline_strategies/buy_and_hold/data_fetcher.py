#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买入并持有策略 - 数据获取模块
从数据库获取基金列表、价格数据、分红数据等
"""

import os
import sys
import pymysql
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(baseline_dir)
sys.path.insert(0, project_root)

from config.db_config import get_db_reits_config
from baseline_strategies.config import BASELINE_TABLE


def get_fund_date_combinations(
    start_date: str,
    end_date: str,
    specific_fund_codes: List[str] = None,
    min_listed_days: int = 365
) -> List[Tuple[str, str]]:
    """
    获取符合条件的(基金代码, 交易日期)组合

    只纳入在回测第一天(start_date)就满足上市天数要求的基金

    Args:
        start_date: 起始日期（格式：YYYY-MM-DD）
        end_date: 截止日期（格式：YYYY-MM-DD）
        specific_fund_codes: 特定基金代码列表（可选，为None或空列表则处理全部基金）
        min_listed_days: 基金必须上市超过的天数（默认365天）

    Returns:
        List[Tuple[str, str]]: [(基金代码, 交易日期), ...]，按日期和基金代码排序
    """
    # 验证日期格式
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"日期格式错误: {e}")

    if start_dt > end_dt:
        raise ValueError(f"起始日期({start_date})不能晚于截止日期({end_date})")

    # 获取数据库配置
    db_config = get_db_reits_config()

    # 连接数据库
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # 1. 获取基金上市日期
            if specific_fund_codes and len(specific_fund_codes) > 0:
                placeholders = ', '.join(['%s'] * len(specific_fund_codes))
                sql_list_date = f"""
                    SELECT fund_code, list_date
                    FROM product_info
                    WHERE fund_code IN ({placeholders})
                """
                cursor.execute(sql_list_date, specific_fund_codes)
            else:
                sql_list_date = """
                    SELECT fund_code, list_date
                    FROM product_info
                """
                cursor.execute(sql_list_date)

            list_dates = cursor.fetchall()

            # 构建基金代码到上市日期的映射
            fund_list_date_map = {
                row['fund_code']: row['list_date']
                for row in list_dates
            }

            if not fund_list_date_map:
                return []

            # 2. 筛选在回测第一天(start_date)就满足上市天数要求的基金
            eligible_funds = set()

            for fund_code, list_date in fund_list_date_map.items():
                # 跳过没有上市日期的基金
                if list_date is None:
                    continue

                # 确保日期类型一致
                if isinstance(list_date, str):
                    list_date = datetime.strptime(list_date, '%Y-%m-%d').date()

                # 计算在start_date时的上市天数
                listed_days_at_start = (start_dt.date() - list_date).days

                # 只纳入在start_date时就满足上市天数要求的基金
                if listed_days_at_start >= min_listed_days:
                    eligible_funds.add(fund_code)

            if not eligible_funds:
                return []

            # 3. 获取这些符合条件基金的价格数据（在整个回测期间）
            eligible_fund_list = list(eligible_funds)
            placeholders = ', '.join(['%s'] * len(eligible_fund_list))
            sql_price = f"""
                SELECT DISTINCT fund_code, trade_date
                FROM price_data
                WHERE fund_code IN ({placeholders})
                  AND trade_date >= %s
                  AND trade_date <= %s
                ORDER BY trade_date ASC, fund_code ASC
            """
            params = eligible_fund_list + [start_date, end_date]
            cursor.execute(sql_price, params)

            price_records = cursor.fetchall()

            # 4. 查询 baseline_buy_hold 表中已处理过的基金
            # 只要基金有首日记录，说明已经开始处理，应该排除该基金的所有日期
            placeholders = ', '.join(['%s'] * len(eligible_fund_list))
            sql_processed = f"""
                SELECT DISTINCT fund_code
                FROM {BASELINE_TABLE}
                WHERE fund_code IN ({placeholders})
                  AND is_first_day = 1
            """
            cursor.execute(sql_processed, eligible_fund_list)

            processed_funds = cursor.fetchall()

            # 构建已处理基金的集合（用于快速查找）
            processed_fund_set = set()
            for record in processed_funds:
                processed_fund_set.add(record['fund_code'])

            # 5. 筛选未处理的组合
            valid_combinations = []

            for record in price_records:
                fund_code = record['fund_code']
                trade_date = record['trade_date']

                # 检查该基金是否已经被处理过
                if fund_code in processed_fund_set:
                    # 该基金已处理过，跳过
                    continue

                # 将日期转换为字符串格式
                if isinstance(trade_date, str):
                    trade_date_str = trade_date
                else:
                    trade_date_str = trade_date.strftime('%Y-%m-%d')

                valid_combinations.append((fund_code, trade_date_str))

            return valid_combinations

    finally:
        connection.close()


def get_fund_trading_dates(
    fund_code: str,
    start_date: str,
    end_date: str
) -> List[str]:
    """
    获取指定基金在日期范围内的所有交易日

    Args:
        fund_code: 基金代码
        start_date: 起始日期
        end_date: 终止日期

    Returns:
        List[str]: 交易日列表
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT DISTINCT trade_date
                FROM price_data
                WHERE fund_code = %s
                  AND trade_date >= %s
                  AND trade_date <= %s
                ORDER BY trade_date
            """
            cursor.execute(sql, (fund_code, start_date, end_date))
            results = cursor.fetchall()

            dates = []
            for r in results:
                d = r['trade_date']
                if isinstance(d, datetime):
                    dates.append(d.strftime('%Y-%m-%d'))
                else:
                    dates.append(str(d))
            return dates

    finally:
        connection.close()


def get_price(fund_code: str, trade_date: str) -> Optional[float]:
    """
    获取指定日期的收盘价

    Args:
        fund_code: 基金代码
        trade_date: 交易日期

    Returns:
        float: 收盘价，如果不存在返回None
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT close
                FROM price_data
                WHERE fund_code = %s AND trade_date = %s
            """
            cursor.execute(sql, (fund_code, trade_date))
            result = cursor.fetchone()

            if result:
                return float(result['close'])
            return None

    finally:
        connection.close()


def get_dividend(fund_code: str, ex_dividend_date: str) -> Optional[float]:
    """
    获取指定除息日的分红金额

    Args:
        fund_code: 基金代码
        ex_dividend_date: 除息日

    Returns:
        float: 每份分红金额，如果没有分红返回None
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT dividend_per_share
                FROM dividend
                WHERE fund_code = %s AND ex_dividend_date = %s
            """
            cursor.execute(sql, (fund_code, ex_dividend_date))
            result = cursor.fetchone()

            if result:
                return float(result['dividend_per_share'])
            return None

    finally:
        connection.close()


def get_existing_records(fund_code: str, start_date: str) -> List[str]:
    """
    获取已有记录的日期列表（用于断点续传）

    Args:
        fund_code: 基金代码
        start_date: 开始日期

    Returns:
        List[str]: 已有记录的日期列表
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = f"""
                SELECT analysis_date
                FROM {BASELINE_TABLE}
                WHERE fund_code = %s
                  AND is_first_day = 1
                  AND analysis_date = %s
            """
            cursor.execute(sql, (fund_code, start_date))
            result = cursor.fetchone()

            if result:
                # 如果首日记录存在，获取所有日期
                sql_all = f"""
                    SELECT DISTINCT analysis_date
                    FROM {BASELINE_TABLE}
                    WHERE fund_code = %s
                    ORDER BY analysis_date
                """
                cursor.execute(sql_all, (fund_code,))
                results = cursor.fetchall()

                dates = []
                for r in results:
                    d = r['analysis_date']
                    if isinstance(d, datetime):
                        dates.append(d.strftime('%Y-%m-%d'))
                    else:
                        dates.append(str(d))
                return dates
            else:
                return []

    finally:
        connection.close()


def get_last_record(fund_code: str) -> Optional[dict]:
    """
    获取最后一条记录（用于恢复账户状态）

    Args:
        fund_code: 基金代码

    Returns:
        dict: 最后一条记录，如果不存在返回None
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = f"""
                SELECT *
                FROM {BASELINE_TABLE}
                WHERE fund_code = %s
                ORDER BY analysis_date DESC
                LIMIT 1
            """
            cursor.execute(sql, (fund_code,))
            result = cursor.fetchone()

            return result

    finally:
        connection.close()
