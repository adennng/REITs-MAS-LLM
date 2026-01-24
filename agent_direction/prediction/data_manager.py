#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理模块
负责获取符合条件的(基金代码+交易日期)组合
"""

import sys
import os
from datetime import datetime
from typing import List, Tuple
import pymysql

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from config.db_config import get_db_reits_config  # noqa: E402


def get_fund_date_combinations(
    start_date: str,
    end_date: str,
    specific_fund_codes: List[str] = None,
    min_listed_days: int = 365
) -> List[Tuple[str, str]]:
    """
    获取符合条件的(基金代码, 交易日期)组合

    Args:
        start_date: 起始日期（格式：YYYY-MM-DD）
        end_date: 截止日期（格式：YYYY-MM-DD）
        specific_fund_codes: 特定基金代码列表（可选，为None或空列表则处理全部基金）
        min_listed_days: 基金必须上市超过的天数（默认365天）

    Returns:
        List[Tuple[str, str]]: [(基金代码, 交易日期), ...]，按日期和基金代码排序

    注意：
        1. 会自动过滤掉 price_predictions 表中已存在的组合（已处理过的数据）
        2. 采用动态基金池：每个交易日独立检查基金是否满足上市天数要求，
           一旦某个(基金代码+交易日期)组合在该交易日满足条件就会被纳入，
           随着时间推移，新上市的基金会逐渐被纳入处理范围
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
                # 如果指定了特定基金，只查询这些基金的上市日期
                placeholders = ', '.join(['%s'] * len(specific_fund_codes))
                sql_list_date = f"""
                    SELECT fund_code, list_date
                    FROM product_info
                    WHERE fund_code IN ({placeholders})
                """
                cursor.execute(sql_list_date, specific_fund_codes)
            else:
                # 否则查询所有基金的上市日期
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

            # 2. 获取价格数据（基金代码+交易日期组合）
            if specific_fund_codes and len(specific_fund_codes) > 0:
                # 如果指定了特定基金
                placeholders = ', '.join(['%s'] * len(specific_fund_codes))
                sql_price = f"""
                    SELECT DISTINCT fund_code, trade_date
                    FROM price_data
                    WHERE fund_code IN ({placeholders})
                      AND trade_date >= %s
                      AND trade_date <= %s
                    ORDER BY trade_date ASC, fund_code ASC
                """
                params = list(specific_fund_codes) + [start_date, end_date]
                cursor.execute(sql_price, params)
            else:
                # 否则查询所有基金
                sql_price = """
                    SELECT DISTINCT fund_code, trade_date
                    FROM price_data
                    WHERE trade_date >= %s
                      AND trade_date <= %s
                    ORDER BY trade_date ASC, fund_code ASC
                """
                cursor.execute(sql_price, (start_date, end_date))

            price_records = cursor.fetchall()

            # 2.5. 查询 price_predictions 表中已处理过的组合
            if specific_fund_codes and len(specific_fund_codes) > 0:
                # 如果指定了特定基金
                placeholders = ', '.join(['%s'] * len(specific_fund_codes))
                sql_processed = f"""
                    SELECT DISTINCT fund_code, analysis_date
                    FROM price_predictions
                    WHERE fund_code IN ({placeholders})
                      AND analysis_date >= %s
                      AND analysis_date <= %s
                """
                params_processed = list(specific_fund_codes) + [start_date, end_date]
                cursor.execute(sql_processed, params_processed)
            else:
                # 否则查询所有基金
                sql_processed = """
                    SELECT DISTINCT fund_code, analysis_date
                    FROM price_predictions
                    WHERE analysis_date >= %s
                      AND analysis_date <= %s
                """
                cursor.execute(sql_processed, (start_date, end_date))

            processed_records = cursor.fetchall()

            # 构建已处理组合的集合（用于快速查找）
            processed_set = set()
            for record in processed_records:
                fund_code = record['fund_code']
                analysis_date = record['analysis_date']
                # 确保日期格式一致
                if isinstance(analysis_date, str):
                    processed_set.add((fund_code, analysis_date))
                else:
                    # 如果是 date 对象，转换为字符串
                    processed_set.add((fund_code, analysis_date.strftime('%Y-%m-%d')))

            # 3. 筛选符合条件的组合（动态检查每个交易日的上市天数）
            valid_combinations = []

            for record in price_records:
                fund_code = record['fund_code']
                trade_date = record['trade_date']

                # 将日期转换为字符串格式和日期对象
                if isinstance(trade_date, str):
                    trade_date_str = trade_date
                    trade_date_obj = datetime.strptime(trade_date, '%Y-%m-%d').date()
                else:
                    trade_date_str = trade_date.strftime('%Y-%m-%d')
                    trade_date_obj = trade_date

                # 检查该组合是否已经在 price_predictions 表中（已处理过）
                if (fund_code, trade_date_str) in processed_set:
                    # 该组合已处理过，跳过
                    continue

                # 获取基金的上市日期
                list_date = fund_list_date_map.get(fund_code)

                # 跳过上市日期为空的基金
                if list_date is None:
                    continue

                # 确保日期类型一致
                if isinstance(list_date, str):
                    list_date_obj = datetime.strptime(list_date, '%Y-%m-%d').date()
                else:
                    list_date_obj = list_date

                # 计算该基金在当前交易日的上市天数
                listed_days = (trade_date_obj - list_date_obj).days

                # 只有在该交易日满足上市天数要求的组合才纳入
                if listed_days >= min_listed_days:
                    valid_combinations.append((fund_code, trade_date_str))

            return valid_combinations

    finally:
        connection.close()


def get_fund_info(fund_code: str) -> dict:
    """
    获取基金的基本信息

    Args:
        fund_code: 基金代码

    Returns:
        dict: 基金信息 {'fund_code': '...', 'fund_name': '...', 'list_date': '...'}
              如果未找到，返回 None
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT fund_code, fund_name, list_date
                FROM product_info
                WHERE fund_code = %s
            """
            cursor.execute(sql, (fund_code,))
            result = cursor.fetchone()

            if result:
                # 将日期转换为字符串
                if result['list_date']:
                    result['list_date'] = result['list_date'].strftime('%Y-%m-%d')
                return result
            return None

    finally:
        connection.close()
