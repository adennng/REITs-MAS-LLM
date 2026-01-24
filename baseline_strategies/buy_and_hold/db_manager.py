#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买入并持有策略 - 数据库管理模块
负责将策略执行结果写入数据库
"""

import os
import sys
import pymysql
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(baseline_dir)
sys.path.insert(0, project_root)

from config.db_config import get_db_reits_config
from baseline_strategies.config import BASELINE_TABLE


def save_record(record: Dict[str, Any]) -> int:
    """
    保存记录到数据库

    Args:
        record: 记录数据字典

    Returns:
        int: 插入记录的ID
    """
    db_config = get_db_reits_config()
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor() as cursor:
            sql = f"""
                INSERT INTO {BASELINE_TABLE} (
                    `fund_code`, `analysis_date`, `holding_days`,
                    `current_price`, `peak_nav`,
                    `shares`, `cash`, `market_value`, `nav`,
                    `max_drawdown`, `total_return`,
                    `total_dividend_received`, `dividend_today`,
                    `initial_capital`, `buy_shares`, `buy_amount`, `buy_cost`,
                    `is_first_day`, `strategy_type`
                ) VALUES (
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s
                )
            """

            params = (
                record['fund_code'],
                record['analysis_date'],
                record['holding_days'],
                record['current_price'],
                record['peak_nav'],
                record['shares'],
                record['cash'],
                record['market_value'],
                record['nav'],
                record['max_drawdown'],
                record['total_return'],
                record['total_dividend_received'],
                record['dividend_today'],
                record['initial_capital'],
                record.get('buy_shares'),
                record.get('buy_amount'),
                record.get('buy_cost'),
                record['is_first_day'],
                record.get('strategy_type', 'buy_and_hold')
            )

            cursor.execute(sql, params)
            connection.commit()
            return cursor.lastrowid

    finally:
        connection.close()
