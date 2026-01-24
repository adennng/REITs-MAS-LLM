#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买入并持有策略 - 策略执行模块
"""

import os
import sys
import logging
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(baseline_dir)
sys.path.insert(0, project_root)

from baseline_strategies.buy_and_hold.account import BuyAndHoldAccount
from baseline_strategies.buy_and_hold.data_fetcher import (
    get_fund_trading_dates,
    get_price,
    get_dividend,
    get_existing_records,
    get_last_record
)
from baseline_strategies.buy_and_hold.db_manager import save_record

logger = logging.getLogger('BuyAndHold')


def run_single_fund(
    fund_code: str,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """
    对单个基金运行买入并持有策略

    Args:
        fund_code: 基金代码
        start_date: 开始日期（首次买入日）
        end_date: 终止日期

    Returns:
        Dict: 策略执行结果
    """
    logger.info(f"=" * 80)
    logger.info(f"开始处理基金: {fund_code}")
    logger.info(f"开始日期: {start_date}")
    logger.info(f"终止日期: {end_date}")
    logger.info(f"=" * 80)

    # 获取该基金在日期范围内的所有交易日
    all_trading_dates = get_fund_trading_dates(fund_code, start_date, end_date)

    if not all_trading_dates:
        logger.warning(f"基金 {fund_code} 在指定日期范围内无交易数据")
        return {'error': 'no_data'}

    # 检查是否有已处理的记录（断点续传）
    existing_dates = get_existing_records(fund_code, start_date)
    last_record = None

    if existing_dates:
        logger.info(f"发现 {len(existing_dates)} 条已有记录")
        # 获取最后一条记录用于恢复账户状态
        last_record = get_last_record(fund_code)

    # 计算待处理的交易日
    pending_dates = [d for d in all_trading_dates if d not in existing_dates]

    logger.info(f"总交易日: {len(all_trading_dates)}")
    logger.info(f"已处理: {len(existing_dates)}")
    logger.info(f"待处理: {len(pending_dates)}")

    if not pending_dates:
        logger.info(f"基金 {fund_code} 所有交易日已处理完毕")
        # 返回已有结果
        if last_record:
            return {
                'fund_code': fund_code,
                'start_date': all_trading_dates[0],
                'end_date': all_trading_dates[-1],
                'trading_days': len(all_trading_dates),
                'final_nav': float(last_record['nav']),
                'total_return': float(last_record['total_return']),
                'status': 'completed'
            }
        else:
            return {'error': 'no_record'}

    # 初始化或恢复账户
    if last_record is None:
        # 全新开始
        logger.info("初始化新账户")
        account = BuyAndHoldAccount()
        is_first_day = True
    else:
        # 从最后一条记录恢复
        logger.info(f"从记录恢复账户状态: {last_record['analysis_date']}")
        account = BuyAndHoldAccount()
        account.cash = float(last_record['cash'])
        account.shares = float(last_record['shares'])
        account.peak_nav = float(last_record['peak_nav'])
        account.max_drawdown_value = float(last_record['max_drawdown'])
        account.holding_days = int(last_record['holding_days'])
        account.total_dividend_received = float(last_record['total_dividend_received'])
        account.buy_shares = float(last_record['buy_shares']) if last_record['buy_shares'] else 0.0
        account.buy_amount = float(last_record['buy_amount']) if last_record['buy_amount'] else 0.0
        account.buy_cost = float(last_record['buy_cost']) if last_record['buy_cost'] else 0.0
        is_first_day = False

    # 遍历待处理的交易日
    success_count = len(existing_dates)
    fail_count = 0

    for date_str in pending_dates:
        logger.info(f"\n处理 {date_str}...")

        # 获取当日价格
        current_price = get_price(fund_code, date_str)
        if current_price is None:
            logger.warning(f"  未找到价格数据，跳过")
            fail_count += 1
            continue

        # 更新价格
        account.update_price(current_price)

        # 检查是否有分红
        dividend_per_share = get_dividend(fund_code, date_str)
        dividend_today = 0.0
        if dividend_per_share:
            dividend_today = account.process_dividend(dividend_per_share)
            logger.info(f"  收到分红: {dividend_today:.2f} 元")

        # 如果是首日，执行买入
        if is_first_day:
            trade_info = account.execute_buy(current_price)
            logger.info(f"  首日全仓买入:")
            logger.info(f"    买入份额: {trade_info['buy_shares']:.4f}")
            logger.info(f"    买入金额: {trade_info['buy_amount']:.2f}")
            logger.info(f"    手续费: {trade_info['buy_cost']:.2f}")

        # 构建记录
        record = {
            'fund_code': fund_code,
            'analysis_date': date_str,
            'holding_days': account.holding_days,
            'current_price': current_price,
            'peak_nav': account.peak_nav,
            'shares': account.shares,
            'cash': account.cash,
            'market_value': account.market_value,
            'nav': account.nav,
            'max_drawdown': account.max_drawdown,
            'total_return': account.total_return,
            'total_dividend_received': account.total_dividend_received,
            'dividend_today': dividend_today,
            'initial_capital': account.initial_capital,
            'buy_shares': account.buy_shares if is_first_day else None,
            'buy_amount': account.buy_amount if is_first_day else None,
            'buy_cost': account.buy_cost if is_first_day else None,
            'is_first_day': 1 if is_first_day else 0,
            'strategy_type': 'buy_and_hold'
        }

        # 保存到数据库
        try:
            save_record(record)
            success_count += 1
            logger.info(f"  ✓ 保存成功")

            if is_first_day:
                is_first_day = False

        except Exception as e:
            logger.error(f"  ✗ 保存失败: {e}")
            fail_count += 1

    # 计算最终结果
    final_nav = account.nav
    total_return = account.total_return

    result = {
        'fund_code': fund_code,
        'start_date': all_trading_dates[0] if all_trading_dates else None,
        'end_date': all_trading_dates[-1] if all_trading_dates else None,
        'trading_days': len(all_trading_dates),
        'existing_days': len(existing_dates),
        'processed_days': len(pending_dates),
        'success_count': success_count,
        'fail_count': fail_count,
        'final_nav': final_nav,
        'total_return': total_return,
        'total_dividend': account.total_dividend_received,
        'status': 'completed'
    }

    logger.info(f"\n基金 {fund_code} 策略执行完成:")
    logger.info(f"  总收益率: {total_return:.2%}")
    logger.info(f"  最终净值: {final_nav:.2f}")
    logger.info(f"  累计分红: {account.total_dividend_received:.2f}")
    logger.info(f"  成功/失败: {success_count}/{fail_count}")

    return result
