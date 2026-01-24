#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策专家回测入口脚本
使用高级推理LLM (deepseek-reasoner) 进行回测
"""

import os
import sys
import argparse

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agent_trading.config import (
    START_DATE,
    END_DATE,
    FUND_CODES,
    INITIAL_CAPITAL
)
from agent_trading.utils.logger import setup_logger
from agent_trading.backtest.backtester import Backtester


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='决策专家回测 - 使用高级推理LLM'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=START_DATE,
        help=f'起始日期 (默认: {START_DATE})'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=END_DATE,
        help=f'终止日期 (默认: {END_DATE})'
    )
    parser.add_argument(
        '--fund-codes',
        type=str,
        nargs='+',
        default=FUND_CODES if FUND_CODES else None,
        help='基金代码列表 (默认: 全部基金)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=INITIAL_CAPITAL,
        help=f'初始资金 (默认: {INITIAL_CAPITAL})'
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logger()

    logger.info("=" * 80)
    logger.info("决策专家回测 - 高级推理LLM模式")
    logger.info("=" * 80)

    # 创建回测器
    backtester = Backtester(
        start_date=args.start_date,
        end_date=args.end_date,
        fund_codes=args.fund_codes,
        initial_capital=args.initial_capital
    )

    # 运行回测
    try:
        backtester.run()
    except KeyboardInterrupt:
        logger.info("\n回测被用户中断")
    except Exception as e:
        logger.error(f"回测出错: {e}")
        raise

    logger.info("\n回测结束")


if __name__ == '__main__':
    main()
