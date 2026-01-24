#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买入并持有基线策略 - 主执行入口
"""

import os
import sys
import logging
import time
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from baseline_strategies.config import (
    START_DATE,
    END_DATE,
    SPECIFIC_FUND_CODES,
    MIN_LISTED_DAYS,
    LOG_DIR,
    LOG_FILE_PREFIX
)
from baseline_strategies.buy_and_hold.data_fetcher import get_fund_date_combinations
from baseline_strategies.buy_and_hold.strategy import run_single_fund


def setup_logger() -> logging.Logger:
    """
    配置日志记录器

    Returns:
        Logger: 配置好的日志记录器
    """
    logger = logging.getLogger('BuyAndHold')
    logger.setLevel(logging.INFO)

    # 清除已有的处理器
    logger.handlers.clear()

    # 确定日志目录
    os.makedirs(LOG_DIR, exist_ok=True)

    # 日志文件路径（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{LOG_FILE_PREFIX}_{timestamp}.log'
    log_path = os.path.join(LOG_DIR, log_file)

    # 文件处理器
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_path}")

    return logger


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger()

    logger.info("=" * 100)
    logger.info("买入并持有基线策略 - 开始执行")
    logger.info("=" * 100)

    # 显示配置参数
    logger.info("配置参数:")
    logger.info(f"  起始日期: {START_DATE}")
    logger.info(f"  截止日期: {END_DATE}")
    logger.info(f"  特定基金代码: {SPECIFIC_FUND_CODES if SPECIFIC_FUND_CODES else '全部基金'}")
    logger.info(f"  基金上市时间要求: ≥{MIN_LISTED_DAYS}天")
    logger.info("=" * 100)

    # Step 1: 获取所有符合条件的(基金代码, 交易日期)组合
    logger.info("Step 1: 获取符合条件的(基金代码, 交易日期)组合")

    try:
        combinations = get_fund_date_combinations(
            start_date=START_DATE,
            end_date=END_DATE,
            specific_fund_codes=SPECIFIC_FUND_CODES if SPECIFIC_FUND_CODES else None,
            min_listed_days=MIN_LISTED_DAYS
        )

        logger.info(f"✓ 共找到 {len(combinations)} 个符合条件的组合")

        if len(combinations) == 0:
            logger.warning("没有找到符合条件的组合，程序退出")
            return

        # 显示前几个组合
        if len(combinations) <= 5:
            logger.info(f"所有组合: {combinations}")
        else:
            logger.info(f"前5个组合: {combinations[:5]}")
            logger.info(f"后5个组合: {combinations[-5:]}")

    except Exception as e:
        logger.error(f"获取组合失败: {e}", exc_info=True)
        return

    # Step 2: 按基金分组处理
    logger.info("=" * 100)
    logger.info("Step 2: 按基金分组处理")
    logger.info("=" * 100)

    # 将组合按基金代码分组
    fund_groups = {}
    for fund_code, trade_date in combinations:
        if fund_code not in fund_groups:
            fund_groups[fund_code] = []
        fund_groups[fund_code].append(trade_date)

    logger.info(f"共 {len(fund_groups)} 个基金需要处理")

    # 记录开始时间
    start_time = time.time()

    # 统计
    success_count = 0
    fail_count = 0
    results = {}

    # 遍历每个基金
    for idx, (fund_code, dates) in enumerate(fund_groups.items(), 1):
        logger.info(f"\n[{idx}/{len(fund_groups)}] 处理基金: {fund_code}")
        logger.info(f"  起始日期: {dates[0]}")
        logger.info(f"  截止日期: {dates[-1]}")
        logger.info(f"  交易日数量: {len(dates)}")

        try:
            # 执行买入并持有策略
            result = run_single_fund(
                fund_code=fund_code,
                start_date=dates[0],
                end_date=dates[-1]
            )

            results[fund_code] = result

            if result.get('status') == 'completed':
                success_count += 1
                logger.info(f"  ✓ 处理成功")
            else:
                fail_count += 1
                logger.info(f"  ✗ 处理失败: {result.get('error', '未知错误')}")

        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ 处理异常: {e}", exc_info=True)
            results[fund_code] = {'error': str(e)}

    # 计算总用时
    total_time = time.time() - start_time

    # Step 3: 输出最终统计
    logger.info("=" * 100)
    logger.info("买入并持有策略 - 执行完成")
    logger.info("=" * 100)
    logger.info(f"最终统计:")
    logger.info(f"  总基金数: {len(fund_groups)}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  成功率: {success_count/len(fund_groups)*100:.1f}%")
    logger.info(f"  总用时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
    if len(fund_groups) > 0:
        logger.info(f"  平均每个基金用时: {total_time/len(fund_groups):.1f}秒")

    # 打印汇总表格
    logger.info("\n" + "=" * 100)
    logger.info("策略执行结果汇总")
    logger.info("=" * 100)
    logger.info(f"{'基金代码':<15} {'交易日':<8} {'最终净值':<15} {'总收益率':<10} {'累计分红':<12}")
    logger.info("-" * 70)

    total_return_sum = 0
    valid_count = 0

    for fund_code, result in results.items():
        if 'error' in result:
            logger.info(f"{fund_code:<15} {'错误: ' + result['error']}")
            continue

        if result.get('status') == 'completed':
            logger.info(
                f"{fund_code:<15} "
                f"{result['trading_days']:<8} "
                f"{result['final_nav']:<15.2f} "
                f"{result['total_return']:<10.2%} "
                f"{result.get('total_dividend', 0):<12.2f}"
            )

            total_return_sum += result['total_return']
            valid_count += 1

    if valid_count > 0:
        avg_return = total_return_sum / valid_count
        logger.info("-" * 70)
        logger.info(f"平均收益率: {avg_return:.2%}")

    logger.info("=" * 100)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行失败: {e}")
        import traceback
        traceback.print_exc()
