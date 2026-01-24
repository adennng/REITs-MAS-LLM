#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测主流程控制器
管理整体回测流程
"""

import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import date

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.config import (
    START_DATE,
    END_DATE,
    FUND_CODES,
    INITIAL_CAPITAL,
    MAX_WORKERS,
    ENABLE_RESUME
)
from agent_trading.core.account import Account
from agent_trading.core.state_builder import StateBuilder
from agent_trading.core.action_executor import ActionExecutor
from agent_trading.data.data_fetcher import DataFetcher
from agent_trading.llm.prompt_builder import build_trading_prompt
from agent_trading.llm.llm_caller import call_trading_llm
from agent_trading.utils.logger import setup_logger, get_logger
from agent_trading.utils.helpers import format_currency, format_percentage

logger = get_logger()


def _run_single_fund_wrapper(
    fund_code: str,
    start_date: str,
    end_date: str,
    initial_capital: float
) -> Dict[str, Any]:
    """
    单个基金回测的包装函数（用于多进程调用）

    注意：每个子进程都会创建独立的数据库连接和模块实例

    Args:
        fund_code: 基金代码
        start_date: 起始日期
        end_date: 终止日期
        initial_capital: 初始资金

    Returns:
        Dict: 回测结果
    """
    # 在子进程中重新初始化logger（确保日志正常）
    from agent_trading.utils.logger import setup_logger
    setup_logger()

    # 创建回测器实例（每个进程独立）
    backtester = Backtester(
        start_date=start_date,
        end_date=end_date,
        fund_codes=None,
        initial_capital=initial_capital
    )

    # 执行单个基金回测
    return backtester._run_single_fund(fund_code)


class Backtester:
    """
    回测控制器

    管理整体回测流程，遍历交易日执行决策
    """

    def __init__(
        self,
        start_date: str = None,
        end_date: str = None,
        fund_codes: List[str] = None,
        initial_capital: float = None
    ):
        """
        初始化回测器

        Args:
            start_date: 起始日期
            end_date: 终止日期
            fund_codes: 基金代码列表
            initial_capital: 初始资金
        """
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.fund_codes = fund_codes or FUND_CODES
        self.initial_capital = initial_capital or INITIAL_CAPITAL

        # 初始化各模块
        self.data_fetcher = DataFetcher()
        self.state_builder = StateBuilder()
        self.action_executor = ActionExecutor()

        # 回测结果
        self.results = {}

    def run(self):
        """
        运行回测

        对每个基金独立运行回测，支持多进程并行
        """
        logger.info("=" * 80)
        logger.info("开始回测")
        logger.info(f"日期范围: {self.start_date} ~ {self.end_date}")
        logger.info(f"初始资金: {format_currency(self.initial_capital)}")
        logger.info(f"断点续传: {'启用' if ENABLE_RESUME else '禁用'}")
        logger.info("=" * 80)

        # 获取基金列表
        if self.fund_codes:
            fund_codes = self.fund_codes
        else:
            fund_codes = self.data_fetcher.get_fund_codes_in_range(
                self.start_date, self.end_date
            )

        if not fund_codes:
            logger.error("未找到任何基金")
            return

        logger.info(f"待回测基金: {', '.join(fund_codes)}")

        # 确定实际使用的进程数
        actual_workers = self._determine_workers(len(fund_codes))

        if actual_workers == 1:
            # 串行处理
            logger.info("使用单进程串行处理")
            logger.info("=" * 80)
            for i, fund_code in enumerate(fund_codes, 1):
                logger.info(f"\n[{i}/{len(fund_codes)}] 开始回测基金: {fund_code}")
                logger.info("-" * 60)

                result = _run_single_fund_wrapper(
                    fund_code,
                    self.start_date,
                    self.end_date,
                    self.initial_capital
                )
                self.results[fund_code] = result

                logger.info("-" * 60)
        else:
            # 并行处理
            logger.info(f"使用 {actual_workers} 个进程并行处理")
            logger.info("=" * 80)

            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                # 提交所有任务
                future_to_fund = {
                    executor.submit(
                        _run_single_fund_wrapper,
                        fund_code,
                        self.start_date,
                        self.end_date,
                        self.initial_capital
                    ): fund_code
                    for fund_code in fund_codes
                }

                # 收集结果
                completed_count = 0
                for future in as_completed(future_to_fund):
                    fund_code = future_to_fund[future]
                    completed_count += 1

                    try:
                        result = future.result()
                        self.results[fund_code] = result
                        logger.info(f"[{completed_count}/{len(fund_codes)}] 基金 {fund_code} 处理完成")
                    except Exception as e:
                        logger.error(f"[{completed_count}/{len(fund_codes)}] 基金 {fund_code} 处理失败: {e}")
                        self.results[fund_code] = {'error': str(e)}

        # 打印汇总
        self._print_summary()

    def _determine_workers(self, fund_count: int) -> int:
        """
        确定实际使用的进程数

        Args:
            fund_count: 基金数量

        Returns:
            int: 实际使用的进程数
        """
        if MAX_WORKERS is None or MAX_WORKERS <= 0:
            # 自动确定
            cpu_count = mp.cpu_count()
            # 使用CPU核心数的60%，但不超过基金数量
            workers = min(max(1, int(cpu_count * 0.6)), fund_count)
        else:
            # 使用配置值，但不超过基金数量（已移除CPU核心数限制）
            workers = min(MAX_WORKERS, fund_count)

        return workers

    def _run_single_fund(self, fund_code: str) -> Dict[str, Any]:
        """
        对单个基金运行回测（支持断点续传）

        Args:
            fund_code: 基金代码

        Returns:
            Dict: 回测结果
        """
        # 获取该基金的所有交易日列表
        all_trading_dates = self.data_fetcher.get_fund_trading_dates(
            fund_code, self.start_date, self.end_date
        )

        if not all_trading_dates:
            logger.warning(f"基金 {fund_code} 在指定日期范围内无交易数据")
            return {'error': 'no_data'}

        # 检查是否有已处理的记录（断点续传）
        existing_dates = []
        last_record = None
        if ENABLE_RESUME:
            existing_dates = self.data_fetcher.get_existing_decision_dates(
                fund_code, self.start_date, self.end_date
            )
            if existing_dates:
                logger.info(f"发现 {len(existing_dates)} 条已有记录")
                # 获取最后一条记录用于恢复账户状态
                last_record = self.data_fetcher.get_last_decision_record(
                    fund_code, existing_dates[-1]
                )

        # 计算待处理的交易日
        pending_dates = [d for d in all_trading_dates if d not in existing_dates]

        logger.info(f"总交易日: {len(all_trading_dates)}")
        logger.info(f"已处理: {len(existing_dates)}")
        logger.info(f"待处理: {len(pending_dates)}")

        if not pending_dates:
            logger.info(f"基金 {fund_code} 所有交易日已处理完毕")
            # 返回已有结果
            return self._load_result_from_last_record(fund_code, last_record, all_trading_dates)

        logger.info(f"首个交易日: {all_trading_dates[0]}")
        logger.info(f"末个交易日: {all_trading_dates[-1]}")
        if existing_dates:
            logger.info(f"续传起点: {pending_dates[0]}")

        # 初始化或恢复账户
        account = self._initialize_or_restore_account(last_record)

        # 用于存储最近的决策记录（供state_builder使用）
        recent_decisions = []
        if existing_dates:
            # 如果是续传，加载最近5条记录
            recent_decisions = self._load_recent_decisions(fund_code, existing_dates[-5:])

        # 统计
        success_count = len(existing_dates)  # 已有记录数
        fail_count = 0

        # 遍历待处理的交易日
        for date_str in pending_dates:
            logger.info(f"\n处理 {date_str}...")

            # 1. 获取当日价格并更新账户
            current_price = self.data_fetcher.get_single_day_price(fund_code, date_str)
            if current_price is None:
                logger.warning(f"  未找到价格数据，跳过")
                fail_count += 1
                continue

            account.update_price(current_price, date_str)

            # 2. 构建state_t
            state = self.state_builder.build_state(
                analysis_date=date_str,
                fund_code=fund_code,
                account=account,
                recent_decisions=recent_decisions[-5:]  # 最近5天的决策
            )

            if state is None:
                logger.warning(f"  构建状态失败，跳过")
                fail_count += 1
                continue

            # 3. 构建提示词
            prompt = build_trading_prompt(state)

            # 4. 调用LLM
            success, llm_result = call_trading_llm(prompt)

            if not success:
                logger.error(f"  LLM调用失败: {llm_result.get('error', '未知错误')}")
                fail_count += 1
                continue

            # 5. 执行交易
            delta_steps = llm_result['delta_steps']
            decision_record = self.action_executor.execute(
                delta_steps=delta_steps,
                state=state,
                account=account,
                llm_result=llm_result
            )

            # 保存决策记录
            recent_decisions.append(decision_record)
            success_count += 1

        # 计算回测结果
        final_nav = account.nav
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        result = {
            'fund_code': fund_code,
            'start_date': all_trading_dates[0] if all_trading_dates else None,
            'end_date': all_trading_dates[-1] if all_trading_dates else None,
            'trading_days': len(all_trading_dates),
            'existing_days': len(existing_dates),
            'processed_days': len(pending_dates),
            'success_count': success_count,
            'fail_count': fail_count,
            'initial_capital': self.initial_capital,
            'final_nav': final_nav,
            'total_return': total_return,
            'final_position': account.position,
            'final_cash': account.cash,
            'final_shares': account.shares
        }

        logger.info(f"\n基金 {fund_code} 回测完成:")
        logger.info(f"  总收益率: {format_percentage(total_return)}")
        logger.info(f"  最终净值: {format_currency(final_nav)}")
        logger.info(f"  最终仓位: {format_percentage(account.position)}")
        logger.info(f"  成功/失败: {success_count}/{fail_count}")

        return result

    def _initialize_or_restore_account(self, last_record: Optional[Dict]) -> Account:
        """
        初始化或从上次记录恢复账户状态

        Args:
            last_record: 最后一条决策记录

        Returns:
            Account: 账户对象
        """
        if last_record is None:
            # 全新开始
            logger.info("初始化新账户")
            return Account(self.initial_capital)

        # 从最后一条记录恢复
        logger.info(f"从记录恢复账户状态: {last_record['analysis_date']}")

        account = Account(self.initial_capital)

        # 恢复基础账户状态
        account.cash = float(last_record['cash_after'])
        account.shares = float(last_record['shares_after'])
        account.current_price = float(last_record['current_price'])

        # 恢复账户全局统计
        if last_record.get('peak_nav') is not None:
            account.peak_nav = float(last_record['peak_nav'])

        if last_record.get('max_drawdown') is not None:
            account.max_drawdown = float(last_record['max_drawdown'])

        # 恢复策略运行天数
        if last_record.get('days_since_start') is not None:
            account.days_since_start = int(last_record['days_since_start'])
            # 重要：恢复start_date，防止update_price()时重置days_since_start
            # 设置为上一次交易日的日期，表示策略已经启动过
            account.start_date = last_record['analysis_date']

        # 恢复分红统计
        if last_record.get('total_dividend_received') is not None:
            account.total_dividend_received = float(last_record['total_dividend_received'])

        # 恢复本轮统计（仅在有持仓时）
        if account.position > 0.001:
            # 有持仓，恢复本轮统计
            if last_record.get('round_entry_price') is not None:
                account.round_entry_price = float(last_record['round_entry_price'])
            if last_record.get('round_peak_nav') is not None:
                account.round_peak_nav = float(last_record['round_peak_nav'])
            if last_record.get('round_max_drawdown') is not None:
                account.round_max_drawdown = float(last_record['round_max_drawdown'])
            if last_record.get('round_holding_days') is not None:
                account.round_holding_days = int(last_record['round_holding_days'])

            logger.info(f"  账户状态: 仓位={account.position:.2%}, 净值={account.nav:.2f}, "
                       f"总收益={account.total_return:.2%}, 最大回撤={account.max_drawdown:.2%}, "
                       f"天数={account.days_since_start}")
            logger.info(f"  本轮状态: 成本={account.round_entry_price:.4f}, "
                       f"收益={account.round_return:.2%}, 回撤={account.round_max_drawdown:.2%}, "
                       f"持仓天数={account.round_holding_days}")
        else:
            # 空仓，确保本轮统计为空
            account.round_entry_price = None
            account.round_peak_nav = None
            account.round_max_drawdown = None
            account.round_holding_days = 0

            logger.info(f"  账户状态: 仓位={account.position:.2%}, 净值={account.nav:.2f}, "
                       f"总收益={account.total_return:.2%}, 最大回撤={account.max_drawdown:.2%}, "
                       f"天数={account.days_since_start}")
            logger.info(f"  空仓状态，本轮统计已重置")

        return account

    def _load_recent_decisions(self, fund_code: str, recent_dates: List[str]) -> List[Dict]:
        """
        加载最近几天的决策记录

        Args:
            fund_code: 基金代码
            recent_dates: 最近的日期列表

        Returns:
            List[Dict]: 决策记录列表
        """
        decisions = []
        for date_str in recent_dates:
            record = self.data_fetcher.get_last_decision_record(fund_code, date_str)
            if record:
                decisions.append(record)
        return decisions

    def _load_result_from_last_record(
        self,
        fund_code: str,
        last_record: Optional[Dict],
        all_dates: List[str]
    ) -> Dict[str, Any]:
        """
        从最后一条记录加载回测结果

        Args:
            fund_code: 基金代码
            last_record: 最后一条决策记录
            all_dates: 所有交易日列表

        Returns:
            Dict: 回测结果
        """
        if last_record is None:
            return {'error': 'no_record'}

        final_nav = float(last_record['nav_after'])
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        result = {
            'fund_code': fund_code,
            'start_date': all_dates[0] if all_dates else None,
            'end_date': all_dates[-1] if all_dates else None,
            'trading_days': len(all_dates),
            'existing_days': len(all_dates),
            'processed_days': 0,
            'success_count': len(all_dates),
            'fail_count': 0,
            'initial_capital': self.initial_capital,
            'final_nav': final_nav,
            'total_return': total_return,
            'final_position': float(last_record['position_after']),
            'final_cash': float(last_record['cash_after']),
            'final_shares': float(last_record['shares_after'])
        }

        logger.info(f"\n基金 {fund_code} (从已有记录):")
        logger.info(f"  总收益率: {format_percentage(total_return)}")
        logger.info(f"  最终净值: {format_currency(final_nav)}")
        logger.info(f"  最终仓位: {format_percentage(result['final_position'])}")

        return result

    def _print_summary(self):
        """打印回测汇总"""
        logger.info("\n" + "=" * 80)
        logger.info("回测汇总")
        logger.info("=" * 80)

        if not self.results:
            logger.info("无回测结果")
            return

        # 打印表头
        logger.info(f"{'基金代码':<15} {'交易日':<8} {'初始资金':<15} {'最终净值':<15} {'总收益率':<10}")
        logger.info("-" * 70)

        total_return_sum = 0
        valid_count = 0

        for fund_code, result in self.results.items():
            if 'error' in result:
                logger.info(f"{fund_code:<15} {'错误: ' + result['error']}")
                continue

            logger.info(
                f"{fund_code:<15} "
                f"{result['trading_days']:<8} "
                f"{format_currency(result['initial_capital']):<15} "
                f"{format_currency(result['final_nav']):<15} "
                f"{format_percentage(result['total_return']):<10}"
            )

            total_return_sum += result['total_return']
            valid_count += 1

        if valid_count > 0:
            avg_return = total_return_sum / valid_count
            logger.info("-" * 70)
            logger.info(f"平均收益率: {format_percentage(avg_return)}")

        logger.info("=" * 80)
