#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作执行模块
根据delta_steps执行交易，更新账户状态
"""

import os
import sys
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.config import STEP, POSITION_MAX, LLM_MODEL
from agent_trading.core.account import Account
from agent_trading.core.risk_controller import RiskController
from agent_trading.core.signal_processor import get_signal_description
from agent_trading.data.data_fetcher import DataFetcher
from agent_trading.utils.helpers import (
    determine_action_type,
    calculate_daily_regime,
    round_decimal,
    clip_value
)
from agent_trading.utils.logger import get_logger

logger = get_logger()


class ActionExecutor:
    """
    动作执行器

    根据LLM决策执行交易，更新账户状态，保存记录
    """

    def __init__(self):
        """初始化动作执行器"""
        self.risk_controller = RiskController()
        self.data_fetcher = DataFetcher()

    def execute(
        self,
        delta_steps: int,
        state: Dict[str, Any],
        account: Account,
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行交易动作

        Args:
            delta_steps: LLM输出的调仓步数
            state: 完整的state_t
            account: 账户对象
            llm_result: LLM调用结果（包含reasoning、prompt等）

        Returns:
            Dict: 执行结果，包含交易详情和完整记录
        """
        analysis_date = state['meta_info']['analysis_date']
        fund_code = state['meta_info']['fund_code']

        # 获取当日收盘价和波动阈值
        current_price = self.data_fetcher.get_single_day_price(fund_code, analysis_date)
        volatility_threshold = float(state['_raw_prediction'].get('volatility_threshold', 0) or 0)

        # 计算当日涨跌幅和ratio_t
        price_change_1d = None
        ratio_t = None
        daily_regime = None

        # 获取前一日价格
        prev_date = self.data_fetcher.get_previous_trading_date(fund_code, analysis_date)
        if prev_date:
            prev_price = self.data_fetcher.get_single_day_price(fund_code, prev_date)
            if prev_price and prev_price > 0:
                # 计算涨跌幅
                price_change_1d = round_decimal((current_price - prev_price) / prev_price, 6)

                # 计算ratio_t（保留用于数据库记录）
                if volatility_threshold and volatility_threshold > 0:
                    ratio_t = round_decimal(abs(price_change_1d) / volatility_threshold, 6)

                    # 计算daily_regime（使用统一函数）
                    daily_regime = calculate_daily_regime(price_change_1d, volatility_threshold)

        # 1. 风控检查
        delta_steps_original = delta_steps
        delta_steps_final, override_reason = self.risk_controller.check_and_override(
            delta_steps, state
        )

        # 2. 计算目标仓位
        current_position = account.position
        target_position = self.risk_controller.calculate_target_position(
            delta_steps_final, current_position
        )

        # 3. 记录执行前状态
        state_before = {
            'position': round_decimal(account.position, 6),
            'shares': round_decimal(account.shares),
            'cash': round_decimal(account.cash),
            'nav': round_decimal(account.nav)
        }

        # 4. 检查并处理分红
        dividend_received = 0.0
        dividend_per_share = self.data_fetcher.get_dividend_on_date(fund_code, analysis_date)
        if dividend_per_share and dividend_per_share > 0:
            # 当日是除息日，计算分红
            # 注意：分红是基于权益登记日（除息日前一个交易日）的持仓
            # 但我们在当日处理，因为分红到账是在除息日
            dividend_received = account.process_dividend(dividend_per_share)
            if dividend_received > 0:
                logger.info(f"  收到分红: {dividend_received:.2f}元 "
                           f"(每份{dividend_per_share:.4f}元 × {account.shares:.4f}份)")

        # 5. 执行交易
        trade_result = account.execute_trade(target_position, current_price)

        # 6. 确定动作类型
        action_type = determine_action_type(
            state_before['position'],
            trade_result['position_after']
        )

        # 7. 组装完整记录
        decision_record = {
            # 元信息
            'analysis_date': analysis_date,
            'fund_code': fund_code,

            # 决策前状态
            'position_before': state_before['position'],
            'shares_before': state_before['shares'],
            'cash_before': state_before['cash'],
            'nav_before': state_before['nav'],

            # 策略状态
            'days_since_start': state['account_state']['days_since_start'],
            'is_building_phase': state['phase_info']['is_building_phase'],

            # 价格上下文
            'current_price': current_price,
            'volatility_threshold': volatility_threshold,
            'price_change_1d': price_change_1d,
            'ratio_t': ratio_t,
            'daily_regime': daily_regime,

            # 信号档位（中文）
            'signal_level_t1': get_signal_description(state['direction_output']['T1']['signal_level']),
            'signal_level_t5': get_signal_description(state['direction_output']['T5']['signal_level']),
            'signal_level_t20': get_signal_description(state['direction_output']['T20']['signal_level']),

            # 决策输出
            'delta_steps': delta_steps_final,
            'target_position': round_decimal(target_position, 6),
            'action_type': action_type,
            'rationale_core': llm_result.get('rationale_core', ''),

            # 决策后状态
            'position_after': trade_result['position_after'],
            'shares_after': trade_result['shares_after'],
            'cash_after': trade_result['cash_after'],
            'nav_after': trade_result['nav_after'],

            # 账户全局统计（基于净值）
            'peak_nav': round_decimal(account.peak_nav),
            'max_drawdown': round_decimal(account.max_drawdown, 6),
            'total_return': round_decimal(account.total_return, 6),

            # 本轮统计（用于止盈止损）
            'round_entry_price': round_decimal(account.round_entry_price, 4) if account.round_entry_price else None,
            'round_peak_nav': round_decimal(account.round_peak_nav) if account.round_peak_nav else None,
            'round_max_drawdown': round_decimal(account.round_max_drawdown, 6) if account.round_max_drawdown is not None else None,
            'round_return': round_decimal(account.round_return, 6) if account.round_return is not None else None,
            'round_holding_days': account.round_holding_days,

            # 交易执行
            'trade_shares': trade_result['trade_shares'],
            'trade_amount': trade_result['trade_amount'],
            'trade_cost': trade_result['trade_cost'],
            'dividend_today': dividend_received,  # 当日分红
            'total_dividend_received': round_decimal(account.total_dividend_received),  # 累计分红

            # LLM相关
            'llm_prompt': llm_result.get('prompt', ''),
            'llm_reasoning': llm_result.get('reasoning', ''),
            'llm_response': llm_result.get('raw_response', ''),

            # 模型信息
            'model_name': LLM_MODEL,
            'model_type': 'strong_llm',

            # 风控信息
            'risk_override_reason': override_reason,
            'delta_steps_original': delta_steps_original if override_reason else None
        }

        # 8. 保存到数据库
        try:
            record_id = self.data_fetcher.save_trading_decision(decision_record)
            logger.debug(f"  决策记录已保存，ID: {record_id}")
        except Exception as e:
            logger.error(f"  保存决策记录失败: {e}")

        # 9. 打印执行摘要
        self._print_execution_summary(decision_record)

        return decision_record

    def _print_execution_summary(self, record: Dict[str, Any]):
        """
        打印执行摘要到控制台

        Args:
            record: 决策记录
        """
        # 构建摘要信息
        date = record['analysis_date']
        fund = record['fund_code']
        delta = record['delta_steps']
        action = record['action_type']

        pos_before = record['position_before']
        pos_after = record['position_after']
        nav_after = record['nav_after']

        # 账户总收益率
        total_return_str = ""
        if record.get('total_return') is not None:
            total_return = record['total_return']
            total_return_str = f"总收益: {total_return:+.2%}"

        # 最大回撤
        drawdown_str = ""
        if record.get('max_drawdown') is not None:
            drawdown = record['max_drawdown']
            drawdown_str = f" 最大回撤: {drawdown:.2%}"

        # 风控标记
        risk_str = ""
        if record.get('risk_override_reason'):
            risk_str = f" [风控: {record['risk_override_reason']}]"

        # 分红标记
        div_str = ""
        if record.get('dividend_today', 0) > 0:
            div_str = f" [分红: +{record['dividend_today']:.2f}]"

        # delta_steps显示
        if delta > 0:
            delta_str = f"+{delta}"
        else:
            delta_str = str(delta)

        logger.info(
            f"  {date} | {fund} | "
            f"决策: {delta_str} ({action}) | "
            f"仓位: {pos_before:.0%} → {pos_after:.0%} | "
            f"净值: ¥{nav_after:,.0f} | "
            f"{total_return_str}{drawdown_str}{div_str}{risk_str}"
        )
