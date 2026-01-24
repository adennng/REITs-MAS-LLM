#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态构建模块
从数据库获取数据并组装完整的state_t
"""

import os
import sys
from typing import Dict, Any, Optional, List
from datetime import date

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.config import (
    POSITION_MAX,
    STEP,
    K_MAX,
    BUILDING_TARGET_POSITION,
    D_BUILD_MAX,
    P_BUILD_DONE,
    SOFT_DD_LIMIT,
    HARD_DD_LIMIT,
    TAKE_PROFIT_LEVEL,
    MIN_HOLDING_DAYS_BEFORE_TP,
    REGIME_5D_SQRT_MULTIPLIER
)
from agent_trading.data.data_fetcher import DataFetcher
from agent_trading.core.account import Account
from agent_trading.core.signal_processor import process_direction_signals
from agent_trading.utils.helpers import (
    calculate_return,
    calculate_daily_regime,
    calculate_5d_regime,
    round_decimal
)
from agent_trading.utils.logger import get_logger

logger = get_logger()


class StateBuilder:
    """
    状态构建器

    负责从数据库获取数据并组装完整的state_t
    """

    def __init__(self):
        """初始化状态构建器"""
        self.data_fetcher = DataFetcher()

    def build_state(
        self,
        analysis_date: str,
        fund_code: str,
        account: Account,
        recent_decisions: List[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        构建完整的state_t

        Args:
            analysis_date: 分析日期
            fund_code: 基金代码
            account: 账户对象
            recent_decisions: 最近几天的决策记录（用于显示历史操作）

        Returns:
            Dict: 完整的state_t，如果数据不足返回None
        """
        # 1. 获取预测数据
        prediction_data = self.data_fetcher.get_prediction_data(analysis_date, fund_code)
        if not prediction_data:
            logger.warning(f"未找到预测数据: {fund_code} @ {analysis_date}")
            return None

        # 2. 获取当日价格
        current_price = self.data_fetcher.get_single_day_price(fund_code, analysis_date)
        if current_price is None:
            logger.warning(f"未找到价格数据: {fund_code} @ {analysis_date}")
            return None

        # 3. 获取历史价格数据（最近5天，包含当日）
        price_history = self.data_fetcher.get_price_data(fund_code, analysis_date, days=5)
        if len(price_history) < 2:
            logger.warning(f"历史价格数据不足: {fund_code} @ {analysis_date}")
            return None

        # 4. 计算价格上下文
        price_context = self._build_price_context(
            price_history,
            current_price,
            float(prediction_data.get('volatility_threshold', 0) or 0),
            recent_decisions
        )

        # 5. 计算信号档位
        signal_levels = process_direction_signals(prediction_data)

        # 6. 判断是否处于建仓期
        is_building_phase = self._is_building_phase(
            account.days_since_start,
            account.position
        )

        # 7. 组装state_t
        state = {
            # 元信息
            'meta_info': {
                'analysis_date': analysis_date,
                'fund_code': fund_code
            },

            # 四专家输出
            'four_experts_output': {
                'announcement': prediction_data.get('output_announcement'),
                'market': prediction_data.get('output_market'),
                'price': prediction_data.get('output_price'),
                'event': prediction_data.get('output_event')
            },

            # 预测专家输出
            'direction_output': {
                'direction_prediction': prediction_data.get('direction_prediction'),
                'T1': {
                    'prob_up': float(prediction_data.get('t1_prob_up', 0) or 0),
                    'prob_down': float(prediction_data.get('t1_prob_down', 0) or 0),
                    'prob_side': float(prediction_data.get('t1_prob_side', 0) or 0),
                    'confidence': float(prediction_data.get('conf_T1', 0) or 0),
                    'pred_dir': prediction_data.get('pred_dir_T1'),
                    'signal_level': signal_levels['signal_level_T1']
                },
                'T5': {
                    'prob_up': float(prediction_data.get('t5_prob_up', 0) or 0),
                    'prob_down': float(prediction_data.get('t5_prob_down', 0) or 0),
                    'prob_side': float(prediction_data.get('t5_prob_side', 0) or 0),
                    'confidence': float(prediction_data.get('conf_T5', 0) or 0),
                    'pred_dir': prediction_data.get('pred_dir_T5'),
                    'signal_level': signal_levels['signal_level_T5']
                },
                'T20': {
                    'prob_up': float(prediction_data.get('t20_prob_up', 0) or 0),
                    'prob_down': float(prediction_data.get('t20_prob_down', 0) or 0),
                    'prob_side': float(prediction_data.get('t20_prob_side', 0) or 0),
                    'confidence': float(prediction_data.get('conf_T20', 0) or 0),
                    'pred_dir': prediction_data.get('pred_dir_T20'),
                    'signal_level': signal_levels['signal_level_T20']
                }
            },

            # 价格上下文
            'price_context': price_context,

            # 账户状态
            'account_state': {
                'position': round_decimal(account.position, 6),
                'cash': round_decimal(account.cash),
                'shares': round_decimal(account.shares),
                'nav': round_decimal(account.nav),
                'days_since_start': account.days_since_start,
                'peak_nav': round_decimal(account.peak_nav),
                'max_drawdown': round_decimal(account.max_drawdown, 6),
                'total_return': round_decimal(account.total_return, 6),
                # 本轮统计
                'round_entry_price': round_decimal(account.round_entry_price, 4) if account.round_entry_price else None,
                'round_peak_nav': round_decimal(account.round_peak_nav) if account.round_peak_nav else None,
                'round_max_drawdown': round_decimal(account.round_max_drawdown, 6) if account.round_max_drawdown is not None else None,
                'round_return': round_decimal(account.round_return, 6) if account.round_return is not None else None,
                'round_holding_days': account.round_holding_days
            },

            # 风险配置
            'risk_config': {
                'position_max': POSITION_MAX,
                'step': STEP,
                'K_max': K_MAX,
                'building_target_position': BUILDING_TARGET_POSITION,
                'D_build_max': D_BUILD_MAX,
                'soft_dd_limit': SOFT_DD_LIMIT,
                'hard_dd_limit': HARD_DD_LIMIT,
                'take_profit_level': TAKE_PROFIT_LEVEL,
                'min_holding_days_before_tp': MIN_HOLDING_DAYS_BEFORE_TP
            },

            # 阶段信息
            'phase_info': {
                'is_building_phase': is_building_phase,
                'days_since_start': account.days_since_start,
                'building_target_position': BUILDING_TARGET_POSITION
            },

            # 原始预测数据（用于保存到数据库）
            '_raw_prediction': prediction_data
        }

        return state

    def _build_price_context(
        self,
        price_history: List[Dict[str, Any]],
        current_price: float,
        volatility_threshold: float,
        recent_decisions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        构建价格上下文

        Args:
            price_history: 历史价格数据（从旧到新）
            current_price: 当前价格
            volatility_threshold: 动态波动阈值
            recent_decisions: 最近的决策记录（包含完整的价格上下文字段）

        Returns:
            Dict: 价格上下文
        """
        # 提取收盘价序列，用于计算5日涨跌幅
        closes = [float(p['close']) for p in price_history]

        # 计算1日涨跌幅（仅用于当前上下文）
        if len(closes) >= 2:
            price_change_1d = calculate_return(closes[-1], closes[-2])
        else:
            price_change_1d = 0.0

        # 计算5日涨跌幅
        if len(closes) >= 5:
            price_change_5d = calculate_return(closes[-1], closes[-5])
        else:
            price_change_5d = 0.0

        # 计算盘整/突破状态（仅用于当前上下文）
        daily_regime = calculate_daily_regime(price_change_1d, volatility_threshold)

        # 构建最近5日详情
        # 从recent_decisions取最近4条历史记录，然后添加当前分析日期作为第5条
        recent_days = []
        if recent_decisions:
            # 取最近4条历史决策记录
            for dec in recent_decisions[-4:]:  # 取最近4条
                dec_date = dec.get('analysis_date')
                if isinstance(dec_date, date):
                    dec_date = dec_date.strftime('%Y-%m-%d')

                day_info = {
                    'date': dec_date,
                    'close': float(dec.get('current_price', 0)),
                    'change_pct': float(dec.get('price_change_1d', 0)) if dec.get('price_change_1d') is not None else None,
                    'delta_steps': dec.get('delta_steps'),
                    'volatility_threshold': float(dec.get('volatility_threshold', 0)) if dec.get('volatility_threshold') is not None else None,
                    'ratio_t': float(dec.get('ratio_t', 0)) if dec.get('ratio_t') is not None else None,
                    'daily_regime': dec.get('daily_regime')
                }
                recent_days.append(day_info)

        # 添加当前分析日期的数据作为第5条（最新的一条）
        if price_history:
            # 从price_history获取当前分析日期
            current_date = price_history[-1].get('trade_date')
            if isinstance(current_date, date):
                current_date = current_date.strftime('%Y-%m-%d')

            current_day_info = {
                'date': current_date,
                'close': float(current_price),
                'change_pct': float(price_change_1d) if price_change_1d is not None else None,
                'delta_steps': None,  # 当前日期还未决策，所以为None
                'volatility_threshold': float(volatility_threshold) if volatility_threshold is not None else None,
                'ratio_t': None,  # 当前日期还未计算ratio_t
                'daily_regime': daily_regime
            }
            recent_days.append(current_day_info)

        # 计算最近5日盘整状态
        regime_5d = calculate_5d_regime(
            price_change_5d,
            volatility_threshold,
            REGIME_5D_SQRT_MULTIPLIER
        )

        return {
            'current_price': round_decimal(current_price),
            'price_change_1d': round_decimal(price_change_1d, 6),
            'price_change_5d': round_decimal(price_change_5d, 6),
            'volatility_threshold': round_decimal(volatility_threshold, 6),
            'daily_regime': daily_regime,
            'regime_5d': regime_5d,
            'recent_5_days': recent_days
        }

    def _is_building_phase(
        self,
        days_since_start: int,
        current_position: float
    ) -> bool:
        """
        判断是否处于建仓期

        Args:
            days_since_start: 策略运行天数
            current_position: 当前仓位

        Returns:
            bool: 是否处于建仓期
        """
        return (days_since_start <= D_BUILD_MAX and
                current_position < P_BUILD_DONE)
