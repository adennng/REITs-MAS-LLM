#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
账户管理模块
负责维护账户状态：现金、份额、净值、回撤、收益率等
"""

import os
import sys
from typing import Optional, Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.config import (
    INITIAL_CAPITAL,
    POSITION_MAX,
    STEP,
    FEE_RATE_BUY,
    FEE_RATE_SELL
)
from agent_trading.utils.helpers import (
    calculate_return,
    calculate_drawdown,
    round_decimal,
    clip_value
)


class Account:
    """
    账户管理类

    管理单个基金的账户状态，包括：
    - 现金、份额、市值、总净值
    - 账户全局统计（峰值净值、最大回撤、总收益率）
    - 交易执行和手续费计算
    """

    def __init__(self, initial_capital: float = None):
        """
        初始化账户

        Args:
            initial_capital: 初始资金，默认使用配置值
        """
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.cash = self.initial_capital
        self.shares = 0.0
        self.current_price = 0.0

        # 账户全局统计（基于净值）
        self.peak_nav = self.initial_capital  # 历史最高净值（初始为初始资金）
        self.max_drawdown = 0.0  # 账户最大回撤率（负数）

        # 策略运行天数（从第一个交易日开始计数）
        self.days_since_start = 0
        self.start_date = None

        # 分红统计
        self.total_dividend_received = 0.0  # 累计收到的分红总额

        # 本轮统计（用于止盈止损判断）
        self.round_entry_price = None    # 本轮平均入场成本
        self.round_peak_nav = None       # 本轮最高净值
        self.round_max_drawdown = None   # 本轮最大回撤率（负数）
        self.round_holding_days = 0      # 本轮持仓天数

    @property
    def market_value(self) -> float:
        """持仓市值"""
        return self.shares * self.current_price

    @property
    def nav(self) -> float:
        """账户总净值（现金 + 市值）"""
        return self.cash + self.market_value

    @property
    def position(self) -> float:
        """当前仓位比例 [0, 1]"""
        if self.nav == 0:
            return 0.0
        return self.market_value / self.nav

    @property
    def total_return(self) -> float:
        """账户总收益率"""
        return (self.nav - self.initial_capital) / self.initial_capital

    @property
    def round_return(self) -> Optional[float]:
        """本轮收益率（仅在有持仓时有效）"""
        if self.position >= 0.001 and self.round_entry_price and self.round_entry_price > 0:
            return (self.current_price - self.round_entry_price) / self.round_entry_price
        return None

    def update_price(self, price: float, date: str = None):
        """
        更新当前价格

        在每个交易日开始时调用，用于更新市值、峰值净值、最大回撤

        Args:
            price: 当日收盘价
            date: 当前日期
        """
        self.current_price = price

        # 更新策略运行天数
        if self.start_date is None:
            self.start_date = date
            self.days_since_start = 1
        else:
            self.days_since_start += 1

        # 更新当前净值
        current_nav = self.nav

        # 更新账户历史最高净值
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav

        # 计算并更新账户最大回撤
        if self.peak_nav > 0:
            current_drawdown = (self.peak_nav - current_nav) / self.peak_nav
            # max_drawdown 存储的是负数（最大回撤）
            if current_drawdown > abs(self.max_drawdown):
                self.max_drawdown = -current_drawdown

        # 更新本轮统计（仅在有持仓时）
        if self.position >= 0.001:
            # 更新本轮持仓天数
            self.round_holding_days += 1

            # 更新本轮最高净值
            if self.round_peak_nav is None or current_nav > self.round_peak_nav:
                self.round_peak_nav = current_nav

            # 计算并更新本轮最大回撤（使用净值）
            if self.round_peak_nav and self.round_peak_nav > 0:
                current_round_dd = (current_nav - self.round_peak_nav) / self.round_peak_nav
                # round_max_drawdown 存储为负数，只会变得更负或保持不变
                if self.round_max_drawdown is None:
                    self.round_max_drawdown = current_round_dd if current_round_dd < 0 else 0.0
                elif current_round_dd < self.round_max_drawdown:
                    self.round_max_drawdown = current_round_dd

    def process_dividend(self, dividend_per_share: float) -> float:
        """
        处理分红

        分红以现金形式留在账户中

        Args:
            dividend_per_share: 每份分红金额

        Returns:
            float: 收到的分红总额
        """
        if self.shares <= 0 or dividend_per_share <= 0:
            return 0.0

        dividend_amount = self.shares * dividend_per_share
        self.cash += dividend_amount
        self.total_dividend_received += dividend_amount  # 累计分红统计
        return round_decimal(dividend_amount)

    def _start_new_round(self, entry_price: float):
        """
        开启新一轮持仓

        Args:
            entry_price: 入场价格
        """
        self.round_entry_price = entry_price
        self.round_peak_nav = self.nav  # 使用当前净值
        self.round_max_drawdown = 0.0
        self.round_holding_days = 0

    def _end_round(self):
        """
        结束本轮持仓，重置所有本轮统计
        """
        self.round_entry_price = None
        self.round_peak_nav = None
        self.round_max_drawdown = None
        self.round_holding_days = 0

    def execute_trade(
        self,
        target_position: float,
        price: float
    ) -> Dict[str, float]:
        """
        执行交易

        根据目标仓位计算交易份额和金额，执行买卖

        Args:
            target_position: 目标仓位比例 [0, 1]
            price: 交易价格

        Returns:
            Dict: 交易详情 {
                'trade_shares': 交易份额（正买负卖）,
                'trade_amount': 交易金额,
                'trade_cost': 手续费,
                'position_before': 交易前仓位,
                'position_after': 交易后仓位
            }
        """
        # 限制目标仓位范围
        target_position = clip_value(target_position, 0.0, POSITION_MAX)

        # 记录交易前状态
        position_before = self.position
        shares_before = self.shares
        cash_before = self.cash

        # 计算目标市值和当前市值
        current_nav = self.nav
        target_market_value = current_nav * target_position
        current_market_value = self.market_value

        # 计算需要交易的金额
        trade_amount = target_market_value - current_market_value

        # 计算交易份额
        if price > 0:
            trade_shares = trade_amount / price
        else:
            trade_shares = 0

        # 计算手续费
        if trade_shares > 0:
            # 买入
            trade_cost = abs(trade_amount) * FEE_RATE_BUY
        else:
            # 卖出
            trade_cost = abs(trade_amount) * FEE_RATE_SELL

        # 执行交易
        if trade_shares > 0:
            # 买入：扣除现金，增加份额
            actual_cost = abs(trade_amount) + trade_cost
            if actual_cost > self.cash:
                # 现金不足，调整买入量
                available = self.cash / (1 + FEE_RATE_BUY)
                trade_shares = available / price
                trade_amount = trade_shares * price
                trade_cost = trade_amount * FEE_RATE_BUY
                actual_cost = trade_amount + trade_cost

            self.cash -= actual_cost
            self.shares += trade_shares

            # 本轮管理：检测新轮开始或更新成本
            if position_before < 0.001:
                # 从0仓位首次建仓，开启新轮
                self._start_new_round(price)
            else:
                # 已有持仓继续加仓，更新加权平均成本
                if self.round_entry_price and shares_before > 0:
                    old_cost = shares_before * self.round_entry_price
                    new_cost = trade_shares * price
                    self.round_entry_price = (old_cost + new_cost) / self.shares

        elif trade_shares < 0:
            # 卖出：增加现金，减少份额
            sell_shares = abs(trade_shares)
            if sell_shares > self.shares:
                sell_shares = self.shares

            sell_amount = sell_shares * price
            trade_cost = sell_amount * FEE_RATE_SELL

            self.cash += sell_amount - trade_cost
            self.shares -= sell_shares

            # 如果清仓，归零
            if self.shares < 1e-6:
                self.shares = 0

            trade_shares = -sell_shares
            trade_amount = -sell_amount

            # 本轮管理：检测清仓
            if self.position < 0.001:
                # 清仓，结束本轮
                self._end_round()

        # 返回交易详情
        return {
            'trade_shares': round_decimal(trade_shares),
            'trade_amount': round_decimal(abs(trade_amount)),
            'trade_cost': round_decimal(trade_cost),
            'position_before': round_decimal(position_before, 6),
            'position_after': round_decimal(self.position, 6),
            'shares_before': round_decimal(shares_before),
            'shares_after': round_decimal(self.shares),
            'cash_before': round_decimal(cash_before),
            'cash_after': round_decimal(self.cash),
            'nav_before': round_decimal(cash_before + shares_before * price),
            'nav_after': round_decimal(self.nav)
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        获取当前账户状态快照

        Returns:
            Dict: 账户状态
        """
        return {
            'cash': round_decimal(self.cash),
            'shares': round_decimal(self.shares),
            'current_price': round_decimal(self.current_price),
            'market_value': round_decimal(self.market_value),
            'nav': round_decimal(self.nav),
            'position': round_decimal(self.position, 6),
            'peak_nav': round_decimal(self.peak_nav),
            'max_drawdown': round_decimal(self.max_drawdown, 6),
            'total_return': round_decimal(self.total_return, 6),
            'days_since_start': self.days_since_start,
            # 本轮统计
            'round_entry_price': round_decimal(self.round_entry_price, 4) if self.round_entry_price else None,
            'round_peak_nav': round_decimal(self.round_peak_nav) if self.round_peak_nav else None,
            'round_max_drawdown': round_decimal(self.round_max_drawdown, 6) if self.round_max_drawdown is not None else None,
            'round_return': round_decimal(self.round_return, 6) if self.round_return is not None else None,
            'round_holding_days': self.round_holding_days
        }

    def reset(self):
        """重置账户到初始状态"""
        self.cash = self.initial_capital
        self.shares = 0.0
        self.current_price = 0.0
        self.peak_nav = self.initial_capital
        self.max_drawdown = 0.0
        self.days_since_start = 0
        self.start_date = None
        # 重置本轮统计
        self.round_entry_price = None
        self.round_peak_nav = None
        self.round_max_drawdown = None
        self.round_holding_days = 0
