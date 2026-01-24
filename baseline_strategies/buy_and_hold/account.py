#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买入并持有策略 - 账户管理模块
管理账户状态：现金、份额、净值、持仓成本、浮盈浮亏等
"""

import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(baseline_dir)
sys.path.insert(0, project_root)

from baseline_strategies.config import (
    INITIAL_CAPITAL,
    FEE_RATE_BUY
)


class BuyAndHoldAccount:
    """
    买入并持有策略账户类

    特点：
    - 首日全仓买入
    - 持有至到期，不卖出
    - 分红留在现金中
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

        # 持仓信息
        self.peak_nav = 0.0     # 历史最高净值
        self.holding_days = 0   # 持仓天数
        self.max_drawdown_value = 0.0  # 历史最大回撤率（负数表示回撤）

        # 分红信息
        self.total_dividend_received = 0.0  # 累计分红

        # 交易信息（仅首日）
        self.buy_shares = 0.0
        self.buy_amount = 0.0
        self.buy_cost = 0.0

    @property
    def market_value(self) -> float:
        """持仓市值"""
        return self.shares * self.current_price

    @property
    def nav(self) -> float:
        """账户总净值（现金 + 市值）"""
        return self.cash + self.market_value

    @property
    def max_drawdown(self) -> float:
        """最大回撤（基于净值，负数表示回撤）

        返回历史上出现过的最大净值回撤率，该值只会增大（更负）或保持不变
        """
        return self.max_drawdown_value

    @property
    def total_return(self) -> float:
        """总收益率"""
        return (self.nav - self.initial_capital) / self.initial_capital

    def update_price(self, price: float):
        """
        更新当前价格

        Args:
            price: 当日收盘价
        """
        self.current_price = price

        # 如果有持仓，更新峰值净值和持仓天数
        if self.shares > 0:
            # 计算当前净值
            current_nav = self.nav

            # 更新净值峰值
            if current_nav > self.peak_nav:
                self.peak_nav = current_nav

            # 计算当前回撤率并更新最大回撤
            if self.peak_nav > 0:
                current_drawdown = (current_nav - self.peak_nav) / self.peak_nav
                # 最大回撤取更小的值（更负的值）
                if current_drawdown < self.max_drawdown_value:
                    self.max_drawdown_value = current_drawdown

            # 更新持仓天数
            self.holding_days += 1

    def execute_buy(self, price: float) -> dict:
        """
        执行买入（仅首日）

        全仓买入，考虑手续费

        Args:
            price: 买入价格

        Returns:
            dict: 交易详情
        """
        # 计算可买入金额（扣除手续费后）
        available_amount = self.cash / (1 + FEE_RATE_BUY)

        # 计算买入份额
        buy_shares = available_amount / price

        # 计算实际买入金额
        buy_amount = buy_shares * price

        # 计算手续费
        buy_cost = buy_amount * FEE_RATE_BUY

        # 执行买入
        self.shares = buy_shares
        self.cash = 0.0  # 全仓买入，现金清零

        # 初始化净值峰值（买入后的初始净值）
        self.peak_nav = self.nav
        self.holding_days = 1

        # 记录交易信息
        self.buy_shares = buy_shares
        self.buy_amount = buy_amount
        self.buy_cost = buy_cost

        return {
            'buy_shares': buy_shares,
            'buy_amount': buy_amount,
            'buy_cost': buy_cost
        }

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
        self.total_dividend_received += dividend_amount

        return dividend_amount
