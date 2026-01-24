#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辅助函数模块
包含计算回撤、盈亏、日期处理等通用函数
"""

from typing import List, Optional
from decimal import Decimal, ROUND_HALF_UP
import math


def calculate_return(current_price: float, base_price: float) -> float:
    """
    计算收益率

    Args:
        current_price: 当前价格
        base_price: 基准价格

    Returns:
        float: 收益率
    """
    if base_price == 0:
        return 0.0
    return (current_price - base_price) / base_price


def calculate_drawdown(current_price: float, peak_price: float) -> float:
    """
    计算回撤（相对于峰值）

    Args:
        current_price: 当前价格
        peak_price: 历史峰值价格

    Returns:
        float: 回撤（负数或0）
    """
    if peak_price == 0:
        return 0.0
    return (current_price - peak_price) / peak_price


def calculate_daily_regime(
    price_change_1d: float,
    volatility_threshold: float
) -> str:
    """
    计算当日盘整/突破状态

    Args:
        price_change_1d: 当日涨跌幅
        volatility_threshold: 动态波动阈值

    Returns:
        str: 状态标识（未突破/上涨突破/下跌突破）
    """
    if volatility_threshold == 0:
        return "未突破"

    # 判断是否突破阈值
    if abs(price_change_1d) < volatility_threshold:
        return "未突破"
    elif price_change_1d >= volatility_threshold:
        return "上涨突破"
    else:
        return "下跌突破"


def calculate_5d_regime(
    price_change_5d: float,
    volatility_threshold: float,
    sqrt_multiplier: float = math.sqrt(5)
) -> str:
    """
    计算最近5日盘整/突破状态

    Args:
        price_change_5d: 最近5日涨跌幅
        volatility_threshold: 单日动态波动阈值
        sqrt_multiplier: 5日阈值系数（默认为√5）

    Returns:
        str: 状态标识（未突破/上涨突破/下跌突破）
    """
    if volatility_threshold == 0:
        return "未突破"

    # 计算5日阈值 = √5 × 单日阈值
    threshold_5d = sqrt_multiplier * volatility_threshold

    # 判断是否突破阈值
    if abs(price_change_5d) < threshold_5d:
        return "未突破"
    elif price_change_5d >= threshold_5d:
        return "上涨突破"
    else:
        return "下跌突破"


def calculate_20d_regime(
    price_change_20d: float,
    volatility_threshold: float,
    sqrt_multiplier: float = math.sqrt(20)
) -> str:
    """
    计算最近20日盘整/突破状态

    Args:
        price_change_20d: 最近20日涨跌幅
        volatility_threshold: 单日动态波动阈值
        sqrt_multiplier: 20日阈值系数（默认为√20）

    Returns:
        str: 状态标识（未突破/上涨突破/下跌突破）
    """
    if volatility_threshold == 0:
        return "未突破"

    # 计算20日阈值 = √20 × 单日阈值
    threshold_20d = sqrt_multiplier * volatility_threshold

    # 判断是否突破阈值
    if abs(price_change_20d) < threshold_20d:
        return "未突破"
    elif price_change_20d >= threshold_20d:
        return "上涨突破"
    else:
        return "下跌突破"


def determine_action_type(
    position_before: float,
    position_after: float
) -> str:
    """
    根据仓位变化确定动作类型

    Args:
        position_before: 调仓前仓位
        position_after: 调仓后仓位

    Returns:
        str: 动作类型 (open/increase/decrease/close/hold)
    """
    epsilon = 1e-6  # 浮点数比较容差

    if abs(position_after - position_before) < epsilon:
        return "hold"
    elif position_before < epsilon and position_after > epsilon:
        return "open"
    elif position_after < epsilon and position_before > epsilon:
        return "close"
    elif position_after > position_before:
        return "increase"
    else:
        return "decrease"


def round_decimal(value: float, decimal_places: int = 4) -> float:
    """
    四舍五入到指定小数位

    Args:
        value: 原始值
        decimal_places: 小数位数

    Returns:
        float: 四舍五入后的值
    """
    d = Decimal(str(value))
    quantize_str = '0.' + '0' * decimal_places
    return float(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """
    将值限制在指定范围内

    Args:
        value: 原始值
        min_val: 最小值
        max_val: 最大值

    Returns:
        float: 限制后的值
    """
    return max(min_val, min(value, max_val))


def extend_threshold_to_horizon(
    daily_threshold: float,
    horizon: int
) -> float:
    """
    将单日阈值扩展到多日

    按照 sqrt(horizon) 进行扩展

    Args:
        daily_threshold: 单日动态阈值 θ_t
        horizon: 时间跨度（天数）

    Returns:
        float: 扩展后的阈值
    """
    return math.sqrt(horizon) * daily_threshold


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    将数值格式化为百分比字符串

    Args:
        value: 数值（如0.15表示15%）
        decimal_places: 小数位数

    Returns:
        str: 格式化后的百分比字符串
    """
    return f"{value * 100:.{decimal_places}f}%"


def format_currency(value: float, decimal_places: int = 2) -> str:
    """
    将数值格式化为货币字符串

    Args:
        value: 金额
        decimal_places: 小数位数

    Returns:
        str: 格式化后的金额字符串
    """
    return f"¥{value:,.{decimal_places}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    Args:
        numerator: 分子
        denominator: 分母
        default: 分母为0时的默认返回值

    Returns:
        float: 除法结果或默认值
    """
    if denominator == 0:
        return default
    return numerator / denominator
