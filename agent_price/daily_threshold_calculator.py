#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日动态阈值计算模块
"""

import sys
import os
from typing import Tuple, Optional
import pandas as pd

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class DailyThresholdCalculator:
    """
    每日动态阈值计算器
    
    实现方案：
    1. 为每个交易日独立计算阈值
    2. 只使用该日之前的历史数据
    3. 包含波动率计算、自适应乘数、边界约束等步骤
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化计算器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = self._get_default_config()
        self.config = config
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            # 核心阈值参数
            'vol_lookback_days': 30,          # 波动率计算窗口
            'base_multiplier': 0.35,            # 基础乘数
            
            # 绝对边界参数
            'boundary_lookback_days': 252,     # 边界计算历史窗口
            'lower_bound_quantile': 0.3,     # 下界分位数
            'upper_bound_quantile': 0.7,     # 上界分位数
            
            # 自适应调整参数
            'historical_vol_period': 60,      # 历史波动率计算周期（用于计算波动率比率）
            'high_vol_threshold': 1.4,        # 高波动阈值
            'low_vol_threshold': 0.7,         # 低波动阈值
            'high_vol_adjustment': 0.5,       # 高波动调整系数
            'low_vol_adjustment': 1.5,         # 低波动调整系数
            
            # 其他参数
            'min_data_points': 60,            # 最小数据点数要求
            'min_vol_data_points': 10         # 波动率计算最少需要的数据点
        }
    
    def calculate_volatility_regime(
        self, 
        returns_series: pd.Series, 
        current_index: int, 
        lookback_days: int
    ) -> float:
        """
        基于波动率区间的自适应调整
        返回自适应乘数
        
        Args:
            returns_series: 收益率序列
            current_index: 当前索引位置（在returns_series中的位置）
            lookback_days: 短期波动率计算窗口
        
        Returns:
            float: 自适应乘数
        """
        if current_index <= lookback_days:
            return self.config['base_multiplier']
        
        # 获取短期和长期历史数据
        short_returns = returns_series.iloc[current_index - lookback_days:current_index]
        long_returns = returns_series.iloc[:current_index]  # 所有可用历史数据
        
        # 计算波动率
        vol_short = short_returns.std()
        vol_long = long_returns.std()
        
        # 避免除零
        if vol_long == 0 or pd.isna(vol_long):
            return self.config['base_multiplier']
        
        vol_ratio = vol_short / vol_long
        
        # 根据波动率状态调整乘数
        if vol_ratio > self.config['high_vol_threshold']:
            # 高波动状态 - 收紧盘整定义
            return self.config['base_multiplier'] * self.config['high_vol_adjustment']
        elif vol_ratio < self.config['low_vol_threshold']:
            # 低波动状态 - 放宽盘整定义
            return self.config['base_multiplier'] * self.config['low_vol_adjustment']
        else:
            # 正常波动状态
            return self.config['base_multiplier']
    
    def calculate_daily_threshold(
        self, 
        price_series: pd.Series, 
        current_index: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        为特定交易日计算动态阈值
        只使用该日之前的历史数据（无未来信息）
        
        Args:
            price_series: 价格序列（pandas Series，索引为日期）
            current_index: 当前交易日的索引位置（在price_series中的位置）
        
        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]: 
            (final_threshold, adaptive_multiplier, current_volatility)
            如果数据不足，返回 (None, None, None)
        """
        # 计算收益率序列
        returns = price_series.pct_change().dropna()
        
        # 确保有足够的数据
        if current_index < self.config['min_data_points']:
            return None, None, None
        
        # 只使用当前索引之前的数据（避免未来信息）
        historical_returns = returns.iloc[:current_index - 1]  # 因为returns比prices少一个
        
        # 1. 计算当前波动率（使用最近的N天）
        lookback_days = min(self.config['vol_lookback_days'], len(historical_returns))
        if lookback_days < self.config['min_vol_data_points']:
            return None, None, None
        
        recent_returns = historical_returns.tail(lookback_days)
        current_volatility = recent_returns.std()
        
        if pd.isna(current_volatility) or current_volatility == 0:
            return None, None, None
        
        # 2. 获取自适应乘数
        # 使用lookback_days作为短期窗口，全部历史作为长期窗口
        adaptive_multiplier = self.calculate_volatility_regime(
            historical_returns, len(historical_returns), lookback_days
        )
        
        # 3. 计算核心动态阈值
        dynamic_threshold = current_volatility * adaptive_multiplier
        
        # 4. 计算绝对边界
        abs_returns = historical_returns.abs()
        boundary_days = min(self.config['boundary_lookback_days'], len(abs_returns))
        
        if boundary_days > 0:
            boundary_returns = abs_returns.tail(boundary_days)
            lower_bound = boundary_returns.quantile(self.config['lower_bound_quantile'])
            upper_bound = boundary_returns.quantile(self.config['upper_bound_quantile'])
        else:
            lower_bound = abs_returns.quantile(self.config['lower_bound_quantile'])
            upper_bound = abs_returns.quantile(self.config['upper_bound_quantile'])
        
        # 处理边界值可能为NaN的情况
        if pd.isna(lower_bound):
            lower_bound = 0.0
        if pd.isna(upper_bound):
            upper_bound = dynamic_threshold * 2.0  # 使用一个较大的默认值
        
        # 5. 应用边界约束
        final_threshold = max(lower_bound, min(dynamic_threshold, upper_bound))
        
        return final_threshold, adaptive_multiplier, current_volatility
    
    def calculate_threshold_for_date(
        self,
        price_series: pd.Series,
        target_date: pd.Timestamp
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        为指定日期计算动态阈值（便捷方法）
        
        Args:
            price_series: 价格序列（pandas Series，索引为日期）
            target_date: 目标日期
        
        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]: 
            (final_threshold, adaptive_multiplier, current_volatility)
        """
        if target_date not in price_series.index:
            return None, None, None
        
        current_index = price_series.index.get_loc(target_date) + 1  # +1因为索引从0开始
        return self.calculate_daily_threshold(price_series, current_index)


def calculate_daily_volatility_threshold(
    price_history: list,
    config: Optional[dict] = None
) -> Optional[float]:
    """
    计算每日动态波动阈值（兼容原有接口）
    
    这个函数是为了兼容 price_analyzer.py 中原有的调用方式
    它会使用最新的价格数据来计算阈值
    
    根据新方案的核心逻辑：
    - 边界约束通过 max(lower_bound, min(dynamic_threshold, upper_bound)) 保证阈值合理性
    - lower_bound 本身来自历史分位数，已经是合理的下限
    - 不需要固定的最小值兜底
    
    Args:
        price_history: 价格序列（从旧到新，长度>=2）
        config: 配置字典，如果为None则使用默认配置
    
    Returns:
        Optional[float]: 动态阈值（百分比形式，如 0.015 表示 1.5%），如果数据不足或计算失败返回 None
    """
    if len(price_history) < 2:
        # 价格历史不足，无法计算阈值
        return None
    
    # 转换为pandas Series
    price_series = pd.Series(price_history)
    
    # 创建计算器
    calculator = DailyThresholdCalculator(config)
    
    # 计算最后一个交易日的阈值
    current_index = len(price_series)
    threshold, _, _ = calculator.calculate_daily_threshold(price_series, current_index)
    
    # 如果计算失败，返回 None
    # 边界约束已经保证了阈值的合理性，不需要额外的固定最小值兜底
    return threshold

