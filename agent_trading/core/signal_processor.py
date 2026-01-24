#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号处理模块
根据方向概率和置信度计算信号分档
"""

import os
import sys
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.config import (
    SIGNAL_P_STRONG,
    SIGNAL_CONF_STRONG,
    SIGNAL_P_MEDIUM,
    SIGNAL_CONF_MEDIUM,
    SIGNAL_P_WEAK
)


def calculate_signal_level(
    prob_up: float,
    prob_down: float,
    prob_side: float,
    confidence: float
) -> str:
    """
    根据方向概率和置信度计算信号档位

    新逻辑（v2）：
    - 只使用主导概率和置信度两个维度
    - 移除边际差距判断
    - 符合预测专家提示词中的强弱信号定义
    - 档位扩展为7档（强、中、弱各3档 + 中性1档）

    档位判断标准：
    1. 主导方向为横盘(side) → 必定返回 neutral
    2. 主导方向为上涨(up)：
       - 主导概率 ≥ 0.70 且 置信度 ≥ 0.60 → strong_long
       - 主导概率 ≥ 0.60 且 置信度 ≥ 0.60 → medium_long
       - 主导概率 ≥ 0.50 → weak_long
       - 其他 → neutral
    3. 主导方向为下跌(down)：同理对称处理

    Args:
        prob_up: 上涨概率
        prob_down: 下跌概率
        prob_side: 横盘概率
        confidence: 置信度

    Returns:
        str: 信号档位 (strong_long/medium_long/weak_long/neutral/weak_short/medium_short/strong_short)
    """
    # 步骤1: 找出主导方向
    max_prob = max(prob_up, prob_down, prob_side)

    if max_prob == prob_up:
        dominant_direction = 'up'
    elif max_prob == prob_down:
        dominant_direction = 'down'
    else:
        dominant_direction = 'side'

    # 步骤2: 根据主导方向判断档位

    # 情况1: 主导方向是横盘 → 必定中性
    if dominant_direction == 'side':
        return 'neutral'

    # 情况2: 主导方向是上涨
    elif dominant_direction == 'up':
        # 强多头：主导概率 ≥ 0.70 且 置信度 ≥ 0.60
        if max_prob >= SIGNAL_P_STRONG and confidence >= SIGNAL_CONF_STRONG:
            return 'strong_long'
        # 中等多头：主导概率 ≥ 0.60 且 置信度 ≥ 0.60
        elif max_prob >= SIGNAL_P_MEDIUM and confidence >= SIGNAL_CONF_MEDIUM:
            return 'medium_long'
        # 弱多头：主导概率 ≥ 0.50
        elif max_prob >= SIGNAL_P_WEAK:
            return 'weak_long'
        # 其他情况：中性
        else:
            return 'neutral'

    # 情况3: 主导方向是下跌
    else:  # dominant_direction == 'down'
        # 强空头：主导概率 ≥ 0.70 且 置信度 ≥ 0.60
        if max_prob >= SIGNAL_P_STRONG and confidence >= SIGNAL_CONF_STRONG:
            return 'strong_short'
        # 中等空头：主导概率 ≥ 0.60 且 置信度 ≥ 0.60
        elif max_prob >= SIGNAL_P_MEDIUM and confidence >= SIGNAL_CONF_MEDIUM:
            return 'medium_short'
        # 弱空头：主导概率 ≥ 0.50
        elif max_prob >= SIGNAL_P_WEAK:
            return 'weak_short'
        # 其他情况：中性
        else:
            return 'neutral'


def process_direction_signals(prediction_data: Dict[str, Any]) -> Dict[str, str]:
    """
    处理预测数据，计算三个时间维度的信号档位

    Args:
        prediction_data: 从数据库获取的预测数据

    Returns:
        Dict: 三个时间维度的信号档位
        {
            'signal_level_T1': 'strong_long',
            'signal_level_T5': 'weak_long',
            'signal_level_T20': 'neutral'
        }
    """
    result = {}

    # T+1
    result['signal_level_T1'] = calculate_signal_level(
        prob_up=float(prediction_data.get('t1_prob_up', 0) or 0),
        prob_down=float(prediction_data.get('t1_prob_down', 0) or 0),
        prob_side=float(prediction_data.get('t1_prob_side', 0) or 0),
        confidence=float(prediction_data.get('conf_T1', 0) or 0)
    )

    # T+5
    result['signal_level_T5'] = calculate_signal_level(
        prob_up=float(prediction_data.get('t5_prob_up', 0) or 0),
        prob_down=float(prediction_data.get('t5_prob_down', 0) or 0),
        prob_side=float(prediction_data.get('t5_prob_side', 0) or 0),
        confidence=float(prediction_data.get('conf_T5', 0) or 0)
    )

    # T+20
    result['signal_level_T20'] = calculate_signal_level(
        prob_up=float(prediction_data.get('t20_prob_up', 0) or 0),
        prob_down=float(prediction_data.get('t20_prob_down', 0) or 0),
        prob_side=float(prediction_data.get('t20_prob_side', 0) or 0),
        confidence=float(prediction_data.get('conf_T20', 0) or 0)
    )

    return result


def get_signal_description(signal_level: str) -> str:
    """
    获取信号档位的中文描述

    Args:
        signal_level: 信号档位

    Returns:
        str: 中文描述
    """
    descriptions = {
        'strong_long': '强多',
        'medium_long': '中多',
        'weak_long': '弱多',
        'neutral': '中性',
        'weak_short': '弱空',
        'medium_short': '中空',
        'strong_short': '强空'
    }
    return descriptions.get(signal_level, '未知')
