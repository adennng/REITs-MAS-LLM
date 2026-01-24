#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控拦截模块
在LLM输出后执行风控检查，必要时override决策
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.config import (
    K_MAX,
    STEP,
    POSITION_MAX,
    HARD_DD_LIMIT
)
from agent_trading.utils.helpers import clip_value
from agent_trading.utils.logger import get_logger

logger = get_logger()


class RiskController:
    """
    风控拦截器

    在LLM输出后执行风控检查，必要时调整delta_steps
    """

    def check_and_override(
        self,
        delta_steps: int,
        state: Dict[str, Any]
    ) -> Tuple[int, Optional[str]]:
        """
        执行风控检查，必要时override决策

        Args:
            delta_steps: LLM输出的调仓步数
            state: 完整的state_t

        Returns:
            Tuple[int, Optional[str]]: (调整后的delta_steps, 拦截原因)
        """
        account = state['account_state']

        delta_steps_final = delta_steps
        override_reason = None

        # 1. 检查delta_steps范围（允许±10作为清仓/满仓指令）
        if delta_steps <= -10 or delta_steps >= 10:
            # ±10是特殊指令（清仓/满仓），不需要调整
            logger.info(f"检测到特殊指令: delta_steps={delta_steps} "
                       f"({'清仓指令' if delta_steps <= -10 else '满仓指令'})")
        elif delta_steps < -K_MAX or delta_steps > K_MAX:
            # 超出正常范围且不是特殊指令，需要调整
            delta_steps_final = int(clip_value(delta_steps, -K_MAX, K_MAX))
            override_reason = f"exceed_max_steps: {delta_steps} -> {delta_steps_final}"
            logger.warning(f"风控: delta_steps超出范围，调整为{delta_steps_final}")

        # 2. 硬止损检查（最高优先级）- 强制清仓
        if (account.get('round_max_drawdown') is not None and
            account['round_max_drawdown'] < HARD_DD_LIMIT and
            account['position'] > 0.001):

            # 计算需要清仓的步数（受限于K_MAX，可能需要多天清完）
            current_pos = account['position']
            steps_to_clear = -int(current_pos / STEP)  # 例如：100%需要-5步，80%需要-4步
            delta_steps_final = max(steps_to_clear, -K_MAX)  # 限制在[-K_MAX, 0]范围内

            override_reason = f"hard_stop_loss: round_dd={account['round_max_drawdown']:.4f}, liquidating"
            logger.warning(
                f"风控: 触发硬止损（本轮回撤{account['round_max_drawdown']:.2%}），"
                f"强制清仓 delta_steps={delta_steps_final} "
                f"(当前仓位{current_pos:.0%} → 目标{max(0, current_pos + delta_steps_final * STEP):.0%})"
            )

        # 注：建仓期仓位上限检查已删除
        # 理由：建仓期判断条件为 (天数<=20 AND 仓位<0.6)
        #      如果仓位达到0.6，下次循环时系统会自动判断为非建仓期
        #      因此不需要在风控中限制建仓期仓位上限

        # 注：最大仓位检查已删除
        # 理由：尊重LLM的delta_steps决策意图，由calculate_target_position()中的clip自然处理
        #      例如：LLM想+2加仓40%，即使会超过100%，系统也会在执行时自动调整到100%

        # 注：仓位下限检查已删除
        # 理由：同上，由calculate_target_position()中的clip自然处理
        #      例如：LLM想-2减仓40%，即使会导致负仓位，系统会在执行时自动调整到0%

        return delta_steps_final, override_reason

    def calculate_target_position(
        self,
        delta_steps: int,
        current_position: float
    ) -> float:
        """
        根据delta_steps计算目标仓位

        Args:
            delta_steps: 调仓步数（支持特殊指令：-10=清仓, +10=满仓）
            current_position: 当前仓位

        Returns:
            float: 目标仓位
        """
        # 检查是否为清仓指令
        if delta_steps <= -10:
            logger.info(f"执行清仓指令: 目标仓位 0%")
            return 0.0

        # 检查是否为满仓指令
        if delta_steps >= 10:
            logger.info(f"执行满仓指令: 目标仓位 {POSITION_MAX:.0%}")
            return POSITION_MAX

        # 正常计算目标仓位
        raw_target = current_position + delta_steps * STEP
        return clip_value(raw_target, 0.0, POSITION_MAX)
