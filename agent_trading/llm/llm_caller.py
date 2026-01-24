#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM调用模块
封装deepseek-reasoner等模型的调用
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
from openai import OpenAI

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from config.model_config import MODEL_CONFIG
from agent_trading.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_RETRIES,
    K_MAX
)
from agent_trading.utils.logger import get_logger

logger = get_logger()


def clean_json_response(content: str) -> str:
    """
    清理LLM返回的内容，去除markdown代码块标记并处理JSON格式问题

    Args:
        content: LLM返回的原始内容

    Returns:
        str: 清理后的JSON字符串
    """
    import re

    # 去除前后空白
    content = content.strip()

    # 检查是否被markdown代码块包裹
    if content.startswith('```'):
        # 去除开头的 ```json 或 ```
        lines = content.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]  # 去除第一行

        # 去除结尾的 ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]  # 去除最后一行

        content = '\n'.join(lines)

    # 处理JSON中带正号的数字（如 ": +1" → ": 1"）
    # 匹配模式：冒号+空格+正号+数字
    content = re.sub(r':\s*\+(\d+)', r': \1', content)

    return content.strip()


def validate_trading_response(response_json: Dict[str, Any]) -> Tuple[bool, str]:
    """
    验证LLM响应格式

    Args:
        response_json: LLM响应的JSON

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    # 检查必需字段
    if 'delta_steps' not in response_json:
        return False, "缺少delta_steps字段"

    delta_steps = response_json['delta_steps']

    # 检查delta_steps是否为整数
    if not isinstance(delta_steps, int):
        return False, f"delta_steps必须是整数，当前类型: {type(delta_steps)}"

    # 检查范围：只允许7个特定值（-10清仓, -2/-1/0/+1/+2正常调仓, +10满仓）
    allowed_values = {-10, -2, -1, 0, 1, 2, 10}
    if delta_steps not in allowed_values:
        return False, f"delta_steps必须是以下7个值之一：-10, -2, -1, 0, +1, +2, +10，当前值: {delta_steps}"

    # 检查rationale_core字段（建议有）
    if 'rationale_core' not in response_json:
        return False, "缺少rationale_core字段"

    if not isinstance(response_json['rationale_core'], str):
        return False, "rationale_core必须是字符串"

    return True, ""


def call_trading_llm(
    prompt: str,
    max_retries: int = None,
    show_reasoning: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    调用LLM进行决策

    Args:
        prompt: 完整提示词
        max_retries: 最大重试次数
        show_reasoning: 是否在日志中显示推理过程

    Returns:
        Tuple[bool, Dict]: (是否成功, 结果)
        成功时返回: {
            'delta_steps': int,
            'rationale_core': str,
            'reasoning': str,
            'raw_response': str,
            'prompt': str
        }
        失败时返回: {
            'error': str
        }
    """
    if max_retries is None:
        max_retries = LLM_MAX_RETRIES

    # 获取模型配置
    model_config = MODEL_CONFIG[LLM_PROVIDER][LLM_MODEL]

    # 初始化客户端
    client = OpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )

    logger.debug(f"调用LLM: {LLM_PROVIDER}/{LLM_MODEL}")
    logger.debug(f"Prompt长度: {len(prompt)}字符")

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  第 {attempt} 次调用LLM...")

            # 调用模型
            response = client.chat.completions.create(
                model=model_config['model'],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE
            )

            # 提取推理过程和最终内容
            reasoning_content = ""
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning_content = response.choices[0].message.reasoning_content or ""

            final_content = response.choices[0].message.content

            # 记录推理过程到日志文件
            if reasoning_content:
                logger.debug("=" * 80)
                logger.debug("LLM推理过程：")
                logger.debug("=" * 80)
                logger.debug(reasoning_content)
                logger.debug("=" * 80)

                # 如果需要在控制台显示
                if show_reasoning:
                    logger.info("LLM推理过程（摘要）：")
                    # 只显示前500字符
                    if len(reasoning_content) > 500:
                        logger.info(reasoning_content[:500] + "...")
                    else:
                        logger.info(reasoning_content)

            logger.info("  LLM调用成功，开始解析响应...")

            # 解析JSON
            try:
                # 清理LLM返回内容，去除markdown代码块标记
                cleaned_content = clean_json_response(final_content)

                # 解析JSON
                response_json = json.loads(cleaned_content)

            except json.JSONDecodeError as e:
                logger.error(f"  ✗ JSON解析失败: {e}")
                logger.debug(f"  原始响应: {final_content}")
                last_error = f"JSON解析失败: {e}"
                if attempt < max_retries:
                    logger.info(f"  准备进行第 {attempt + 1} 次尝试...")
                    continue
                else:
                    return False, {'error': last_error}

            # 验证响应格式
            is_valid, error_msg = validate_trading_response(response_json)

            if is_valid:
                logger.info(f"  ✓ 响应验证通过（第 {attempt} 次尝试）")
                result = {
                    'delta_steps': response_json['delta_steps'],
                    'rationale_core': response_json.get('rationale_core', ''),
                    'reasoning': reasoning_content,
                    'raw_response': cleaned_content,  # 保存清理后的纯JSON，而不是原始响应
                    'prompt': prompt
                }
                return True, result
            else:
                logger.warning(f"  ✗ 响应验证失败（第 {attempt} 次尝试）: {error_msg}")
                last_error = f"响应验证失败: {error_msg}"

                if attempt < max_retries:
                    logger.info(f"  准备进行第 {attempt + 1} 次尝试...")
                else:
                    logger.error(f"  已达最大重试次数（{max_retries}次）")
                    return False, {'error': last_error}

        except Exception as e:
            logger.error(f"  第 {attempt} 次调用LLM失败: {e}")
            last_error = str(e)

            if attempt < max_retries:
                logger.info(f"  准备进行第 {attempt + 1} 次尝试...")
            else:
                logger.error(f"  已达最大重试次数（{max_retries}次）")
                return False, {'error': last_error}

    return False, {'error': last_error}
