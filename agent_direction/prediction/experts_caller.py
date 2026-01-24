#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四专家调用模块
负责调用四个专家进行分析（支持异步并行和同步顺序两种模式）
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

# 导入四个专家模块
from agent_announcement.announcement_recent_impact import analyze_recent_announcements  # noqa: E402
from agent_market_analysis.market_analyzer import MarketAnalyzer  # noqa: E402
from agent_price.price_analyzer import analyze_price  # noqa: E402
from agent_event.event_analyzer import analyze_event  # noqa: E402


# 获取logger
logger = logging.getLogger(__name__)


def call_announcement_expert(fund_code: str, date: str) -> Dict[str, Any]:
    """
    调用公告专家

    Args:
        fund_code: 基金代码
        date: 分析日期

    Returns:
        Dict: 专家返回结果
    """
    logger.info(f"  调用公告专家 - 基金: {fund_code}, 日期: {date}")
    try:
        result = analyze_recent_announcements(
            fund_code=fund_code,
            date=date,
            include_reasoning=False  # 不需要推理过程
        )

        if result['status'] == 'success':
            # 解析analysis_result为JSON
            analysis_json = json.loads(result['analysis_result'])
            logger.info(f"  ✓ 公告专家调用成功")
            return {
                'status': 'success',
                'expert_type': 'announcement',
                'result': analysis_json
            }
        else:
            logger.error(f"  ✗ 公告专家调用失败: {result.get('message', '未知错误')}")
            return {
                'status': 'error',
                'expert_type': 'announcement',
                'message': result.get('message', '未知错误')
            }

    except Exception as e:
        logger.error(f"  ✗ 公告专家调用异常: {e}")
        return {
            'status': 'error',
            'expert_type': 'announcement',
            'message': str(e)
        }


def call_market_expert(fund_code: str, date: str) -> Dict[str, Any]:
    """
    调用市场专家

    Args:
        fund_code: 基金代码
        date: 分析日期

    Returns:
        Dict: 专家返回结果
    """
    logger.info(f"  调用市场专家 - 基金: {fund_code}, 日期: {date}")
    try:
        analyzer = MarketAnalyzer()
        result = analyzer.analyze(
            fund_code=fund_code,
            analysis_date=date,
            fund_name=None
        )

        if result['status'] == 'success':
            # 解析analysis_result为JSON
            analysis_json = json.loads(result['analysis_result'])
            logger.info(f"  ✓ 市场专家调用成功")
            return {
                'status': 'success',
                'expert_type': 'market',
                'result': analysis_json
            }
        else:
            logger.error(f"  ✗ 市场专家调用失败: {result.get('message', '未知错误')}")
            return {
                'status': 'error',
                'expert_type': 'market',
                'message': result.get('message', '未知错误')
            }

    except Exception as e:
        logger.error(f"  ✗ 市场专家调用异常: {e}")
        return {
            'status': 'error',
            'expert_type': 'market',
            'message': str(e)
        }


def call_price_expert(fund_code: str, date: str) -> Dict[str, Any]:
    """
    调用价格专家

    Args:
        fund_code: 基金代码
        date: 分析日期

    Returns:
        Dict: 专家返回结果
    """
    logger.info(f"  调用价格专家 - 基金: {fund_code}, 日期: {date}")
    try:
        result = analyze_price(
            fund_code=fund_code,
            date=date
        )

        if result['status'] == 'success':
            # 解析analysis_result为JSON
            analysis_json = json.loads(result['analysis_result'])
            logger.info(f"  ✓ 价格专家调用成功")
            return {
                'status': 'success',
                'expert_type': 'price',
                'result': analysis_json
            }
        else:
            logger.error(f"  ✗ 价格专家调用失败: {result.get('message', '未知错误')}")
            return {
                'status': 'error',
                'expert_type': 'price',
                'message': result.get('message', '未知错误')
            }

    except Exception as e:
        logger.error(f"  ✗ 价格专家调用异常: {e}")
        return {
            'status': 'error',
            'expert_type': 'price',
            'message': str(e)
        }


def call_event_expert(fund_code: str, date: str) -> Dict[str, Any]:
    """
    调用事件专家

    Args:
        fund_code: 基金代码
        date: 分析日期

    Returns:
        Dict: 专家返回结果
    """
    logger.info(f"  调用事件专家 - 基金: {fund_code}, 日期: {date}")
    try:
        result = analyze_event(
            fund_code=fund_code,
            date=date
        )

        if result['status'] == 'success':
            # 解析analysis_result为JSON
            analysis_json = json.loads(result['analysis_result'])
            logger.info(f"  ✓ 事件专家调用成功")
            return {
                'status': 'success',
                'expert_type': 'event',
                'result': analysis_json
            }
        else:
            logger.error(f"  ✗ 事件专家调用失败: {result.get('message', '未知错误')}")
            return {
                'status': 'error',
                'expert_type': 'event',
                'message': result.get('message', '未知错误')
            }

    except Exception as e:
        logger.error(f"  ✗ 事件专家调用异常: {e}")
        return {
            'status': 'error',
            'expert_type': 'event',
            'message': str(e)
        }


def call_experts_async(
    fund_code: str,
    date: str,
    timeout: int = 300
) -> Tuple[bool, Dict[str, Any]]:
    """
    异步并行调用四个专家

    Args:
        fund_code: 基金代码
        date: 分析日期
        timeout: 超时时间（秒）

    Returns:
        Tuple[bool, Dict]: (是否全部成功, 结果字典)
            结果字典: {
                'announcement': {...},
                'market': {...},
                'price': {...},
                'event': {...},
                'errors': []  # 失败的专家列表
            }
    """
    logger.info(f"开始异步调用四个专家 - 基金: {fund_code}, 日期: {date}")

    # 定义要调用的专家函数
    expert_functions = {
        'announcement': call_announcement_expert,
        'market': call_market_expert,
        'price': call_price_expert,
        'event': call_event_expert
    }

    results = {}
    errors = []

    # 使用ThreadPoolExecutor进行并行调用
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_expert = {
            executor.submit(func, fund_code, date): expert_name
            for expert_name, func in expert_functions.items()
        }

        # 等待所有任务完成
        for future in as_completed(future_to_expert, timeout=timeout):
            expert_name = future_to_expert[future]
            try:
                expert_result = future.result(timeout=timeout)

                if expert_result['status'] == 'success':
                    results[expert_name] = expert_result['result']
                else:
                    logger.error(f"  {expert_name}专家失败: {expert_result.get('message', '未知错误')}")
                    errors.append(expert_name)

            except TimeoutError:
                logger.error(f"  {expert_name}专家超时")
                errors.append(expert_name)
            except Exception as e:
                logger.error(f"  {expert_name}专家异常: {e}")
                errors.append(expert_name)

    # 检查是否全部成功
    all_success = len(errors) == 0

    if all_success:
        logger.info("✓ 四个专家全部调用成功")
    else:
        logger.warning(f"✗ 有{len(errors)}个专家调用失败: {errors}")

    return all_success, {
        **results,
        'errors': errors
    }


def call_experts_sync(
    fund_code: str,
    date: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    同步顺序调用四个专家

    Args:
        fund_code: 基金代码
        date: 分析日期

    Returns:
        Tuple[bool, Dict]: (是否全部成功, 结果字典)
    """
    logger.info(f"开始同步调用四个专家 - 基金: {fund_code}, 日期: {date}")

    results = {}
    errors = []

    # 按顺序调用四个专家
    experts = [
        ('announcement', call_announcement_expert),
        ('market', call_market_expert),
        ('price', call_price_expert),
        ('event', call_event_expert)
    ]

    for expert_name, expert_func in experts:
        try:
            expert_result = expert_func(fund_code, date)

            if expert_result['status'] == 'success':
                results[expert_name] = expert_result['result']
            else:
                logger.error(f"  {expert_name}专家失败: {expert_result.get('message', '未知错误')}")
                errors.append(expert_name)

        except Exception as e:
            logger.error(f"  {expert_name}专家异常: {e}")
            errors.append(expert_name)

    # 检查是否全部成功
    all_success = len(errors) == 0

    if all_success:
        logger.info("✓ 四个专家全部调用成功")
    else:
        logger.warning(f"✗ 有{len(errors)}个专家调用失败: {errors}")

    return all_success, {
        **results,
        'errors': errors
    }
