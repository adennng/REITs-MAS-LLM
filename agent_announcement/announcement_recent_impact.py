#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
近期公告影响分析工具
分析最近7日内发布的公告对REITs价格的潜在影响，为交易决策提供支持
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pymysql
import pandas as pd
from openai import OpenAI

# 导入配置（使用相对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 以下导入依赖于上面的 sys.path 设置，必须在此之后导入
from config.db_config import get_db_reits_config  # noqa: E402
from config.model_config import MODEL_CONFIG  # noqa: E402
from agent_price.daily_threshold_calculator import calculate_daily_volatility_threshold  # noqa: E402

# 导入历史影响分析模块
from agent_announcement.announcement_historical_impact import analyze_announcement_impact  # noqa: E402
from agent_announcement.event_historical_impact_analyzer import analyze_event_impact  # noqa: E402


# ============================================================================
# 配置参数 - 可手动修改
# ============================================================================
# 日志模式
LOG_MODE = 'detailed'  # 'detailed' 或 'simple'
# detailed: 记录全过程(入参、数据库查询结果、历史分析、LLM调用等)
# simple: 只记录入参、输出、执行状态

# 需要获取历史影响分析的公告类型
DOC_TYPES_NEED_HISTORICAL = [
    "季报", "年报", "中报", "基金份额解除限售", "分红公告",
    "主要运营数据", "权益变动", "交易情况提示"
]

# 查询公告的天数范围
DAYS_RANGE = 7

# 查询价格的交易日数量
PRICE_DAYS = 6  # 获取6个交易日，计算5个涨幅

# 查询事件的天数范围（当前日期前后N天）
EVENT_DAYS_RANGE = 5  # 查询当前日期前5天到后5天范围内的事件

# 需要关注的事件类型
EVENT_TYPES = ['分红公告', '基金份额解除限售']
# ============================================================================


# 配置日志
def setup_logger(log_mode: str = LOG_MODE) -> logging.Logger:
    """
    配置日志记录器

    Args:
        log_mode: 日志模式 ('detailed' 或 'simple')

    Returns:
        Logger: 配置好的日志记录器
    """
    logger = logging.getLogger('AnnouncementRecentImpact')
    logger.setLevel(logging.DEBUG if log_mode == 'detailed' else logging.INFO)

    # 清除已有的处理器（避免重复）
    logger.handlers.clear()

    # 确定日志目录（相对路径：../log/）
    log_dir = os.path.join(parent_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = 'announcement_recent_impact.log'
    log_path = os.path.join(log_dir, log_file)

    # 文件处理器
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG if log_mode == 'detailed' else logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger()


def format_recent_5d_changes(recent_5d_changes: List[Dict], volatility_threshold: float = None) -> str:
    """
    格式化最近5日涨跌幅，标注是否突破整盘阈值

    Args:
        recent_5d_changes: 最近5日涨跌幅列表，每项包含 date 和 change_pct
        volatility_threshold: 动态波动阈值（小数形式，如0.015表示1.5%）

    Returns:
        格式化的字符串
    """
    if not recent_5d_changes:
        return "- 数据不足，无法计算最近5日涨跌幅\n"

    if volatility_threshold is None:
        # 如果阈值计算失败，仅显示涨跌幅
        result = ""
        for idx, item in enumerate(recent_5d_changes, 1):
            result += f"- T-{6-idx} ({item['date']}): {item['change_pct']:+.2f}%\n"
        return result

    # 标注是否突破阈值
    result = ""
    threshold_pct = volatility_threshold * 100
    breakthrough_count = 0

    for idx, item in enumerate(recent_5d_changes, 1):
        change_pct = item['change_pct']
        abs_change = abs(change_pct)

        if abs_change > threshold_pct:
            status = f"突破整盘阈值{threshold_pct:.2f}%，显著{'上涨' if change_pct > 0 else '下跌'}"
            breakthrough_count += 1
        else:
            status = f"未突破整盘阈值{threshold_pct:.2f}%"

        result += f"- T-{6-idx} ({item['date']}): {change_pct:+.2f}%（{status}）\n"

    # 添加总结
    result += f"\n小结：最近5个交易日中，{breakthrough_count}天突破整盘阈值，{5-breakthrough_count}天处于整盘状态"

    return result


def get_fund_price_data(fund_code: str, end_date: str, days: int = 150) -> pd.DataFrame:
    """
    获取指定基金的历史价格数据（用于计算动态阈值和最近涨跌幅）

    Args:
        fund_code: 基金代码
        end_date: 截止日期 (格式: YYYY-MM-DD)
        days: 往前获取的自然日天数

    Returns:
        DataFrame: 包含 trade_date, close, vol 的数据框，按日期升序排列
    """
    logger.debug(f"开始查询基金 {fund_code} 的价格数据，截止日期：{end_date}，往前{days}天")

    config = get_db_reits_config()
    conn = pymysql.connect(**config)

    try:
        # 计算起始日期
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days)
        start_date = start_dt.strftime('%Y-%m-%d')

        sql = """
            SELECT trade_date, close, vol
            FROM price_data
            WHERE fund_code = %s
              AND trade_date >= %s
              AND trade_date <= %s
            ORDER BY trade_date ASC
        """

        df = pd.read_sql(sql, conn, params=(fund_code, start_date, end_date))

        logger.debug(f"查询到 {len(df)} 条价格数据")
        if logger.level == logging.DEBUG and len(df) > 0:
            logger.debug(f"数据范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")

        return df

    finally:
        conn.close()


def validate_date_format(date_str: str) -> bool:
    """
    验证日期格式是否符合 YYYY-MM-DD

    Args:
        date_str: 日期字符串

    Returns:
        bool: 格式是否正确
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def query_recent_announcements(
    fund_code: str,
    date: str,
    days_range: int = DAYS_RANGE
) -> List[Dict[str, Any]]:
    """
    查询指定日期前N天内的公告

    Args:
        fund_code: 基金代码
        date: 当前日期 (格式: YYYY-MM-DD)
        days_range: 查询天数范围

    Returns:
        List[Dict]: 公告记录列表
    """
    logger.info(f"开始查询近{days_range}天内的公告 - fund_code: {fund_code}, date: {date}")

    # 计算起始日期
    current_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = (current_date - timedelta(days=days_range)).strftime('%Y-%m-%d')

    logger.info(f"查询日期范围: {start_date} 至 {date}")

    # 获取数据库配置
    db_config = get_db_reits_config()

    # 连接数据库
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT date, announcement_title, digest, doc_type_2, sentiment, sentiment_reason
                FROM processed_files
                WHERE fund_code = %s
                  AND digest IS NOT NULL
                  AND digest != ''
                  AND date > %s
                  AND date < %s
                  AND (sentiment IS NULL OR sentiment != 100)
                ORDER BY date DESC
            """

            cursor.execute(sql, (fund_code, start_date, date))
            results = cursor.fetchall()

            logger.info(f"查询完成，共找到 {len(results)} 条公告记录")

            if LOG_MODE == 'detailed':
                logger.debug(f"查询SQL: {sql}")
                logger.debug(f"查询参数: fund_code={fund_code}, start_date={start_date}, date={date}")
                logger.debug(f"查询结果: {json.dumps(results, ensure_ascii=False, indent=2, default=str)}")

            return results

    finally:
        connection.close()


def get_historical_impact_for_doc_types(
    fund_code: str,
    date: str,
    doc_types: List[str]
) -> Dict[str, Any]:
    """
    获取特定公告类型的历史影响分析

    注意：年报和中报会合并查询

    Args:
        fund_code: 基金代码
        date: 当前日期
        doc_types: 需要查询的公告类型列表

    Returns:
        Dict: {doc_type: analysis_result}
    """
    logger.info(f"开始获取历史影响分析 - doc_types: {doc_types}")

    historical_results = {}

    # 处理年报/中报合并查询
    processed_types = set()

    for doc_type in doc_types:
        # 如果已经处理过（比如年报和中报合并处理），跳过
        if doc_type in processed_types:
            continue

        # 年报和中报合并查询
        if doc_type in ['年报', '中报']:
            if '年报' not in processed_types and '中报' not in processed_types:
                logger.info("查询年报和中报的历史影响（合并查询）")
                result = analyze_announcement_impact(
                    fund_code=fund_code,
                    date=date,
                    doc_type_2=['年报', '中报']
                )
                historical_results['年报/中报'] = result
                processed_types.add('年报')
                processed_types.add('中报')

                if LOG_MODE == 'detailed':
                    logger.debug(f"年报/中报历史分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            # 其他类型单独查询
            logger.info(f"查询 {doc_type} 的历史影响")
            result = analyze_announcement_impact(
                fund_code=fund_code,
                date=date,
                doc_type_2=doc_type
            )
            historical_results[doc_type] = result
            processed_types.add(doc_type)

            if LOG_MODE == 'detailed':
                logger.debug(f"{doc_type} 历史分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    logger.info(f"历史影响分析完成，共获取 {len(historical_results)} 种类型")

    return historical_results


def get_recent_price_changes(
    fund_code: str,
    date: str,
    days: int = PRICE_DAYS
) -> Dict[str, Any]:
    """
    获取最近N个交易日的价格数据及涨幅

    Args:
        fund_code: 基金代码
        date: 当前日期 (格式: YYYY-MM-DD)
        days: 交易日数量

    Returns:
        Dict: {
            '交易日数据': [
                {'日期': 'xxx', '收盘价': xxx, '涨幅': 'xxx%'},
                ...
            ]
        }
    """
    logger.info(f"开始获取最近{days}个交易日的价格数据 - fund_code: {fund_code}, date: {date}")

    # 获取数据库配置
    db_config = get_db_reits_config()

    # 连接数据库
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT trade_date, close
                FROM price_data
                WHERE fund_code = %s
                  AND trade_date < %s
                ORDER BY trade_date DESC
                LIMIT %s
            """

            cursor.execute(sql, (fund_code, date, days))
            results = cursor.fetchall()

            logger.info(f"查询完成，共找到 {len(results)} 个交易日数据")

            if LOG_MODE == 'detailed':
                logger.debug(f"查询SQL: {sql}")
                logger.debug(f"查询参数: fund_code={fund_code}, date={date}, days={days}")
                logger.debug(f"原始价格数据: {json.dumps(results, ensure_ascii=False, default=str)}")

            if not results:
                logger.warning("未找到价格数据")
                return {'交易日数据': []}

            # 计算涨幅（需要反转列表，使其按时间正序）
            results_reversed = list(reversed(results))
            price_data = []

            for i, record in enumerate(results_reversed):
                data_item = {
                    '日期': record['trade_date'].strftime('%Y-%m-%d'),
                    '收盘价': float(record['close'])
                }

                # 计算涨幅（相对于前一个交易日）
                if i > 0:
                    prev_close = float(results_reversed[i-1]['close'])
                    current_close = float(record['close'])
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    data_item['涨幅'] = f"{change_pct:.2f}%"
                else:
                    # 第一个交易日（最早的）没有涨幅
                    data_item['涨幅'] = 'N/A'

                price_data.append(data_item)

            # 再反转回来，使最近的在前面
            price_data = list(reversed(price_data))

            result = {'交易日数据': price_data}

            logger.info(f"价格数据处理完成，包含{len(price_data)}个交易日")

            if LOG_MODE == 'detailed':
                logger.debug(f"处理后的价格数据: {json.dumps(result, ensure_ascii=False, indent=2)}")

            return result

    finally:
        connection.close()


def query_recent_events(
    fund_code: str,
    date: str,
    days_range: int = EVENT_DAYS_RANGE
) -> List[Dict[str, Any]]:
    """
    查询当前日期前后N天范围内的事件（分红、基金份额解除限售）

    Args:
        fund_code: 基金代码
        date: 当前日期 (格式: YYYY-MM-DD)
        days_range: 查询天数范围（前后N天）

    Returns:
        List[Dict]: 事件记录列表（已去重）
    """
    logger.info(f"开始查询近期事件（当前日期前后{days_range}天） - fund_code: {fund_code}, date: {date}")

    # 计算日期范围
    current_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = (current_date - timedelta(days=days_range)).strftime('%Y-%m-%d')
    end_date = (current_date + timedelta(days=days_range)).strftime('%Y-%m-%d')

    logger.info(f"事件查询日期范围: {start_date} 至 {end_date}")

    # 获取数据库配置
    db_config = get_db_reits_config()

    # 连接数据库
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # 构建 IN 子句的占位符
            placeholders = ', '.join(['%s'] * len(EVENT_TYPES))

            sql = f"""
                SELECT event_date, doc_type_2, digest, date, sentiment, sentiment_reason
                FROM processed_files
                WHERE fund_code = %s
                  AND doc_type_2 IN ({placeholders})
                  AND event_date IS NOT NULL
                  AND event_date >= %s
                  AND event_date <= %s
                  AND digest IS NOT NULL
                  AND digest != ''
                  AND (sentiment IS NULL OR sentiment != 100)
                ORDER BY event_date DESC, date DESC
            """

            # 构建参数
            params = [fund_code] + EVENT_TYPES + [start_date, end_date]
            cursor.execute(sql, params)
            results = cursor.fetchall()

            logger.info(f"查询完成，共找到 {len(results)} 条事件记录")

            if LOG_MODE == 'detailed':
                logger.debug(f"查询SQL: {sql}")
                logger.debug(f"查询参数: fund_code={fund_code}, event_types={EVENT_TYPES}, start_date={start_date}, end_date={end_date}")
                logger.debug(f"查询结果: {json.dumps(results, ensure_ascii=False, indent=2, default=str)}")

            # 去重：按event_date去重，保留date（公告发布日期）最晚的
            unique_events = {}
            for record in results:
                event_date_str = str(record['event_date'])
                # 如果该event_date还没有记录，或者当前记录的date更晚，则更新
                if event_date_str not in unique_events:
                    unique_events[event_date_str] = record
                else:
                    # 比较date字段，保留更晚的
                    existing_date = unique_events[event_date_str]['date']
                    current_date_val = record['date']
                    if current_date_val > existing_date:
                        unique_events[event_date_str] = record

            # 转换回列表，按event_date倒序排列
            deduplicated_events = sorted(
                unique_events.values(),
                key=lambda x: x['event_date'],
                reverse=True
            )

            if len(deduplicated_events) < len(results):
                logger.info(f"去重后剩余 {len(deduplicated_events)} 条事件记录（去除了 {len(results) - len(deduplicated_events)} 条重复记录）")

            if LOG_MODE == 'detailed':
                logger.debug(f"去重后的事件列表: {json.dumps(deduplicated_events, ensure_ascii=False, indent=2, default=str)}")

            return deduplicated_events

    finally:
        connection.close()


def get_historical_impact_for_events(
    fund_code: str,
    date: str,
    events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    获取每个事件的历史影响分析

    Args:
        fund_code: 基金代码
        date: 当前日期
        events: 事件列表

    Returns:
        List[Dict]: 包含事件信息和历史分析的列表
    """
    logger.info(f"开始获取事件的历史影响分析 - 共{len(events)}个事件")

    enriched_events = []

    for event in events:
        event_type = event['doc_type_2']
        event_date = str(event['event_date'])

        logger.info(f"查询事件 {event_type}（{event_date}）的历史影响")

        # 调用事件历史影响分析工具
        historical_result = analyze_event_impact(
            fund_code=fund_code,
            date=date,
            doc_type_2=event_type,
            return_reasoning=False,
            log_level='simple'  # 使用简单日志避免过多输出
        )

        # 构建enriched事件数据
        enriched_event = {
            'event_date': event_date,
            'event_type': event_type,
            'digest': event['digest'],
            'publish_date': str(event['date']),  # 公告发布日期
            'sentiment': event.get('sentiment', 0),  # 情绪判断，默认为0（中性）
            'sentiment_reason': event.get('sentiment_reason', ''),  # 情绪理由
            'historical_analysis': historical_result
        }

        enriched_events.append(enriched_event)

        if LOG_MODE == 'detailed':
            logger.debug(f"{event_type}（{event_date}）历史分析结果: {json.dumps(historical_result, ensure_ascii=False, indent=2)}")

    logger.info(f"事件历史影响分析完成，共获取 {len(enriched_events)} 个事件")

    return enriched_events


def clean_json_response(content: str) -> str:
    """
    清理LLM返回的内容，去除markdown代码块标记

    Args:
        content: LLM返回的原始内容

    Returns:
        str: 清理后的JSON字符串
    """
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

    return content.strip()


def validate_probability_sum(analysis_json_str: str) -> Tuple[bool, str]:
    """
    验证direction_judgment中所有期限的概率和是否等于1.0

    Args:
        analysis_json_str: LLM返回的JSON字符串

    Returns:
        Tuple[bool, str]: (验证是否通过, 错误信息)
    """
    try:
        # 清理markdown代码块标记
        cleaned_json = clean_json_response(analysis_json_str)

        # 解析JSON
        data = json.loads(cleaned_json)

        # 检查是否是特殊情况（无数据或全中性）
        if data.get('no_data_or_neutral', False):
            logger.debug("检测到no_data_or_neutral标记，跳过概率和验证")
            return True, ""

        # 检查是否有direction_judgment字段
        if 'direction_judgment' not in data:
            return False, "缺少direction_judgment字段"

        direction_judgment = data['direction_judgment']

        # 定义需要检查的期限
        periods_to_check = []

        # 检查short_term中的T+1和T+5
        if 'short_term' in direction_judgment:
            short_term = direction_judgment['short_term']
            if 'T+1' in short_term:
                periods_to_check.append(('short_term.T+1', short_term['T+1']))
            if 'T+5' in short_term:
                periods_to_check.append(('short_term.T+5', short_term['T+5']))

        # 检查mid_long_term
        if 'mid_long_term' in direction_judgment:
            periods_to_check.append(('mid_long_term', direction_judgment['mid_long_term']))

        # 验证每个期限的概率和
        tolerance = 0.001  # 容忍误差
        for period_name, period_data in periods_to_check:
            if 'direction_probs' not in period_data:
                return False, f"{period_name}缺少direction_probs字段"

            probs = period_data['direction_probs']

            # 检查必需的键
            if 'up' not in probs or 'down' not in probs or 'side' not in probs:
                return False, f"{period_name}的direction_probs缺少必需的键(up/down/side)"

            # 计算概率和
            prob_sum = probs['up'] + probs['down'] + probs['side']

            # 验证概率和是否等于1.0
            if abs(prob_sum - 1.0) > tolerance:
                error_msg = f"{period_name}的概率和不等于1.0: up={probs['up']}, down={probs['down']}, side={probs['side']}, sum={prob_sum:.6f}"
                logger.warning(error_msg)
                return False, error_msg

        logger.debug(f"概率和验证通过，共检查了{len(periods_to_check)}个期限")
        return True, ""

    except json.JSONDecodeError as e:
        return False, f"JSON解析失败: {str(e)}"
    except Exception as e:
        return False, f"验证过程发生错误: {str(e)}"


def validate_dominant_probability(analysis_json_str: str, min_dominant_prob: float = 0.5) -> Tuple[bool, str]:
    """
    验证每个时间维度的主导方向概率是否大于等于指定阈值

    Args:
        analysis_json_str: LLM返回的JSON字符串
        min_dominant_prob: 主导方向的最小概率阈值，默认0.5

    Returns:
        Tuple[bool, str]: (验证是否通过, 错误信息)
    """
    try:
        # 清理markdown代码块标记
        cleaned_json = clean_json_response(analysis_json_str)

        # 解析JSON
        data = json.loads(cleaned_json)

        # 检查是否是特殊情况（无数据或全中性）
        if data.get('no_data_or_neutral', False):
            logger.debug("检测到no_data_or_neutral标记，跳过主导概率验证")
            return True, ""

        # 检查是否有direction_judgment字段
        if 'direction_judgment' not in data:
            return False, "缺少direction_judgment字段"

        direction_judgment = data['direction_judgment']

        # 定义需要检查的期限
        periods_to_check = []

        # 检查short_term中的T+1和T+5
        if 'short_term' in direction_judgment:
            short_term = direction_judgment['short_term']
            if 'T+1' in short_term:
                periods_to_check.append(('short_term.T+1', short_term['T+1']))
            if 'T+5' in short_term:
                periods_to_check.append(('short_term.T+5', short_term['T+5']))

        # 检查mid_long_term
        if 'mid_long_term' in direction_judgment:
            periods_to_check.append(('mid_long_term', direction_judgment['mid_long_term']))

        # 验证每个期限的主导概率
        for period_name, period_data in periods_to_check:
            if 'direction_probs' not in period_data:
                return False, f"{period_name}缺少direction_probs字段"

            probs = period_data['direction_probs']

            # 检查必需的键
            if 'up' not in probs or 'down' not in probs or 'side' not in probs:
                return False, f"{period_name}的direction_probs缺少必需的键(up/down/side)"

            # 找到主导方向（概率最大的方向）
            max_prob = max(probs['up'], probs['down'], probs['side'])

            # 确定主导方向的名称
            if max_prob == probs['up']:
                dominant_direction = 'up'
            elif max_prob == probs['down']:
                dominant_direction = 'down'
            else:
                dominant_direction = 'side'

            # 验证主导概率是否大于等于阈值
            if max_prob < min_dominant_prob:
                error_msg = f"{period_name}的主导方向({dominant_direction})概率{max_prob:.4f}小于阈值{min_dominant_prob}"
                logger.warning(error_msg)
                return False, error_msg

        logger.debug(f"主导概率验证通过（阈值{min_dominant_prob}），共检查了{len(periods_to_check)}个期限")
        return True, ""

    except json.JSONDecodeError as e:
        return False, f"JSON解析失败: {str(e)}"
    except Exception as e:
        return False, f"主导概率验证过程发生错误: {str(e)}"


def call_deepseek_for_trading_advice(
    fund_code: str,
    date: str,
    announcements: List[Dict[str, Any]],
    historical_analysis: Dict[str, Any],
    recent_prices: Dict[str, Any],
    events: List[Dict[str, Any]] = None,
    volatility_threshold: float = None,
    recent_5d_changes: List[Dict] = None,
    recent_20d_data: List[Dict] = None,
    min_dominant_prob: float = 0.5
) -> Tuple[str, str]:
    """
    调用 DeepSeek Reasoner 模型进行综合分析（支持概率和验证重试）

    Args:
        fund_code: 基金代码
        date: 当前日期
        announcements: 公告列表
        historical_analysis: 历史影响分析结果
        recent_prices: 最近价格数据
        events: 近期事件列表（包含事件信息和历史分析）
        volatility_threshold: 动态波动阈值
        recent_5d_changes: 最近5日涨跌幅列表
        recent_20d_data: 最近20日价格和涨跌幅列表
        min_dominant_prob: 主导方向的最小概率阈值，默认0.5

    Returns:
        Tuple[str, str]: (分析结果, 推理过程)
    """
    logger.info("开始调用 DeepSeek Reasoner 模型进行综合分析（最多尝试3次）")

    # 获取模型配置
    model_cfg = MODEL_CONFIG['deepseek']['deepseek-reasoner']

    # 初始化客户端
    client = OpenAI(
        api_key=model_cfg['api_key'],
        base_url=model_cfg['base_url']
    )

    # 构建提示词（不变，放在循环外）
    prompt = f"""你是一位专业的REITs投资分析师。分析基金 {fund_code} 在 {date} 时点，基于近几日的公告和事件，为决策者提供**短期交易建议**。

## 背景
这是多agent投资决策系统，你负责公告维度分析，其他agent负责技术面、市场面等。决策agent会综合所有信息做出最终交易策略。

## 字段说明
- **sentiment**: 1=利好, 0=中性, -1=利空（情绪初步判断，并非最终判断，仅供参考）
- **历史影响分析**: 同类公告的历史价格表现（有一定参考价值，但不代表未来）

---

## 数据

### 近7日公告列表（共{len(announcements)}条）

"""

    # 添加公告详情
    for i, ann in enumerate(announcements, 1):
        prompt += f"""
【公告{i}】
- 发布日期: {ann['date']}
- 标题: {ann['announcement_title']}
- 类型: {ann['doc_type_2']}
- 摘要: {ann['digest']}
- 情绪: {ann['sentiment']} ({['利空', '中性', '利好'][ann['sentiment'] + 1]})
- 理由: {ann['sentiment_reason']}
"""

    # 添加历史影响分析
    if historical_analysis:
        prompt += "\n### 相关类型的历史影响分析\n"
        for doc_type, analysis in historical_analysis.items():
            if analysis['status'] == 'success' and analysis.get('analysis'):
                prompt += f"\n【{doc_type}类型】\n{analysis['analysis']}\n"
            else:
                prompt += f"\n【{doc_type}类型】\n暂无足够历史数据\n"

    # 添加近期事件（如果有）
    if events and len(events) > 0:
        prompt += f"\n### 近期事件（当前日期前后{EVENT_DAYS_RANGE}天，共{len(events)}个）\n\n"
        prompt += "**说明**：以下事件是根据event_date（事件生效日期）筛选出来的，包括已发生和即将发生的事件。\n\n"

        for i, evt in enumerate(events, 1):
            # 判断事件类型的中文描述
            event_type_cn = "分红" if evt['event_type'] == "分红公告" else "基金份额解除限售"

            # 判断事件是已发生还是即将发生
            current_date_obj = datetime.strptime(date, '%Y-%m-%d')
            event_date_obj = datetime.strptime(evt['event_date'], '%Y-%m-%d')

            if event_date_obj < current_date_obj:
                time_status = f"已于 {evt['event_date']} 发生"
            elif event_date_obj > current_date_obj:
                days_diff = (event_date_obj - current_date_obj).days
                time_status = f"将于 {evt['event_date']} 发生（{days_diff}天后）"
            else:
                time_status = f"今日（{evt['event_date']}）生效"

            prompt += f"""
【事件{i}】{event_type_cn}
- 事件生效日期: {evt['event_date']} ({time_status})
- 公告发布日期: {evt['publish_date']}
- 事件详情: {evt['digest']}
- 情绪判断（初步）: {evt['sentiment']} ({['利空', '中性', '利好'][evt['sentiment'] + 1]})
- 判断理由（初步）: {evt['sentiment_reason']}
"""

            # 添加历史影响分析（如果成功）
            hist_analysis = evt.get('historical_analysis', {})
            if hist_analysis.get('status') == 'success' and hist_analysis.get('analysis'):
                prompt += f"- 同类事件历史影响:\n{hist_analysis['analysis']}\n"
            else:
                prompt += f"- 同类事件历史影响: 暂无足够历史数据\n"

    # 添加价格数据
    prompt += f"\n### 最近{len(recent_prices.get('交易日数据', []))}个交易日价格数据\n\n"
    prompt += json.dumps(recent_prices, ensure_ascii=False, indent=2)

    # 添加整盘判断专项说明板块
    if volatility_threshold is not None and recent_5d_changes:
        prompt += f"""

### 整盘判断专项说明

#### 一、动态阈值计算方法
本系统使用每日动态阈值方案判断价格变动显著性（即是否处于整盘状态）：
- 基于历史波动率、自适应乘数和边界约束计算
- 每个交易日独立计算

#### 二、当前动态阈值
当前阈值：{volatility_threshold*100:.2f}%
- 预计日收益率绝对值 > {volatility_threshold*100:.2f}% → 显著变动（上涨或下跌）
- 预计日收益率绝对值 ≤ {volatility_threshold*100:.2f}% → 横盘震荡（整盘）

#### 三、公募REITs波动率特性
- 日涨跌幅通常在1%以内
- 超过2%的变动即为显著变动
- 日最高涨跌幅不超过10%
- 整体波动率远低于股票市场

#### 四、基于动态阈值的历史整盘分布统计
根据动态阈值参数设定，历史回测显示：
- 约33%的交易日处于整盘状态
- 整盘是正常的市场状态，代表价格走势不显著

#### 五、最近5个交易日涨跌幅明细
（用于判断最近是否处于整盘状态）
{format_recent_5d_changes(recent_5d_changes, volatility_threshold)}

#### 六、本次预测的整盘判断条件

**不同时间维度的整盘判断标准：**

1. **T+1（次日预测）**：
   - 判断标准：预计次日收益率绝对值 ≤ θₜ（当前动态阈值 {volatility_threshold*100:.2f}%）
   - 即：|次日收益率| ≤ {volatility_threshold*100:.2f}% → 判断为整盘

2. **T+5（未来5个交易日）**：
   - 阈值扩展系数：ε₅ = √5 · θₜ = {(volatility_threshold * (5**0.5))*100:.2f}%
   - 判断标准：预计未来5个交易日的累计波动幅度 ≤ ε₅
   - 或等价于：未来5日的平均日收益率绝对值 ≤ θₜ
   - 解释：基于价格随机游走假设，多日波动率按时间的平方根缩放

**重要说明：**
- 整盘不等于不确定，是正常的市场状态，代表价格走势不显著
- 当判断为整盘时，应给予合理的主导概率和合理的置信度
- 整盘的判断应基于公告影响的综合分析
"""

    prompt += """

---

## 分析任务

你必须**严格按照以下JSON格式**输出分析结果。输出内容必须是**纯JSON格式**。

### 输出格式示例：

{{
  "expert_type": "announcement",
  "analysis_date": "{date}",
  "fund_code": "{fund_code}",

  "direction_judgment": {{
    "short_term": {
      "T+1": {
        "direction_probs": {
          "up": 0.00,
          "down": 0.00,
          "side": 0.00
        },
        "confidence": 0.00,
        "rationale": "核心理由(1句话，50字以内)"
      },
      "T+5": {
        "direction_probs": {
          "up": 0.00,
          "down": 0.00,
          "side": 0.00
        },
        "confidence": 0.00,
        "rationale": "核心理由(1句话，50字以内)"
      }
    }
  },

  "detailed_analysis": {
    "announcements_overview": "近X日共X条公告，利好X条/中性X条/利空X条，主要类型为...",
    "key_announcements": [
      {
        "title": "公告标题或类型",
        "impact_analysis": "核心内容+影响判断（100字以内）",
        "time_decay": "时间衰减评估（50字以内）"
      }
    ],
    "comprehensive_analysis": "价格观察、综合影响分析、投资建议等（300字以内）",
    "risk_alerts": ["风险点1", "风险点2"]
  }
}

---

## 字段填写说明——严格遵守

### 1. direction_judgment（方向判断）

#### direction_probs（方向概率）
- **必须满足**：`up + down + side = 1.0`
- **主导方向概率范围**：[0.50, 0.90]，主导方向的概率必须大于另外两个方向，不要出现两个相同的最大概率
- **避免极端概率**：任何单一方向概率不应低于0.05
- **含义**：
  - `up`: 上涨概率
  - `down`: 下跌概率
  - `side`: 横盘概率
-  **整盘的判断**：由于公募REITs波动率较低，本系统使用每日价格波动是否超过动态阈值来判断是否处于整盘状态，请充分考虑这一点，在确定主导方向时除了分析关键信息以外还需结合动态阈值、最近整盘突破值等来综合判断


**强信号**（多项指标共振、方向一致）：
- 主导方向概率：≥ 0.70
- 主导概率与次方向差距：≥ 0.45

**中等信号**（主要指标明确但有少量矛盾）：
- 主导方向概率：0.60 - 0.65
- 主导概率与次方向差距：≥ 0.25

**弱信号**（信号模糊或矛盾较多）：
- 主导方向概率：0.50 - 0.55
- 主导概率与次方向差距：≥ 0.10

#### confidence（置信度）
反映你对该判断的整体信心（0.00-1.00）：

**强信号**（多项指标共振）：
- 置信度：≥ 0.70
- 情况：多公告方向明确，且历史分析明确，价格已有反应

**中等信号**（主要指标明确但有矛盾）：
- 置信度：0.60 - 0.65
- 情况：公告方向明确，但数量少历史分析不明确，或存在次要矛盾

**弱信号**（信号模糊）：
- 置信度：0.50 - 0.55
- 情况：公告中性，或历史数据不足

**极弱信号**（矛盾严重）：
- 置信度：≤ 0.45
- 情况：公告相互矛盾，或价格走势与公告sentiment严重背离

**重要**：
- 在公募REITs低波动背景下，公告信号的微小变化也具有指导意义

#### rationale（理由）
用1句话（50字以内）说明核心理由

---

### 2. detailed_analysis（详细分析）

#### announcements_overview（公告概况）
简述近X日公告总数、利好/中性/利空数量、主要类型（不超过50字）

#### key_announcements（关键公告分析）
- 选择最重要的公告进行分析（最多3篇）
- 如全是中性且无实质影响，可为空数组 `[]`
- 每篇包含：
  - **title**: 公告标题或类型
  - **impact_analysis**: 核心内容+影响判断，是否认同sentiment？结合历史数据判断影响方向和强度（100字以内）
  - **time_decay**: 发布几天了？影响已体现多少？未来还有多少影响空间？（50字以内）

#### comprehensive_analysis（综合分析）
基于上述公告的综合分析（300字以内），包括：
- 价格观察：最近5日价格走势特征，与公告sentiment的一致性
- 综合影响分析：多个公告的叠加效应
- 投资建议：短期交易建议
- 使用具体数据作为支撑（如需）

#### risk_alerts（风险提示）
数组格式，列出1-3个当前最需要关注的风险点，每个风险点用一句话表述

---

## 特殊情况处理

**如果所有公告均为中性且无实质影响**：
- **不要提供direction_judgment字段**（即不提供方向判断和置信度）
- 使用以下特殊JSON格式：


{{
  "expert_type": "announcement",
  "analysis_date": "{date}",
  "fund_code": "{fund_code}",
  "no_data_or_neutral": true,
  "reason": "所有公告均为中性且无实质影响",
  "message": "公告维度信息有限，无法提供有效的方向判断，建议重点参考其他维度分析"
}}


**说明**：
- 当所有公告都是中性且无实质影响时，强行给出方向判断会误导决策
- 此时应明确告知"无法判断"，引导决策者参考其他维度
- `no_data_or_neutral` 字段标识这是特殊情况
- `reason` 字段说明原因
- `message` 字段给出建议

---

## 重要提醒

1. **必须输出有效的JSON格式**，确保所有括号、引号、逗号正确
2. **概率之和必须=1.0**，系统会自动检查
3. **保持一致性**：JSON中的概率判断应与detailed_analysis一致
4. **使用具体数据**：在阐述观点时，优先使用具体数据作为支撑
5. **控制字数**：严格遵守各字段的字数限制
6. **避免过度保守，敢于判断**：根据各维度信号强度给出相应的概率和置信度，不要因为害怕错误而系统性降低数值。请记住：
   **市场特性要求更果断的判断**：
   - 公募REITs波动率远低于股票
   - 整盘（side）是公募REITs常见且正常的状态，判断为整盘时也应给予合理的主导概率和置信度

   **投资者的策略依赖你的预测质量**：
   - 策略目标：在回撤可控的前提下最大化收益
   - 你的角色：提供清晰、有置信度的方向预测，作为交易决策的关键输入
   - "模糊安全"倾向（总是选择中间值或较低值），会显著降低策略的收益潜力
   - 因此在有明确信号（无论上涨、下跌还是整盘）时请给予较高的概率和置信度

   **注意**：这并不意味着盲目乐观，而是要求你基于充分的证据做出匹配的判断强度
7. **检查概率与置信度**：返回前检查 `direction_probs` 和 `confidence` 是否符合字段填写说明的要求，如果发现不符合要求，请根据分析重新打分

请开始分析，输出JSON格式结果。
"""

    if LOG_MODE == 'detailed':
        logger.debug(f"发送给LLM的提示词:\n{prompt}")

    # 最多尝试3次
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"第 {attempt} 次调用 LLM...")

            # 调用模型
            response = client.chat.completions.create(
                model=model_cfg['model'],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )

            # 提取结果
            analysis = response.choices[0].message.content

            # DeepSeek Reasoner 的推理过程在 reasoning_content 字段
            reasoning = ""
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning = response.choices[0].message.reasoning_content
                logger.info(f"成功获取推理过程，长度: {len(reasoning) if reasoning else 0} 字符")

            logger.info("LLM 调用成功，开始验证概率和...")

            if LOG_MODE == 'detailed':
                logger.debug(f"LLM 返回的分析结果:\n{analysis}")
                if reasoning:
                    logger.debug(f"LLM 推理过程:\n{reasoning}")

            # 验证概率和
            is_valid_sum, error_msg_sum = validate_probability_sum(analysis)

            if not is_valid_sum:
                logger.warning(f"✗ 概率和验证失败（第 {attempt} 次尝试）: {error_msg_sum}")
                last_error = f"概率和验证失败: {error_msg_sum}"

                if attempt < max_attempts:
                    logger.info(f"准备进行第 {attempt + 1} 次尝试...")
                    continue
                else:
                    logger.error(f"已达最大重试次数（{max_attempts}次），概率和验证仍未通过")
                    raise ValueError(f"概率和验证失败（尝试{max_attempts}次后仍未通过）: {error_msg_sum}")

            # 验证主导概率
            is_valid_dominant, error_msg_dominant = validate_dominant_probability(analysis, min_dominant_prob)

            if not is_valid_dominant:
                logger.warning(f"✗ 主导概率验证失败（第 {attempt} 次尝试）: {error_msg_dominant}")
                last_error = f"主导概率验证失败: {error_msg_dominant}"

                if attempt < max_attempts:
                    logger.info(f"准备进行第 {attempt + 1} 次尝试...")
                    continue
                else:
                    logger.error(f"已达最大重试次数（{max_attempts}次），主导概率验证仍未通过")
                    raise ValueError(f"主导概率验证失败（尝试{max_attempts}次后仍未通过）: {error_msg_dominant}")

            # 所有验证通过
            logger.info(f"✓ 概率和验证通过（第 {attempt} 次尝试）")
            logger.info(f"✓ 主导概率验证通过（阈值={min_dominant_prob}，第 {attempt} 次尝试）")
            # 清理analysis，去除markdown代码块标记
            cleaned_analysis = clean_json_response(analysis)
            return cleaned_analysis, reasoning

        except ValueError:
            # 重新抛出概率验证失败的异常
            raise
        except Exception as e:
            logger.error(f"第 {attempt} 次调用 LLM 失败: {e}")
            last_error = str(e)

            if attempt < max_attempts:
                logger.info(f"准备进行第 {attempt + 1} 次尝试...")
            else:
                logger.error(f"已达最大重试次数（{max_attempts}次）")
                raise Exception(f"调用 LLM 失败（尝试{max_attempts}次）: {last_error}")

    # 理论上不会到达这里
    raise Exception(f"调用 LLM 失败: {last_error}")


def analyze_recent_announcements(
    fund_code: str,
    date: str,
    include_reasoning: bool = False,
    min_dominant_prob: float = 0.5
) -> Dict[str, Any]:
    """
    分析近期公告对价格的影响，为交易决策提供建议

    Args:
        fund_code: 基金代码
        date: 当前日期 (格式: YYYY-MM-DD)
        include_reasoning: 是否包含推理过程
        min_dominant_prob: 主导方向的最小概率阈值，默认0.5

    Returns:
        Dict: {
            'status': 'success' 或 'error',
            'message': 错误信息（如果有）,
            'analysis': LLM分析结果,
            'reasoning': LLM推理过程（仅当include_reasoning=True时）
        }
    """
    logger.info("=" * 80)
    logger.info("开始执行近期公告影响分析")
    logger.info(f"输入参数 - fund_code: {fund_code}, date: {date}, include_reasoning: {include_reasoning}")
    logger.info(f"日志模式: {LOG_MODE}")

    try:
        # 1. 验证日期格式
        if not validate_date_format(date):
            error_msg = f"日期格式错误: {date}，应为 YYYY-MM-DD 格式"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }

        # 2-5. 第一批并行任务：查询公告、事件、价格（互不依赖，可并行）
        logger.info("开始第一批并行任务：查询公告、事件、价格")

        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交三个并行任务
            future_announcements = executor.submit(query_recent_announcements, fund_code, date)
            future_events = executor.submit(query_recent_events, fund_code, date)
            future_prices = executor.submit(get_recent_price_changes, fund_code, date)

            # 等待所有任务完成并获取结果
            announcements = future_announcements.result()
            recent_events = future_events.result()
            recent_prices = future_prices.result()

        logger.info("第一批并行任务完成")

        # 检查公告是否为空
        if not announcements:
            warning_msg = f"未找到近{DAYS_RANGE}日内的公告记录"
            logger.warning(warning_msg)
            # 返回特殊JSON格式，不包含方向判断
            no_data_json = {
                "expert_type": "announcement",
                "analysis_date": date,
                "fund_code": fund_code,
                "no_data_or_neutral": True,
                "reason": "近期没有发布新公告",
                "message": "公告维度数据不足，无法提供有效的方向判断，建议重点参考其他维度分析"
            }
            import json
            return {
                'status': 'success',
                'message': warning_msg,
                'analysis_result': json.dumps(no_data_json, ensure_ascii=False, indent=2),
                'metadata': {
                    'expert_type': 'announcement',
                    'fund_code': fund_code,
                    'analysis_date': date,
                    'timestamp': datetime.now().isoformat()
                }
            }

        # 3. 获取需要历史影响分析的公告类型
        doc_types_need_analysis = []
        for ann in announcements:
            doc_type = ann['doc_type_2']
            if doc_type in DOC_TYPES_NEED_HISTORICAL and doc_type not in doc_types_need_analysis:
                doc_types_need_analysis.append(doc_type)

        logger.info(f"需要获取历史分析的公告类型: {doc_types_need_analysis}")

        # 4-7. 第二批并行任务：获取公告历史影响、获取事件历史影响（可并行）
        logger.info("开始第二批并行任务：获取历史影响分析")

        historical_analysis = {}
        enriched_events = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            # 提交公告历史影响分析任务
            if doc_types_need_analysis:
                futures['announcement'] = executor.submit(
                    get_historical_impact_for_doc_types,
                    fund_code, date, doc_types_need_analysis
                )

            # 提交事件历史影响分析任务
            if recent_events:
                futures['event'] = executor.submit(
                    get_historical_impact_for_events,
                    fund_code, date, recent_events
                )

            # 等待所有任务完成并获取结果
            for future_type, future in futures.items():
                if future_type == 'announcement':
                    historical_analysis = future.result()
                elif future_type == 'event':
                    enriched_events = future.result()
                    logger.info(f"近期事件数量: {len(enriched_events)}")

        if not recent_events:
            logger.info("未找到近期事件")

        logger.info("第二批并行任务完成")

        # 7.5. 获取价格数据并计算动态阈值
        logger.info("步骤7.5: 获取价格数据并计算动态阈值")
        fund_price_df = get_fund_price_data(fund_code, date, 150)

        # 计算动态阈值
        volatility_threshold = None
        if len(fund_price_df) >= 2:
            price_list = fund_price_df['close'].tolist()
            volatility_threshold = calculate_daily_volatility_threshold(price_list)
            logger.info(f"✓ 计算动态阈值: {volatility_threshold*100:.2f}%" if volatility_threshold else "✗ 动态阈值计算失败")
        else:
            logger.warning("✗ 基金价格数据不足，无法计算动态阈值")

        # 计算最近5日涨跌幅
        recent_5d_changes = []
        if len(fund_price_df) >= 6:
            for i in range(-5, 0):
                daily_change = ((fund_price_df['close'].iloc[i] - fund_price_df['close'].iloc[i-1]) /
                                fund_price_df['close'].iloc[i-1] * 100)
                recent_5d_changes.append({
                    'date': str(fund_price_df['trade_date'].iloc[i]),
                    'change_pct': daily_change
                })
            logger.info(f"✓ 计算最近5日涨跌幅")
        else:
            logger.warning("✗ 基金价格数据不足，无法计算最近5日涨跌幅")

        # 计算最近20日价格和涨幅
        recent_20d_data = []
        if len(fund_price_df) >= 21:
            for i in range(-20, 0):
                daily_change = ((fund_price_df['close'].iloc[i] - fund_price_df['close'].iloc[i-1]) /
                                fund_price_df['close'].iloc[i-1] * 100) if i > -21 else 0.0
                recent_20d_data.append({
                    'date': str(fund_price_df['trade_date'].iloc[i]),
                    'price': float(fund_price_df['close'].iloc[i]),
                    'change_pct': daily_change
                })
            logger.info(f"✓ 计算最近20日价格数据")
        else:
            logger.warning("✗ 基金价格数据不足，无法计算最近20日数据")

        # 8. 调用 LLM 进行综合分析
        analysis, reasoning = call_deepseek_for_trading_advice(
            fund_code, date, announcements, historical_analysis, recent_prices, enriched_events,
            volatility_threshold, recent_5d_changes, recent_20d_data, min_dominant_prob
        )

        # 9. 构建返回结果
        result = {
            'status': 'success',
            'analysis_result': analysis,
            'metadata': {
                'expert_type': 'announcement',
                'fund_code': fund_code,
                'analysis_date': date,
                'timestamp': datetime.now().isoformat()
            }
        }

        if include_reasoning:
            result['reasoning_process'] = reasoning

        logger.info("近期公告影响分析完成")
        logger.info("=" * 80)

        return result

    except Exception as e:
        error_msg = f"执行过程中发生错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'message': error_msg,
            'metadata': {
                'expert_type': 'announcement',
                'fund_code': fund_code,
                'analysis_date': date,
                'timestamp': datetime.now().isoformat()
            }
        }
