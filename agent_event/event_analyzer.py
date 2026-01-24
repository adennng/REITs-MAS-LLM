#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REITs事件分析Agent
从新闻、季报、运营数据等事件维度分析投资机会，为决策提供支持
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
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


# ============================================================================
# 配置参数 - 可手动修改
# ============================================================================
# 新闻查询天数范围
NEWS_DAYS_RANGE = 7

# 季报提醒开始日期（每个季报月的第N日开始检查）
QUARTERLY_REMINDER_START_DAY = 10

# 日志模式
LOG_MODE = 'detailed'  # 'detailed' 或 'simple'
# detailed: 记录全过程(入参、数据库查询结果、LLM调用、推理过程等)
# simple: 只记录入参、输出、执行状态
# ============================================================================


def setup_logger(log_mode: str = LOG_MODE) -> logging.Logger:
    """
    配置日志记录器

    Args:
        log_mode: 日志模式 ('detailed' 或 'simple')

    Returns:
        Logger: 配置好的日志记录器
    """
    logger = logging.getLogger('EventAnalyzer')
    logger.setLevel(logging.DEBUG if log_mode == 'detailed' else logging.INFO)

    # 清除已有的处理器（避免重复）
    logger.handlers.clear()

    # 确定日志目录（相对路径：../log/）
    log_dir = os.path.join(parent_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = 'event_analyzer.log'
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


# 初始化日志
logger = setup_logger()


def get_db_connection():
    """获取数据库连接"""
    config = get_db_reits_config()
    return pymysql.connect(**config)


def format_recent_5d_changes(recent_5d_changes: List[Dict], volatility_threshold: Optional[float]) -> str:
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


def get_recent_news(current_date: str, days_range: int = NEWS_DAYS_RANGE) -> List[Dict[str, Any]]:
    """
    获取最近的新闻

    Args:
        current_date: 当前日期 (格式: YYYY-MM-DD)
        days_range: 查询天数范围

    Returns:
        新闻列表，每条新闻包含 title, digest, score, date
    """
    logger.debug(f"开始查询最近{days_range}天的新闻，截止日期：{current_date}")

    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 计算起始日期
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
            start_dt = current_dt - timedelta(days=days_range - 1)
            start_date = start_dt.strftime('%Y-%m-%d')

            sql = """
                SELECT title, digest, score, date
                FROM news
                WHERE date >= %s AND date <= %s
                ORDER BY date DESC, score DESC
            """
            cursor.execute(sql, (start_date, current_date))
            results = cursor.fetchall()

            logger.debug(f"查询到 {len(results)} 条新闻")
            if logger.level == logging.DEBUG:
                for idx, news in enumerate(results, 1):
                    logger.debug(f"  新闻{idx}: [{news['date']}] score={news['score']} - {news['title']}")

            return results
    finally:
        conn.close()


def get_latest_quarterly_and_operations(fund_code: str, current_date: str) -> Tuple[Optional[Dict], List[Dict]]:
    """
    获取最近一期的季报和近期的运营数据报告

    Args:
        fund_code: 基金代码
        current_date: 当前日期 (格式: YYYY-MM-DD)

    Returns:
        (最近季报, 运营数据报告列表)
    """
    logger.debug(f"开始查询基金 {fund_code} 的最近季报和运营数据")

    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 1. 查找最近的季报
            sql_quarterly = """
                SELECT announcement_title, digest, sentiment, sentiment_reason, date
                FROM processed_files
                WHERE fund_code = %s
                  AND doc_type_2 = '季报'
                  AND date <= %s
                ORDER BY date DESC
                LIMIT 1
            """
            cursor.execute(sql_quarterly, (fund_code, current_date))
            quarterly_report = cursor.fetchone()

            if not quarterly_report:
                logger.warning(f"未找到基金 {fund_code} 在 {current_date} 之前的季报")
                return None, []

            logger.debug(f"找到最近季报: [{quarterly_report['date']}] {quarterly_report['announcement_title']}")
            logger.debug(f"  sentiment={quarterly_report['sentiment']}, reason={quarterly_report['sentiment_reason']}")

            # 2. 查找该季报日期与当前日期之间的运营数据报告
            quarterly_date = quarterly_report['date']
            sql_operations = """
                SELECT announcement_title, digest, sentiment, sentiment_reason, date
                FROM processed_files
                WHERE fund_code = %s
                  AND doc_type_2 = '主要运营数据'
                  AND date > %s
                  AND date <= %s
                ORDER BY date ASC
            """
            cursor.execute(sql_operations, (fund_code, quarterly_date, current_date))
            operation_reports = cursor.fetchall()

            logger.debug(f"找到 {len(operation_reports)} 条运营数据报告")
            if logger.level == logging.DEBUG:
                for idx, op in enumerate(operation_reports, 1):
                    logger.debug(f"  运营数据{idx}: [{op['date']}] {op['announcement_title']}")

            return quarterly_report, operation_reports
    finally:
        conn.close()


def check_quarterly_report_reminder(fund_code: str, current_date: str) -> Optional[Dict[str, Any]]:
    """
    检查是否需要季报提醒

    Args:
        fund_code: 基金代码
        current_date: 当前日期 (格式: YYYY-MM-DD)

    Returns:
        如果需要提醒，返回提醒信息字典；否则返回 None
    """
    logger.debug(f"检查是否需要季报提醒，基金: {fund_code}, 日期: {current_date}")

    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
    current_month = current_dt.month
    current_day = current_dt.day
    current_year = current_dt.year

    # 定义季报月份和对应的季度
    quarterly_months = {
        4: '1季度',
        7: '2季度',
        10: '3季度',
        1: '4季度'
    }

    # 判断是否是季报月
    if current_month not in quarterly_months:
        logger.debug(f"当前月份 {current_month} 不是季报发布月，无需提醒")
        return None

    # 判断是否达到提醒起始日期
    if current_day < QUARTERLY_REMINDER_START_DAY:
        logger.debug(f"当前日期 {current_day} 小于提醒起始日期 {QUARTERLY_REMINDER_START_DAY}，无需提醒")
        return None

    quarter = quarterly_months[current_month]
    logger.debug(f"当前是 {quarter} 报告发布月，且已过 {QUARTERLY_REMINDER_START_DAY} 日，检查是否已发布")

    # 检查当月是否已发布季报（发布日期必须在当前日期之前或等于当前日期）
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT COUNT(*) as count
                FROM processed_files
                WHERE fund_code = %s
                  AND doc_type_2 = '季报'
                  AND YEAR(date) = %s
                  AND MONTH(date) = %s
                  AND date <= %s
            """
            cursor.execute(sql, (fund_code, current_year, current_month, current_date))
            result = cursor.fetchone()

            if result['count'] > 0:
                logger.debug(f"{quarter}报告已在本月发布（发布日期 <= {current_date}），无需提醒")
                return None

            logger.debug(f"{quarter}报告尚未发布（当前日期：{current_date}），需要提醒")
            return {
                'quarter': quarter,
                'month': current_month,
                'year': current_year
            }
    finally:
        conn.close()


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

    conn = get_db_connection()
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


def build_analysis_prompt(
    news_list: List[Dict[str, Any]],
    quarterly_report: Optional[Dict],
    operation_reports: List[Dict],
    quarterly_reminder: Optional[Dict[str, Any]],
    fund_code: str,
    current_date: str,
    volatility_threshold: Optional[float],
    recent_5d_changes: List[Dict],
    recent_20d_data: List[Dict]
) -> str:
    """
    构建分析提示词

    Args:
        news_list: 新闻列表
        quarterly_report: 最近季报
        operation_reports: 运营数据报告列表
        quarterly_reminder: 季报提醒信息
        fund_code: 基金代码
        current_date: 当前日期
        volatility_threshold: 动态波动阈值
        recent_5d_changes: 最近5日涨跌幅列表
        recent_20d_data: 最近20日价格和涨跌幅列表

    Returns:
        完整的提示词
    """
    # ========== 开头说明 ==========
    intro = f"""你是一位专业的REITs投资分析师，现在需要从事件角度分析基金 {fund_code} 在 {current_date} 的投资价值。背景：这是多agent投资决策系统，你负责从以下维度分析，其他agent负责技术面、市场面等。决策agent会综合所有信息做出最终交易策略。

请基于以下信息进行分析：
"""

    # ========== 1. 新闻部分 ==========
    if news_list:
        news_section = f"""{'='*60}
一、最近的市场新闻
{'='*60}

以下是最近{NEWS_DAYS_RANGE}天内的REITs市场新闻：

【评分说明】
- score=1: 比较积极但可能对价格无影响，一般可以忽略
- score=2: 非常积极且很可能对价格有影响，短期内（发布后1-2个交易日）可能有利于价格上涨
"""
        # 添加每条新闻
        news_items = []
        for idx, news in enumerate(news_list, 1):
            days_ago = (datetime.strptime(current_date, '%Y-%m-%d') -
                       datetime.strptime(str(news['date']), '%Y-%m-%d')).days
            time_desc = "今日" if days_ago == 0 else f"{days_ago}天前"

            news_item = f"""
新闻{idx} ({time_desc}, score={news['score']}):
标题: {news['title']}
摘要: {news['digest']}"""
            news_items.append(news_item)

        news_section += "\n".join(news_items)

        news_section += """

【分析要点】
- 重点关注score=2的新闻，特别是发布时间越近影响越大
- 新闻影响通常集中在发布后1-2个交易日，时间越久影响越小
- 需评估新闻对该基金的具体影响程度（市场整体利好不一定等于该基金利好）"""
    else:
        news_section = "\n一、最近无重要市场新闻\n"

    # ========== 2. 季报和运营数据部分 ==========
    fundamental_section = ""
    if quarterly_report or operation_reports:
        fundamental_section = f"""
{'='*60}
二、基本面情况（季报与运营数据）
{'='*60}
"""
        if quarterly_report:
            fundamental_section += f"""
【最近一期季报】({quarterly_report['date']})
标题: {quarterly_report['announcement_title']}
摘要: {quarterly_report['digest']}
情绪判断: {quarterly_report['sentiment']} (0=中性, 1=利好, -1=利空)
判断理由: {quarterly_report['sentiment_reason']}

注: 情绪判断为初步判断，仅供参考"""

        if operation_reports:
            fundamental_section += f"""

【近期运营数据报告】(季报发布后共{len(operation_reports)}期)"""

            for idx, op in enumerate(operation_reports, 1):
                fundamental_section += f"""

运营数据{idx} ({op['date']}):
标题: {op['announcement_title']}
摘要: {op['digest']}
情绪判断: {op['sentiment']} (0=中性, 1=利好, -1=利空)
判断理由: {op['sentiment_reason']}"""

        fundamental_section += """

【分析要点】
- 以上是该基金最近的项目经营情况，反映了当前的基本面和经营基调
- 情绪判断仅为初步分析，请结合具体数据综合评估
- 建议以**判断理由**为准来分析，需关注经营是否在改善或恶化，关注经营水平是否达到行业平均水平之上。就算财务指标虽然有上涨，但是运营情况依旧不佳或经营指标依旧不好看，或者有明显负面事件，或者有影响未来现金流的重大情况，则也视为消极影响。这表明本期的上涨并不能从根本上扭转经营状况不佳的局面，经营情况依旧不乐观，只有财务或相关数据达到正常水平及以上的程度才算积极影响。"""

    # ========== 3. 季报提醒部分 ==========
    reminder_section = ""
    if quarterly_reminder:
        quarter = quarterly_reminder['quarter']
        month = quarterly_reminder['month']

        reminder_section = f"""
{'='*60}
三、季报发布预警
{'='*60}

【重要提醒】{quarter}报告预计将于本月（{month}月）20日左右发布

【投资影响分析】
1. 大型机构投资者通常会在季报发布前1-2周根据预期调整持仓
2. 结合最近运营数据的趋势和整体运营水平，初步预判本季度经营情况，如果近期运营数据显示经营情况明显改善或恶化，可能引发提前布局或撤离
3. 建议投资者密切关注：
   - 近期价格走势和成交量变化（可能反映机构动向）
   - 同板块REITs指数的表现，判断是否存在板块性机会或风险"""

    # ========== 4. 价格形态 ==========
    price_pattern_section = ""
    if recent_20d_data:
        price_pattern_section = f"""
{'='*60}
四、价格形态
{'='*60}

### 最近20个交易日价格和涨跌幅
以下是基金 {fund_code} 最近20个交易日的价格走势：

"""
        # 添加最近20日数据表格
        for idx, item in enumerate(recent_20d_data, 1):
            price_pattern_section += f"- 第{idx}天 ({item['date']}): 收盘价 {item['price']:.3f}元, 涨跌幅 {item['change_pct']:+.2f}%\n"

        price_pattern_section += """
### 价格形态说明
- 以上数据展示了目标基金最近20个交易日的价格走势
- 可以帮助你判断：
  - 价格趋势（上涨/下跌/横盘）
  - 波动幅度
  - 连续涨跌情况
  - 是否处于整盘状态
"""

    # ========== 5. 整盘判断专项说明 ==========
    sideways_section = ""
    if volatility_threshold is not None:
        sideways_section = f"""
{'='*60}
五、整盘判断专项说明
{'='*60}

### 一、动态阈值计算方法
本系统使用每日动态阈值方案判断价格变动显著性（即是否处于整盘状态）：
- 基于历史波动率、自适应乘数和边界约束计算
- 每个交易日独立计算

### 二、当前动态阈值
当前阈值：{volatility_threshold*100:.2f}%
- 预计日收益率绝对值 > {volatility_threshold*100:.2f}% → 显著变动（上涨或下跌）
- 预计日收益率绝对值 ≤ {volatility_threshold*100:.2f}% → 横盘震荡（整盘）

### 三、公募REITs波动率特性
- 日涨跌幅通常在1%以内
- 超过2%的变动即为显著变动
- 日最高涨跌幅不超过10%
- 整体波动率远低于股票市场

### 四、基于动态阈值的历史整盘分布统计
根据动态阈值参数设定，历史回测显示：
- 约33%的交易日处于整盘状态
- 整盘是正常的市场状态，代表价格走势不显著

### 五、最近5个交易日涨跌幅明细
（用于判断最近是否处于整盘状态）
{format_recent_5d_changes(recent_5d_changes, volatility_threshold)}

### 六、本次预测的整盘判断条件

**不同时间维度的整盘判断标准：**

1. **T+1（次日预测）**：
   - 判断标准：预计次日收益率绝对值 ≤ θₜ（当前动态阈值 {volatility_threshold*100:.2f}%）
   - 即：|次日收益率| ≤ {volatility_threshold*100:.2f}% → 判断为整盘

2. **T+5（未来5个交易日）**：
   - 阈值扩展系数：ε₅ = √5 · θₜ = {(volatility_threshold * (5**0.5))*100:.2f}%
   - 判断标准：预计未来5个交易日的累计波动幅度 ≤ ε₅
   - 或等价于：未来5日的平均日收益率绝对值 ≤ θₜ
   - 解释：基于价格随机游走假设，多日波动率按时间的平方根缩放

3. **T+20（未来20个交易日）**：
   - 阈值扩展系数：ε₂₀ = √20 · θₜ = {(volatility_threshold * (20**0.5))*100:.2f}%
   - 判断标准：预计未来20个交易日的累计波动幅度 ≤ ε₂₀
   - 或等价于：未来20日的平均日收益率绝对值 ≤ θₜ
   - 解释：同样基于波动率的时间平方根缩放规律

**重要说明：**
- 整盘不等于不确定，是正常的市场状态，代表价格走势不显著
- 当判断为整盘时，应给予合理的主导概率和合理的置信度
- 整盘的判断应基于事件影响的综合分析
"""

    # ========== 6. 分析策略指引 ==========
    # 根据实际情况动态生成分析策略
    has_news = bool(news_list)
    has_high_score_news = bool([n for n in news_list if n['score'] == 2]) if news_list else False
    has_fundamental = bool(quarterly_report or operation_reports)
    has_reminder = bool(quarterly_reminder)

    # 判断当前属于哪种情况
    if has_fundamental and not has_news and not has_reminder:
        situation = "情况1：仅有基本面数据"
        strategy = """当前仅有季报和运营数据，无重要新闻催化和季报预警。
分析策略：
- 以基本面分析为主，重点评估项目经营状况和趋势
- 基于运营数据判断当前是否处于良好的投资时点
- 置信度相对中性，建议等待更多信号确认"""

    elif has_fundamental and has_high_score_news and not has_reminder:
        situation = "情况2：基本面+积极新闻"
        strategy = """当前有基本面数据且存在高分新闻（score=2），无季报预警。
分析策略：
- 综合考虑基本面基调和新闻催化作用
- 重点评估：新闻对该基金的影响是否与基本面相符
- 如果新闻利好且基本面良好，短期（1-2交易日）可能有交易性机会
- 如果新闻利好但基本面较差，需警惕短期反弹后的风险"""

    elif has_fundamental and not has_news and has_reminder:
        situation = "情况3：基本面+季报预警"
        strategy = """当前有基本面数据且处于季报发布预警期，无重要新闻。
分析策略：
- 重点关注运营数据趋势，预判即将发布的季报可能表现
- 评估机构提前调仓的可能性和方向
- 如果运营数据持续改善或表现良好，可能存在季报前的布局机会
- 如果运营数据恶化或整体水平较差，需警惕机构提前撤离的风险
- 建议密切监控近期价格和成交量变化"""

    elif has_fundamental and has_high_score_news and has_reminder:
        situation = "情况4：基本面+积极新闻+季报预警"
        strategy = """当前同时存在基本面数据、高分新闻和季报预警，信息最为丰富。
分析策略：
- 综合权衡三方面因素，给出最全面的判断
- 短期看新闻催化（1-2交易日）
- 中期看季报预期（未来2周）
- 长期看基本面基调
- 需特别关注三者之间是否一致：
  * 如果三者共振（都积极或都消极），置信度可以较高
  * 如果存在矛盾（如新闻利好但基本面恶化），需谨慎判断并在风险提示中说明"""
    else:
        # 其他情况（如没有基本面数据，或只有低分新闻等）
        situation = "特殊情况：信息不足或无明显信号"
        strategy = """当前信息较为有限或无明显投资信号。
分析策略：
- 基于现有信息给出谨慎判断
- 置信度应设为"低"
- 在风险提示中明确说明信息不足的问题
- 建议等待更多信息或信号出现"""

    strategy_section = f"""
{'='*60}
分析策略指引
{'='*60}

【当前情况】{situation}

【策略建议】
{strategy}"""

    # ========== 7. 分析任务要求 ==========
    task_section = f"""
{'='*60}
分析任务要求
{'='*60}

你必须**严格按照以下JSON格式**输出分析结果。输出内容必须是**纯JSON格式**。

### 输出格式示例：

{{
  "expert_type": "event",
  "analysis_date": "{current_date}",
  "fund_code": "{fund_code}",

  "direction_judgment": {{
    "short_term": {{
      "T+1": {{
        "direction_probs": {{
          "up": 0.00,
          "down": 0.00,
          "side": 0.00
        }},
        "confidence": 0.00,
        "rationale": "核心理由(1句话，50字以内)"
      }},
      "T+5": {{
        "direction_probs": {{
          "up": 0.00,
          "down": 0.00,
          "side": 0.00
        }},
        "confidence": 0.00,
        "rationale": "核心理由(1句话，50字以内)"
      }}
    }},
    "T+20": {{
      "direction_probs": {{
        "up": 0.00,
        "down": 0.00,
        "side": 0.00
      }},
      "confidence": 0.00,
      "rationale": "核心理由(1句话，50字以内)"
    }}
  }},

  "detailed_analysis": {{
    "news_analysis": {{
      "high_score_news": [
        {{
          "title": "新闻标题",
          "score": 2,
          "key_info": "关键信息（50字以内）"
        }}
      ]
    }},
    "fundamental_analysis": {{
      "quarterly_report": {{
        "title": "季报标题简写",
        "sentiment": 0,
        "key_info": "影响情绪判断的关键信息/数据/事件（100字以内）"
      }},
      "operation_reports": [
        {{
          "title": "运营数据标题简写",
          "sentiment": 0,
          "key_info": "影响情绪判断的关键信息/数据/事件（100字以内）"
        }}
      ]
    }},
    "quarterly_reminder": {{
      "has_reminder": false,
      "expected_date": "YYYY-MM-DD",
      "note": "提示信息（如：预计本月20日左右发布新公告，建议关注机构提前布局）"
    }},
    "comprehensive_analysis": "综合上述因素进行分析，给出投资逻辑和建议（200字以内，不使用换行符）",
    "risk_alerts": ["风险点1", "风险点2"]
  }}
}}

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

#### 时间敏感性与概率差异

**立即影响**：
- T+1高置信度，概率倾向明确
- T+5置信度下降（影响已消化）

**短期影响**（：
- T+1和T+5概率接近
- 置信度可能略降

**中长期影响**（如基本面变化）：
- 短期影响有限
- T+20置信度更高

#### confidence（置信度）
反映你对该判断的整体信心（0.00-1.00）：

**强信号**（多项指标共振）：
- 置信度：≥ 0.70
- 情况：新闻score=2且基本面积极，或季报预警且运营数据持续改善，或多项事件共振

**中等信号**（主要指标明确但有矛盾）：
- 置信度：0.60 - 0.65
- 情况：新闻利好但基本面一般，或基本面好但无催化，或存在次要矛盾

**弱信号**（信号模糊）：
- 置信度：0.50 - 0.55
- 情况：新闻score=1，或基本面中性，或信息不足

**极弱信号**（矛盾严重）：
- 置信度：≤ 0.45
- 情况：各项指标相互矛盾，新闻与基本面相悖

**重要**：
- 在公募REITs低波动背景下，事件信号的微小变化也具有指导意义

#### rationale（理由）
用1句话（50字以内）说明核心理由

---

### 2. detailed_analysis（详细分析）

#### news_analysis（新闻分析）
- **high_score_news**: 数组，仅包含 score=2 的新闻
- 如果没有 score=2 的新闻，设为空数组 `[]`
- 每条新闻包含：
  - **title**: 新闻标题（保留原标题）
  - **score**: 必须为 2
  - **key_info**: 关键信息，简述该新闻的与REITs相关的核心内容（50字以内）

#### fundamental_analysis（基本面分析）
包含季报和运营数据两部分：

**quarterly_report（季报）**：
- **title**: 季报标题简写（如"2024年第3季度报告"）
- **sentiment**: 情绪判断（-1/0/1）
- **key_info**: 关键信息/数据，提取最重要的财务指标或经营数据（100字以内）

**operation_reports（运营数据）**：数组，包含所有运营数据报告
- 如果没有运营数据报告，设为空数组 `[]`
- 每个报告包含：
  - **title**: 运营数据标题简写（如"2024年11月运营数据"）
  - **sentiment**: 情绪判断（-1/0/1）
  - **key_info**: 关键信息/数据，提取最重要的运营指标（100字以内）

#### quarterly_reminder（季报发布预警）
- **has_reminder**: 布尔值
  - 如果有季报预警，设为 `true`
  - 如果没有季报预警，设为 `false`
- **note**: 提示信息
  - 如果有预警，填写提示（如"预计本月20日左右发布新的季报，建议关注机构提前布局"）
  - 如果没有预警，可省略此字段或填空字符串

#### comprehensive_analysis（综合分析）
结合上述新闻、基本面、季报预警等因素综合分析影响，给出投资逻辑和建议（200字以内）

**重要**：
- 使用连贯的段落形式，**不要使用换行符**
- 可以使用逗号、句号等标点符号组织内容
- 基于上述"分析策略指引"，针对当前情况给出分析
- 提供必要的具体数据支撑，避免模棱两可

#### risk_alerts（风险提示）
数组格式，列出1-3个当前最需要关注的风险点，每个风险点用一句话表述

---

## 重要提醒

1. **必须输出有效的JSON格式**，确保所有括号、引号、逗号正确
2. **概率之和必须=1.0**，系统会自动检查
3. **保持一致性**：JSON中的概率判断应与detailed_analysis一致
4. **使用具体数据**：在阐述观点或介绍时，优先使用具体数据作为支撑
5. **控制字数**：严格遵守各字段的字数限制
6. **不使用换行符**：comprehensive_analysis字段必须是连贯的段落，不要包含换行符
7. **参考分析策略指引**：针对当前情况（{situation}）给出最合适的分析
8. **正确处理空数据**：如果没有相关数据（如无score=2新闻、无运营数据、无季报预警），正确使用空数组`[]`或`false`标记
9. **避免过度保守，敢于判断**：根据各维度信号强度给出相应的概率和置信度，不要因为害怕错误而系统性降低数值。请记住：
   **市场特性要求更果断的判断**：
   - 公募REITs波动率远低于股票
   - 整盘（side）是公募REITs常见且正常的状态，判断为整盘时也应给予合理的主导概率和置信度

   **投资者的策略依赖你的预测质量**：
   - 策略目标：在回撤可控的前提下最大化收益
   - 你的角色：提供清晰、有置信度的方向预测，作为交易决策的关键输入
   - "模糊安全"倾向（总是选择中间值或较低值），会显著降低策略的收益潜力
   - 因此在有明确信号（无论上涨、下跌还是整盘）时请给予较高的概率和置信度

   **注意**：这并不意味着盲目乐观，而是要求你基于充分的证据做出匹配的判断强度
10. **检查概率与置信度**：返回前检查 `direction_probs` 和 `confidence` 是否符合字段填写说明的要求，如果发现不符合要求，请根据分析重新打分

请开始分析，输出JSON格式结果。"""

    # ========== 组合完整提示词 ==========
    prompt = intro + news_section + fundamental_section + reminder_section + price_pattern_section + sideways_section + strategy_section + task_section

    return prompt


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
        import json
        cleaned_json = clean_json_response(analysis_json_str)

        # 解析JSON
        data = json.loads(cleaned_json)

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
        import json
        cleaned_json = clean_json_response(analysis_json_str)

        # 解析JSON
        data = json.loads(cleaned_json)

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

        # 检查T+20
        if 'T+20' in direction_judgment:
            periods_to_check.append(('T+20', direction_judgment['T+20']))

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


def call_deepseek_reasoner(prompt: str, min_dominant_prob: float = 0.5) -> Dict[str, str]:
    """
    调用 deepseek-reasoner 模型进行分析（支持概率和验证重试）

    Args:
        prompt: 分析提示词
        min_dominant_prob: 主导方向的最小概率阈值，默认0.5

    Returns:
        包含 reasoning_content 和 content 的字典
    """
    logger.debug("开始调用 deepseek-reasoner 模型（最多尝试3次）")

    # 获取模型配置
    model_config = MODEL_CONFIG['deepseek']['deepseek-reasoner']

    # 初始化客户端
    client = OpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )

    # 最多尝试3次
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"第 {attempt} 次调用 LLM...")

            response = client.chat.completions.create(
                model=model_config['model'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # 提取推理过程和最终内容
            message = response.choices[0].message

            # reasoning_content 是推理过程
            reasoning_content = getattr(message, 'reasoning_content', '')

            # content 是最终输出
            content = message.content

            logger.debug("模型调用成功，开始验证概率和...")
            logger.debug(f"推理过程长度: {len(reasoning_content)} 字符")
            logger.debug(f"输出内容长度: {len(content)} 字符")

            # 验证概率和
            is_valid_sum, error_msg_sum = validate_probability_sum(content)

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
            is_valid_dominant, error_msg_dominant = validate_dominant_probability(content, min_dominant_prob)

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
            # 清理content，去除markdown代码块标记
            cleaned_content = clean_json_response(content)
            return {
                'reasoning_content': reasoning_content,
                'content': cleaned_content  # 返回清理后的内容
            }

        except ValueError:
            # 重新抛出概率验证失败的异常
            raise
        except Exception as e:
            logger.error(f"第 {attempt} 次调用模型失败: {str(e)}")
            last_error = str(e)

            if attempt < max_attempts:
                logger.info(f"准备进行第 {attempt + 1} 次尝试...")
            else:
                logger.error(f"已达最大重试次数（{max_attempts}次）")
                raise Exception(f"调用模型失败（尝试{max_attempts}次）: {last_error}")

    # 理论上不会到达这里
    raise Exception(f"调用模型失败: {last_error}")


def analyze_event(fund_code: str, date: str, min_dominant_prob: float = 0.5) -> Dict[str, Any]:
    """
    事件分析主函数（可被其他脚本调用）

    Args:
        fund_code: 基金代码
        date: 当前日期 (格式: YYYY-MM-DD)
        min_dominant_prob: 主导方向的最小概率阈值，默认0.5

    Returns:
        Dict: {
            'status': 'success' | 'error',
            'analysis_result': LLM返回的JSON字符串,
            'reasoning_process': LLM推理过程,
            'metadata': 元数据信息
        }
    """
    logger.info("=" * 80)
    logger.info(f"开始执行事件分析 - 基金: {fund_code}, 日期: {date}")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # 第1步：获取最近新闻
        logger.info("步骤1: 获取最近的市场新闻")
        news_list = get_recent_news(date, NEWS_DAYS_RANGE)
        logger.info(f"✓ 查询到 {len(news_list)} 条新闻")

        # 第2步：获取季报和运营数据
        logger.info("步骤2: 获取最近季报和运营数据报告")
        quarterly_report, operation_reports = get_latest_quarterly_and_operations(fund_code, date)
        if quarterly_report:
            logger.info(f"✓ 找到最近季报: {quarterly_report['date']} - {quarterly_report['announcement_title']}")
            logger.info(f"✓ 找到 {len(operation_reports)} 条运营数据报告")
        else:
            logger.warning("✗ 未找到季报数据")

        # 第3步：检查季报提醒
        logger.info("步骤3: 检查是否需要季报发布提醒")
        quarterly_reminder = check_quarterly_report_reminder(fund_code, date)
        if quarterly_reminder:
            logger.info(f"✓ 需要提醒: {quarterly_reminder['quarter']}报告即将发布")
        else:
            logger.info("✓ 当前无需季报提醒")

        # 第3.5步：获取价格数据并计算动态阈值
        logger.info("步骤3.5: 获取价格数据并计算动态阈值")
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

        # 第4步：构建提示词
        logger.info("步骤4: 构建分析提示词")
        prompt = build_analysis_prompt(
            news_list,
            quarterly_report,
            operation_reports,
            quarterly_reminder,
            fund_code,
            date,
            volatility_threshold,
            recent_5d_changes,
            recent_20d_data
        )

        if logger.level == logging.DEBUG:
            logger.debug("\n" + "=" * 80)
            logger.debug("完整提示词内容:")
            logger.debug("=" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)

        # 第5步：调用LLM分析
        logger.info("步骤5: 调用 deepseek-reasoner 模型进行分析")
        result = call_deepseek_reasoner(prompt, min_dominant_prob)

        # 记录推理过程
        if result['reasoning_content']:
            logger.info("\n" + "=" * 80)
            logger.info("模型推理过程:")
            logger.info("=" * 80)
            logger.info(result['reasoning_content'])
            logger.info("=" * 80)

        # 记录最终输出
        logger.info("\n" + "=" * 80)
        logger.info("模型分析结果:")
        logger.info("=" * 80)
        logger.info(result['content'])
        logger.info("=" * 80)

        # 计算耗时
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        logger.info(f"\n✓ 事件分析完成，总耗时: {elapsed:.2f} 秒")
        logger.info("=" * 80)

        # 构建返回结果
        return_data = {
            'status': 'success',
            'analysis_result': result['content'],
            'reasoning_process': result['reasoning_content'],
            'metadata': {
                'expert_type': 'event',
                'fund_code': fund_code,
                'analysis_date': date,
                'timestamp': datetime.now().isoformat()
            }
        }

        # 记录最终传出的内容（用于验证）
        logger.info("\n" + "=" * 80)
        logger.info("最终传出的内容（返回给调用方）:")
        logger.info("=" * 80)
        logger.info(f"status: {return_data['status']}")
        logger.info(f"analysis_result 长度: {len(return_data['analysis_result'])} 字符")
        logger.info(f"reasoning_process 长度: {len(return_data['reasoning_process'])} 字符")
        logger.info(f"metadata: {return_data['metadata']}")
        logger.info("=" * 80)

        return return_data

    except Exception as e:
        logger.error(f"事件分析执行失败: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'metadata': {
                'expert_type': 'event',
                'fund_code': fund_code,
                'analysis_date': date,
                'timestamp': datetime.now().isoformat()
            }
        }
