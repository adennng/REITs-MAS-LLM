#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公告影响分析工具
分析特定类型公告对REITs价格的历史影响
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pymysql
from openai import OpenAI

# 导入配置（使用相对路径）
# 获取当前脚本所在目录的父目录（即项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config.db_config import get_db_reits_config
from config.model_config import MODEL_CONFIG


# ============================================================================
# 配置参数 - 可手动修改
# ============================================================================
# LLM提示词中每类公告展示的详细记录数量
MAX_RECORDS_PER_SENTIMENT = 8
# ============================================================================


# 配置日志
def setup_logger(log_file: str = 'announcement_impact_analyzer.log') -> logging.Logger:
    """配置日志记录器"""
    logger = logging.getLogger('AnnouncementImpactAnalyzer')
    logger.setLevel(logging.DEBUG)

    # 确定日志目录（相对路径：../log/）
    log_dir = os.path.join(parent_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 完整的日志文件路径
    log_path = os.path.join(log_dir, log_file)

    # 文件处理器
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

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


def query_announcements(fund_code: str, date: str, doc_type_2) -> List[Dict[str, Any]]:
    """
    从数据库查询符合条件的公告记录

    Args:
        fund_code: 基金代码
        date: 当前日期
        doc_type_2: 公告类型，可以是字符串（单个类型）或列表（多个类型）

    Returns:
        List[Dict]: 查询结果列表
    """
    # 统一处理为列表
    if isinstance(doc_type_2, str):
        doc_type_2_list = [doc_type_2]
    else:
        doc_type_2_list = list(doc_type_2)

    logger.info(f"开始查询数据库 - fund_code: {fund_code}, date: {date}, doc_type_2: {doc_type_2_list}")

    # 获取数据库配置
    db_config = get_db_reits_config()

    # 连接数据库
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # 构建 IN 子句的占位符
            placeholders = ', '.join(['%s'] * len(doc_type_2_list))

            sql = f"""
                SELECT date, sentiment, announcement_title, digest, price_change
                FROM processed_files
                WHERE fund_code = %s
                  AND doc_type_2 IN ({placeholders})
                  AND digest IS NOT NULL
                  AND digest != ''
                  AND date < %s
                ORDER BY date DESC
            """

            # 构建参数：fund_code, doc_type_2_list的各个元素, date
            params = [fund_code] + doc_type_2_list + [date]
            cursor.execute(sql, params)
            results = cursor.fetchall()

            logger.info(f"查询完成，共找到 {len(results)} 条记录")
            logger.debug(f"查询SQL: {sql}")
            logger.debug(f"查询参数: fund_code={fund_code}, doc_type_2={doc_type_2_list}, date={date}")

            return results

    finally:
        connection.close()


def parse_price_change(price_change_str: str) -> Dict[str, Any]:
    """
    解析 price_change JSON 字符串

    Args:
        price_change_str: JSON 字符串

    Returns:
        Dict: 解析后的数据
    """
    try:
        return json.loads(price_change_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"解析 price_change 失败: {e}")
        return {}


def calculate_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算统计数据

    Args:
        records: 查询到的记录列表

    Returns:
        Dict: 统计结果
    """
    logger.info("开始计算统计数据")

    # 初始化计数器
    total_count = len(records)
    sentiment_counts = {1: 0, 0: 0, -1: 0}

    # 按sentiment分组的数据
    grouped_data = {1: [], 0: [], -1: []}

    for record in records:
        sentiment = record['sentiment']
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

        # 解析 price_change
        price_data = parse_price_change(record.get('price_change', ''))

        if price_data and '上涨比率' in price_data:
            grouped_data[sentiment].append({
                'date': record['date'],
                'title': record['announcement_title'],
                'digest': record['digest'],
                'price_change': price_data,
                'up_ratios': price_data['上涨比率']
            })

    # 计算各sentiment下的上涨概率
    def calc_up_probability(data_list: List[Dict], period: str) -> float:
        """计算指定周期的上涨概率（上涨比率>=50%）"""
        if not data_list:
            return 0.0

        up_count = sum(
            1 for item in data_list
            if float(item['up_ratios'].get(period, '0%').strip('%')) >= 50.0
        )

        return (up_count / len(data_list)) * 100

    # 计算平均上涨比率
    def calc_avg_up_ratio(data_list: List[Dict], period: str) -> float:
        """计算指定周期的平均上涨比率"""
        if not data_list:
            return 0.0

        ratios = [
            float(item['up_ratios'].get(period, '0%').strip('%'))
            for item in data_list
        ]

        return sum(ratios) / len(ratios) if ratios else 0.0

    # 计算利好情况下日涨幅>2%的比率
    def calc_big_rise_ratio(data_list: List[Dict]) -> float:
        """计算前5天内至少有一天涨幅>2%的比率"""
        if not data_list:
            return 0.0

        big_rise_count = 0
        for item in data_list:
            price_data = item['price_change'].get('价格数据', [])
            # 检查前5天
            has_big_rise = any(
                float(day['涨跌幅'].strip('%')) > 2.0
                for day in price_data[:5]
            )
            if has_big_rise:
                big_rise_count += 1

        return (big_rise_count / len(data_list)) * 100

    # 计算利空情况下日跌幅>2%的比率
    def calc_big_drop_ratio(data_list: List[Dict]) -> float:
        """计算前5天内至少有一天跌幅>2%的比率（即涨跌幅<-2%）"""
        if not data_list:
            return 0.0

        big_drop_count = 0
        for item in data_list:
            price_data = item['price_change'].get('价格数据', [])
            # 检查前5天
            has_big_drop = any(
                float(day['涨跌幅'].strip('%')) < -2.0
                for day in price_data[:5]
            )
            if has_big_drop:
                big_drop_count += 1

        return (big_drop_count / len(data_list)) * 100

    # 计算各sentiment的统计数据
    periods = ['前1日', '前2日', '前3日', '前5日', '前10日', '前20日']

    statistics = {
        '总计': {
            '公告总数': total_count,
            '利好数量': sentiment_counts[1],
            '中性数量': sentiment_counts[0],
            '利空数量': sentiment_counts[-1]
        }
    }

    sentiment_labels = {1: '利好情况', 0: '中性情况', -1: '利空情况'}

    for sentiment_val, label in sentiment_labels.items():
        data_list = grouped_data[sentiment_val]

        stats = {
            '样本数量': len(data_list)
        }

        # 计算各周期的上涨概率
        for period in periods[:4]:  # 前1日、前2日、前3日、前5日
            prob = calc_up_probability(data_list, period)
            stats[f'{period}上涨概率'] = f'{prob:.2f}%'

        # 计算前10日和前20日的平均上涨比率
        for period in ['前10日', '前20日']:
            avg_ratio = calc_avg_up_ratio(data_list, period)
            stats[f'{period}上涨比率均值'] = f'{avg_ratio:.2f}%'

        # 仅利好情况计算大涨比率
        if sentiment_val == 1:
            big_rise_ratio = calc_big_rise_ratio(data_list)
            stats['日涨幅超2%的比率'] = f'{big_rise_ratio:.2f}%'

        # 仅利空情况计算大跌比率
        if sentiment_val == -1:
            big_drop_ratio = calc_big_drop_ratio(data_list)
            stats['日跌幅超2%的比率'] = f'{big_drop_ratio:.2f}%'

        statistics[label] = stats

    # 计算总体的长期影响
    all_data = []
    for data_list in grouped_data.values():
        all_data.extend(data_list)

    statistics['总体长期影响'] = {
        '前10日上涨比率均值': f'{calc_avg_up_ratio(all_data, "前10日"):.2f}%',
        '前20日上涨比率均值': f'{calc_avg_up_ratio(all_data, "前20日"):.2f}%'
    }

    logger.info("统计计算完成")
    logger.debug(f"统计结果: {json.dumps(statistics, ensure_ascii=False, indent=2)}")

    return statistics, grouped_data


def call_deepseek_reasoner(
    statistics: Dict[str, Any],
    grouped_data: Dict[int, List[Dict]],
    doc_type_2,
    fund_code: str
) -> Tuple[str, str]:
    """
    调用 DeepSeek Reasoner 模型进行分析

    Args:
        statistics: 统计数据
        grouped_data: 按sentiment分组的原始数据
        doc_type_2: 公告类型，可以是字符串或列表
        fund_code: 基金代码

    Returns:
        Tuple[str, str]: (分析结果, 推理过程)
    """
    logger.info("开始调用 DeepSeek Reasoner 模型")

    # 获取模型配置
    model_cfg = MODEL_CONFIG['deepseek']['deepseek-reasoner']

    # 初始化客户端
    client = OpenAI(
        api_key=model_cfg['api_key'],
        base_url=model_cfg['base_url']
    )

    # 格式化公告类型显示
    if isinstance(doc_type_2, str):
        doc_type_2_display = f'"{doc_type_2}"'
    else:
        doc_type_2_display = '、'.join([f'"{t}"' for t in doc_type_2])

    # 构建提示词
    prompt = f"""你是一位专业的REITs投资分析师。现在需要你分析基金代码 {fund_code} 的{doc_type_2_display}类型公告对价格的历史影响，**为投资者提供操作决策支持**。

## 字段说明

### 原始数据字段
- **sentiment**: 情绪判断，表示公告对价格的影响方向
  - 1 = 利好
  - 0 = 中性
  - -1 = 利空

- **digest**: 公告信息的摘要

- **price_change**: JSON格式，包含：
  - 公告日期
  - 公告发布后20个交易日内的收盘价和涨跌幅
  - 上涨比率：公告发布后前1日（公告发布当日）、前2日（公告发布日和之后第1个交易日）、前3日（公告发布日和之后2个交易日）、前5日（公告发布日和之后4个交易日）、前10日（公告发布日和之后9个交易日）、前20日（公告发布日和之后19个交易日）的上涨比率（上涨天数/总天数）

### 统计数据的计算方法（帮助你理解统计数据的含义）

以下统计指标已经由系统预先计算好，你只需要理解其含义并进行分析：

1. **"前X日上涨概率"** 的计算方法：
   - 判断标准：某条记录的price_change中"上涨比率"的"前X日" >= 50% 则认为该记录在前X日整体向上
   - 计算公式：(满足条件的记录数) / (该sentiment的总记录数) × 100%
   - 例如："利好情况"下"前2日上涨概率"为60%，意思是：在所有利好公告中，有60%的公告在发布后前2个交易日的上涨比率>=50%

2. **"前X日上涨比率均值"** 的计算方法：
   - 直接对所有记录的price_change中"上涨比率"的"前X日"字段求平均值
   - 例如："利好情况"下"前10日上涨比率均值"为55%，意思是：所有利好公告发布后前10日的上涨比率平均值为55%

3. **"日涨幅超2%的比率"**（仅利好情况）的计算方法：
   - 遍历price_change中"价格数据"数组的前5天，判断是否至少有一天的涨跌幅 > 2%
   - 计算公式：(前5天内至少有一天涨幅>2%的记录数) / (利好公告总数) × 100%

4. **"日跌幅超2%的比率"**（仅利空情况）的计算方法：
   - 遍历price_change中"价格数据"数组的前5天，判断是否至少有一天的涨跌幅 < -2%
   - 计算公式：(前5天内至少有一天跌幅>2%的记录数) / (利空公告总数) × 100%

**重要提示**：所有统计计算已完成，你无需再次计算，请直接使用统计数据中的数值进行分析。

## 统计数据
{json.dumps(statistics, ensure_ascii=False, indent=2)}

## 历史公告详细记录

### 利好公告 ({len(grouped_data[1])}条)
"""

    # 添加利好公告详情（使用配置的展示数量）
    for i, item in enumerate(grouped_data[1][:MAX_RECORDS_PER_SENTIMENT], 1):
        prompt += f"\n{i}. 日期: {item['date']}\n"
        prompt += f"   标题: {item['title']}\n"
        prompt += f"   摘要: {item['digest']}\n"
        prompt += f"   价格变动数据(price_change): {json.dumps(item['price_change'], ensure_ascii=False)}\n"

    if len(grouped_data[1]) > MAX_RECORDS_PER_SENTIMENT:
        prompt += f"\n（省略剩余 {len(grouped_data[1]) - MAX_RECORDS_PER_SENTIMENT} 条利好公告）\n"

    prompt += f"\n### 中性公告 ({len(grouped_data[0])}条)\n"
    for i, item in enumerate(grouped_data[0][:MAX_RECORDS_PER_SENTIMENT], 1):
        prompt += f"\n{i}. 日期: {item['date']}\n"
        prompt += f"   标题: {item['title']}\n"
        prompt += f"   摘要: {item['digest']}\n"
        prompt += f"   价格变动数据(price_change): {json.dumps(item['price_change'], ensure_ascii=False)}\n"

    if len(grouped_data[0]) > MAX_RECORDS_PER_SENTIMENT:
        prompt += f"\n（省略剩余 {len(grouped_data[0]) - MAX_RECORDS_PER_SENTIMENT} 条中性公告）\n"

    prompt += f"\n### 利空公告 ({len(grouped_data[-1])}条)\n"
    for i, item in enumerate(grouped_data[-1][:MAX_RECORDS_PER_SENTIMENT], 1):
        prompt += f"\n{i}. 日期: {item['date']}\n"
        prompt += f"   标题: {item['title']}\n"
        prompt += f"   摘要: {item['digest']}\n"
        prompt += f"   价格变动数据(price_change): {json.dumps(item['price_change'], ensure_ascii=False)}\n"

    if len(grouped_data[-1]) > MAX_RECORDS_PER_SENTIMENT:
        prompt += f"\n（省略剩余 {len(grouped_data[-1]) - MAX_RECORDS_PER_SENTIMENT} 条利空公告）\n"

    prompt += """

## 分析要求

你正在为投资者提供决策支持。假设**刚刚发布了一份同类型公告**，请基于历史数据给出专业分析和操作建议：

### 1. 历史规律总结
- 该基金历史上发布同类公告的次数（利好/中性/利空各多少次）
- 样本数量是否充足（至少需要3个样本才有参考价值，请明确指出样本是否充足）
- 不同情绪下的典型价格走势特征

### 2. 价格影响的时间窗口分析
- **短期影响**（1-3个交易日）：
  * 利好/中性/利空情况下的上涨概率
  * 是否存在明显的方向性
  * 短期反应是否迅速（当日就有反应 vs 滞后反应）

- **中期影响**（3-5个交易日）：
  * 影响是否持续或衰减
  * 是否出现反转迹象

- **长期影响**（10-20个交易日）：
  * 长期趋势是否显著
  * 是否回归常态

### 3. 涨跌幅度特征
- **利好公告**：
  * 日涨幅>2%的概率（是否容易出现大涨）
  * 典型涨幅区间（根据历史数据推断）
  * 是否存在超预期的大涨案例（查看历史记录）

- **利空公告**：
  * 日跌幅>2%的概率（是否容易出现大跌）
  * 典型跌幅区间（根据历史数据推断）
  * 是否存在超预期的大跌案例（查看历史记录）

- **中性公告**：
  * 价格波动是否真的很小
  * 是否存在"中性公告但价格大幅波动"的异常情况

### 4. 历史规律的可靠性评估 ⭐⭐⭐（重要）
- **一致性检验**：历史表现是否稳定？还是每次差异很大？
- **样本充分性**：样本数量是否足够支撑结论？如果样本不足，请明确警示
- **异常值识别**：是否存在显著偏离常规的案例？需特别说明
- **情绪判断准确性**：
  * 利好公告是否真的带来上涨？准确率大概多少？
  * 利空公告是否真的带来下跌？准确率大概多少？
  * 中性公告是否真的影响不大？
- **置信度评级**：对历史规律的可信度评级（高/中/低），并说明理由

### 5. 投资操作建议 ⭐⭐⭐⭐⭐（核心）

假设**今天刚发布了这类公告**，请根据历史数据给出操作建议：

#### 5.1 如果是利好公告：
- **是否值得买入**？基于历史数据的胜率（上涨概率）和赔率（涨幅大小）综合判断
- **最佳买入时点建议**：
  * 当日买入 vs 次日买入 vs 观察2-3天再买入？
  * 理由：基于历史上价格反应的时间特征和规律
- **预期收益区间**：基于历史数据，合理的收益预期大概是多少？（给出大致范围）
- **持有周期建议**：建议持有多少天？何时考虑止盈？
- **风险提示**：
  * 历史上是否有"利好公告但价格下跌"的反例？比例多少？
  * 当前策略的主要风险在哪里？

#### 5.2 如果是利空公告：
- **是否应该卖出**？基于历史数据的判断
- **操作时点建议**：
  * 应该立即卖出 vs 观察反应 vs 不必恐慌？
  * 理由：基于历史上的价格走势特征
- **预期下跌幅度**：基于历史数据，可能的下跌空间是多少？
- **抄底时机建议**：如果价格下跌，大约在什么时候可以考虑抄底？（基于历史影响持续时间）
- **风险提示**：是否有超预期大跌的风险？

#### 5.3 如果是中性公告：
- **操作建议**：持有不动 vs 观察市场反应
- **是否存在误判风险**：历史上是否有"中性公告但价格大幅波动"的情况？

### 6. 关键风险警示
- 样本数量不足的警示（如果适用）
- 历史规律不稳定的警示（如果适用）
- 特殊情况说明（如某次异常波动及其可能原因）
- **重要声明**：历史表现不代表未来，需结合当前市场环境、基金基本面、宏观经济等因素综合判断

### 7. 结论与决策参考 ⭐⭐⭐
- **一句话总结**：这类公告对价格的影响特征（简洁明了）
- **核心建议**：如果今天发布此类公告，投资者应该___（买入/卖出/观察/持有不动），时点建议___
- **信心等级**：对上述建议的信心程度（高/中/低），基于历史数据的可靠性和样本充分性

---

## 输出要求
1. **数据驱动**：所有结论必须基于统计数据，引用具体数字
2. **客观诚实**：如实指出历史规律的不足之处，不要过度自信
3. **可操作性强**：给出明确的"做什么、何时做"，而不是模糊的"可能、也许"
4. **风险提示充分**：诚实告知历史数据的局限性，样本不足时明确警示
5. **语言简洁专业**：避免冗长描述，突出关键决策点，使用要点式表述
6. **结构清晰**：按照上述部分组织内容，便于投资者快速理解

请给出专业、客观、可操作的分析报告。
"""

    logger.debug(f"发送给LLM的提示词:\n{prompt}")

    try:
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

        logger.info("LLM 调用成功")
        logger.debug(f"LLM 返回的分析结果:\n{analysis}")

        # 记录推理过程到日志（使用INFO级别以确保记录）
        if reasoning:
            logger.info(f"LLM 推理过程:\n{reasoning}")
        else:
            logger.warning("未能获取到LLM的推理过程")

        return analysis, reasoning

    except Exception as e:
        logger.error(f"调用 LLM 失败: {e}")
        raise


def analyze_announcement_impact(
    fund_code: str,
    date: str,
    doc_type_2
) -> Dict[str, Any]:
    """
    分析特定类型公告对价格的历史影响

    Args:
        fund_code: 基金代码
        date: 当前日期 (格式: YYYY-MM-DD)
        doc_type_2: 公告类型，可以是字符串（单个类型）或列表（多个类型）

    Returns:
        Dict: {
            'analysis': LLM分析结果,
            'reasoning': LLM推理过程,
            'statistics': 统计数据,
            'status': 'success' 或 'error',
            'message': 错误信息（如果有）
        }
    """
    logger.info("=" * 80)
    logger.info("开始执行公告影响分析")
    logger.info(f"输入参数 - fund_code: {fund_code}, date: {date}, doc_type_2: {doc_type_2}")

    try:
        # 1. 验证日期格式
        if not validate_date_format(date):
            error_msg = f"日期格式错误: {date}，应为 YYYY-MM-DD 格式"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }

        # 2. 查询数据库
        records = query_announcements(fund_code, date, doc_type_2)

        if not records:
            warning_msg = f"未找到符合条件的公告记录"
            logger.warning(warning_msg)
            return {
                'status': 'success',
                'message': warning_msg,
                'analysis': '暂无历史数据可供分析',
                'reasoning': '',
                'statistics': {}
            }

        # 3. 计算统计数据
        statistics, grouped_data = calculate_statistics(records)

        # 4. 调用 LLM 分析
        analysis, reasoning = call_deepseek_reasoner(
            statistics, grouped_data, doc_type_2, fund_code
        )

        # 5. 返回结果
        result = {
            'status': 'success',
            'analysis': analysis,
            'reasoning': reasoning,
            'statistics': statistics
        }

        logger.info("公告影响分析完成")
        logger.info("=" * 80)

        return result

    except Exception as e:
        error_msg = f"执行过程中发生错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'message': error_msg
        }
