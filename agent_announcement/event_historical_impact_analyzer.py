#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件影响分析工具
分析特定类型事件（如分红公告、基金份额解除限售）对REITs价格的历史影响
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
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config.db_config import get_db_reits_config
from config.model_config import MODEL_CONFIG


# ============================================================================
# 日志配置
# ============================================================================
def setup_logger(log_level: str = 'simple', log_file: str = 'event_impact_analyzer.log') -> logging.Logger:
    """
    配置日志记录器

    Args:
        log_level: 日志级别，'detailed'（详细版）或 'simple'（简单版）
        log_file: 日志文件名

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger('EventImpactAnalyzer')
    logger.setLevel(logging.DEBUG if log_level == 'detailed' else logging.INFO)

    # 清除已有的处理器
    logger.handlers.clear()

    # 确定日志目录
    log_dir = os.path.join(parent_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # 文件处理器
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG if log_level == 'detailed' else logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    if log_level == 'detailed':
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# 全局logger（默认简单版，会在主函数中根据参数重新配置）
logger = setup_logger('simple')


# ============================================================================
# 工具函数
# ============================================================================
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


def parse_json_field(json_str: str) -> Dict[str, Any]:
    """
    解析JSON字符串

    Args:
        json_str: JSON字符串

    Returns:
        Dict: 解析后的数据
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"解析JSON失败: {e}")
        return {}


# ============================================================================
# 数据库查询
# ============================================================================
def query_event_data(fund_code: str, date: str, doc_type_2: str) -> List[Dict[str, Any]]:
    """
    从数据库查询符合条件的事件记录

    Args:
        fund_code: 基金代码
        date: 当前日期
        doc_type_2: 事件类型

    Returns:
        List[Dict]: 查询结果列表
    """
    logger.info(f"开始查询数据库 - fund_code: {fund_code}, date: {date}, doc_type_2: {doc_type_2}")

    # 获取数据库配置
    db_config = get_db_reits_config()

    # 连接数据库
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT event_date, event_price_change
                FROM processed_files
                WHERE fund_code = %s
                  AND doc_type_2 = %s
                  AND event_price_change IS NOT NULL
                  AND event_price_change != ''
                  AND event_date < %s
                ORDER BY event_date DESC
            """

            cursor.execute(sql, (fund_code, doc_type_2, date))
            results = cursor.fetchall()

            logger.info(f"查询完成，共找到 {len(results)} 条记录")
            if logger.level == logging.DEBUG:
                logger.debug(f"查询SQL: {sql}")
                logger.debug(f"查询参数: fund_code={fund_code}, doc_type_2={doc_type_2}, date={date}")

            # 去重：按event_date去重，保留任意一个
            unique_records = {}
            for record in results:
                event_date_str = str(record['event_date'])
                if event_date_str not in unique_records:
                    unique_records[event_date_str] = record

            # 转换回列表，保持原有的排序（从晚到早）
            deduplicated_results = list(unique_records.values())

            if len(deduplicated_results) < len(results):
                logger.info(f"去重后剩余 {len(deduplicated_results)} 条记录（去除了 {len(results) - len(deduplicated_results)} 条重复记录）")

            if logger.level == logging.DEBUG:
                logger.debug(f"去重后的event_date列表: {[str(r['event_date']) for r in deduplicated_results]}")

            return deduplicated_results

    finally:
        connection.close()


# ============================================================================
# 统计计算 - 分红公告
# ============================================================================
def calculate_dividend_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算分红公告的统计数据

    Args:
        records: 查询到的记录列表

    Returns:
        Dict: 统计结果
    """
    logger.info("开始计算分红公告统计数据")

    total_count = len(records)

    # 权益登记日统计
    equity_date_positive = 0  # 涨跌幅为正
    equity_date_positive_gt2 = 0  # 涨跌幅为正且>2%
    equity_date_negative = 0  # 涨跌幅为负
    equity_date_negative_gt2 = 0  # 涨跌幅为负且<-2%

    # 除息日统计
    ex_date_positive = 0
    ex_date_positive_gt2 = 0
    ex_date_negative = 0
    ex_date_negative_gt2 = 0

    # 详细数据（用于传递给LLM）
    detailed_records = []

    for record in records:
        event_data = parse_json_field(record.get('event_price_change', ''))

        if not event_data:
            continue

        detailed_records.append({
            'event_date': str(record['event_date']),
            'event_data': event_data
        })

        # 权益登记日涨跌幅
        equity_change_str = event_data.get('权益登记日涨跌幅', '0%')
        try:
            equity_change = float(equity_change_str.strip('%'))
            if equity_change > 0:
                equity_date_positive += 1
                if equity_change > 2:
                    equity_date_positive_gt2 += 1
            elif equity_change < 0:
                equity_date_negative += 1
                if equity_change < -2:
                    equity_date_negative_gt2 += 1
        except ValueError:
            logger.warning(f"无法解析权益登记日涨跌幅: {equity_change_str}")

        # 除息日涨跌幅
        ex_change_str = event_data.get('除息日涨跌幅', '0%')
        try:
            ex_change = float(ex_change_str.strip('%'))
            if ex_change > 0:
                ex_date_positive += 1
                if ex_change > 2:
                    ex_date_positive_gt2 += 1
            elif ex_change < 0:
                ex_date_negative += 1
                if ex_change < -2:
                    ex_date_negative_gt2 += 1
        except ValueError:
            logger.warning(f"无法解析除息日涨跌幅: {ex_change_str}")

    # 计算占比
    statistics = {
        '总样本数': total_count,
        '权益登记日': {
            '上涨占比': f'{(equity_date_positive / total_count * 100):.2f}%' if total_count > 0 else '0.00%',
            '上涨且涨幅>2%占比': f'{(equity_date_positive_gt2 / equity_date_positive * 100):.2f}%' if equity_date_positive > 0 else '0.00%',
            '下跌占比': f'{(equity_date_negative / total_count * 100):.2f}%' if total_count > 0 else '0.00%',
            '下跌且跌幅>2%占比': f'{(equity_date_negative_gt2 / equity_date_negative * 100):.2f}%' if equity_date_negative > 0 else '0.00%'
        },
        '除息日': {
            '上涨占比': f'{(ex_date_positive / total_count * 100):.2f}%' if total_count > 0 else '0.00%',
            '上涨且涨幅>2%占比': f'{(ex_date_positive_gt2 / ex_date_positive * 100):.2f}%' if ex_date_positive > 0 else '0.00%',
            '下跌占比': f'{(ex_date_negative / total_count * 100):.2f}%' if total_count > 0 else '0.00%',
            '下跌且跌幅>2%占比': f'{(ex_date_negative_gt2 / ex_date_negative * 100):.2f}%' if ex_date_negative > 0 else '0.00%'
        }
    }

    logger.info("分红公告统计计算完成")
    if logger.level == logging.DEBUG:
        logger.debug(f"统计结果: {json.dumps(statistics, ensure_ascii=False, indent=2)}")

    return statistics, detailed_records


# ============================================================================
# 统计计算 - 基金份额解除限售
# ============================================================================
def calculate_unlock_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算基金份额解除限售的统计数据

    Args:
        records: 查询到的记录列表

    Returns:
        Dict: 统计结果
    """
    logger.info("开始计算基金份额解除限售统计数据")

    total_count = len(records)
    periods = ['前1日', '前2日', '前3日', '前5日']

    # 统计各周期上涨概率
    def calc_up_probability(data_list: List[Dict], period: str) -> float:
        """计算指定周期的上涨概率（上涨比率>=50%）"""
        if not data_list:
            return 0.0

        up_count = sum(
            1 for item in data_list
            if float(item['上涨比率'].get(period, '0%').strip('%')) >= 50.0
        )

        return (up_count / len(data_list)) * 100

    # 解析所有记录
    parsed_records = []
    for record in records:
        event_data = parse_json_field(record.get('event_price_change', ''))

        if event_data and '上涨比率' in event_data:
            parsed_records.append({
                'event_date': str(record['event_date']),
                'event_data': event_data,
                '上涨比率': event_data['上涨比率']
            })

    # 计算统计数据
    statistics = {
        '总样本数': total_count,
        '有效样本数': len(parsed_records)
    }

    for period in periods:
        prob = calc_up_probability(parsed_records, period)
        statistics[f'{period}上涨概率'] = f'{prob:.2f}%'

    logger.info("基金份额解除限售统计计算完成")
    if logger.level == logging.DEBUG:
        logger.debug(f"统计结果: {json.dumps(statistics, ensure_ascii=False, indent=2)}")

    return statistics, parsed_records


# ============================================================================
# LLM调用
# ============================================================================
def call_deepseek_reasoner(
    statistics: Dict[str, Any],
    detailed_records: List[Dict],
    doc_type_2: str,
    fund_code: str
) -> Tuple[str, str]:
    """
    调用 DeepSeek Reasoner 模型进行分析

    Args:
        statistics: 统计数据
        detailed_records: 详细记录数据
        doc_type_2: 事件类型
        fund_code: 基金代码

    Returns:
        Tuple[str, str]: (分析结果, 推理过程)
    """
    logger.info(f"开始调用 DeepSeek Reasoner 模型分析 {doc_type_2}")

    # 获取模型配置
    model_cfg = MODEL_CONFIG['deepseek']['deepseek-reasoner']

    # 初始化客户端
    client = OpenAI(
        api_key=model_cfg['api_key'],
        base_url=model_cfg['base_url']
    )

    # 根据事件类型构建不同的提示词
    if doc_type_2 == "分红公告":
        prompt = build_dividend_prompt(statistics, detailed_records, fund_code)
    elif doc_type_2 == "基金份额解除限售":
        prompt = build_unlock_prompt(statistics, detailed_records, fund_code)
    else:
        prompt = build_generic_prompt(statistics, detailed_records, doc_type_2, fund_code)

    if logger.level == logging.DEBUG:
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
        if logger.level == logging.DEBUG:
            logger.debug(f"LLM 返回的分析结果:\n{analysis}")
            if reasoning:
                logger.debug(f"LLM 推理过程:\n{reasoning}")

        return analysis, reasoning

    except Exception as e:
        logger.error(f"调用 LLM 失败: {e}")
        raise


def build_dividend_prompt(
    statistics: Dict[str, Any],
    detailed_records: List[Dict],
    fund_code: str
) -> str:
    """构建分红公告的提示词"""

    prompt = f"""你是一位专业的REITs投资分析师。现在需要你分析基金代码 {fund_code} 的"分红公告"对价格的历史影响，**为投资者提供操作决策支持**。

## 字段说明

### event_price_change字段的含义
该字段是JSON格式，包含分红公告的关键信息和价格变动数据：
- **类型**: 公告类型（"分红公告"）
- **权益登记日**: 股权登记的日期，该日收盘后持有者享有分红权
- **除息日**: 除息交易日，该日起交易价格中不再包含本次分红
- **每份派现**: 每份基金分配的现金金额
- **权益登记日涨跌幅**: 权益登记日当天的价格涨跌幅百分比
- **除息日涨跌幅**: 用除息日当天将分红加回后的复权价格计算出来涨跌幅百分比

### 统计数据的计算规则
以下统计数据已由系统预先计算：

1. **权益登记日/除息日的上涨占比**: 该日涨跌幅为正的记录数 / 总记录数 × 100%
2. **上涨且涨幅>2%占比**: 在上涨的记录中，涨幅超过2%的记录数 / 上涨记录总数 × 100%
3. **下跌占比**: 该日涨跌幅为负的记录数 / 总记录数 × 100%
4. **下跌且跌幅>2%占比**: 在下跌的记录中，跌幅超过2%的记录数 / 下跌记录总数 × 100%

**重要提示**：所有统计计算已完成，你无需再次计算，请直接使用统计数据中的数值进行分析。

## 统计数据
{json.dumps(statistics, ensure_ascii=False, indent=2)}

## 历史分红公告详细记录（共{len(detailed_records)}条）
"""

    # 添加详细记录（最多展示10条）
    max_display = min(10, len(detailed_records))
    for i, item in enumerate(detailed_records[:max_display], 1):
        prompt += f"\n{i}. 公告日期: {item['event_date']}\n"
        prompt += f"   详细数据: {json.dumps(item['event_data'], ensure_ascii=False)}\n"

    if len(detailed_records) > max_display:
        prompt += f"\n（省略剩余 {len(detailed_records) - max_display} 条记录）\n"

    prompt += """

## 分析要求

你正在为投资者提供决策支持。假设**刚刚发布了一份分红公告**，请基于历史数据给出专业分析和操作建议：

### 1. 历史规律总结
- 该基金历史上发布分红公告的次数
- 样本数量是否充足（至少需要3个样本才有参考价值，请明确指出样本是否充足）
- 权益登记日和除息日的价格表现特征

### 2. 权益登记日价格影响分析
- **上涨概率**: 权益登记日上涨的占比是多少？
- **涨幅特征**: 上涨时通常涨幅如何？容易出现大涨（>2%）吗？
- **下跌风险**: 下跌的概率和幅度如何？
- **规律性判断**: 权益登记日的价格变动是否有明显规律？

### 3. 除息日价格影响分析
- **上涨概率**: 除息日上涨的占比是多少？
- **涨幅特征**: 上涨时通常涨幅如何？容易出现大涨（>2%）吗？
- **下跌风险**: 下跌的概率和幅度如何？
- **除权影响**: 除息日价格下跌是否主要因为除权因素？

### 5. 历史规律的可靠性评估 ⭐⭐⭐（重要）
- **样本充分性**: 样本数量是否足够支撑结论？如果样本不足，请明确警示
- **一致性检验**: 历史表现是否稳定？还是每次差异很大？
- **异常值识别**: 是否存在显著偏离常规的案例？需特别说明
- **置信度评级**: 对历史规律的可信度评级（高/中/低），并说明理由

### 6. 投资操作建议 ⭐⭐⭐⭐⭐（核心）

假设**今天刚发布了分红公告**，请根据历史数据给出操作建议：

#### 6.1 交易时机选择
- **权益登记日前**: 是否应该提前买入以获得分红权？基于历史上涨概率判断
- **权益登记日当天**: 适合买入还是观望？
- **除息日**: 是否存在交易机会？除息后是否值得买入？
- **最佳操作时点**: 根据历史数据，最佳的买入/卖出时点是什么？

#### 6.2 预期收益与风险
- **预期收益**: 基于历史数据，合理的收益预期是多少？
- **主要风险**: 历史上是否有"分红公告但价格下跌"的反例？比例多少？
- **风险警示**: 当前策略的主要风险在哪里？

#### 6.3 具体操作建议⭐⭐⭐
- **一句话总结**: 分红公告对该基金价格的历史影响特征
- **核心建议**：给出明确的"做什么、何时做"
- **信心等级**: 对上述建议的信心程度（高/中/低），基于历史数据的可靠性和样本充分性

### 7. 关键风险警示
- 样本数量不足的警示（如果适用）
- 历史规律不稳定的警示（如果适用）
- **重要声明**: 历史表现不代表未来，分红公告的价格影响还受市场环境、基金基本面、投资者预期等多重因素影响
---

## 输出要求
1. **数据驱动**: 所有结论必须基于统计数据，引用具体数字
2. **客观诚实**: 如实指出历史规律的不足之处，不要过度自信
3. **可操作性强**: 给出明确的操作建议，而不是模糊的可能性
4. **风险提示充分**: 诚实告知历史数据的局限性，样本不足时明确警示
5. **语言简洁专业**: 避免冗长描述，突出关键决策点，使用要点式表述
6. **结构清晰**: 按照上述部分组织内容，便于投资者快速理解

请给出专业、客观、可操作的分析报告。
"""

    return prompt


def build_unlock_prompt(
    statistics: Dict[str, Any],
    detailed_records: List[Dict],
    fund_code: str
) -> str:
    """构建基金份额解除限售的提示词"""

    prompt = f"""你是一位专业的REITs投资分析师。现在需要你分析基金代码 {fund_code} 的"基金份额解除限售"事件对价格的历史影响，**为投资者提供操作决策支持**。

## 字段说明

### event_price_change字段的含义
该字段是JSON格式，包含基金份额解除限售的关键信息和价格变动数据：
- **类型**: 事件类型（"基金份额解除限售"）
- **生效日**: 限售份额解除限售的生效日期
- **价格数据**: 生效日及之后连续交易日的价格数据数组，每个元素包含：
  - 日期: 交易日期
  - 收盘价: 当日收盘价
  - 涨跌幅: 当日涨跌幅百分比
- **上涨比率**: 生效后不同时间窗口的上涨比率
  - 前1日: 生效日当日上涨的概率（即当日涨跌幅>0的概率）
  - 前2日: 生效日和之后第1个交易日中，上涨天数占比（上涨天数/2）
  - 前3日: 生效日和之后2个交易日中，上涨天数占比（上涨天数/3）
  - 前5日: 生效日和之后4个交易日中，上涨天数占比（上涨天数/5）

### 统计数据的计算规则
以下统计数据已由系统预先计算：

1. **"前X日上涨概率"的计算方法**:
   - 判断标准: 某条记录的"上涨比率"中"前X日" >= 50% 则认为该事件在前X日整体向上
   - 计算公式: (满足条件的记录数) / (总记录数) × 100%
   - 例如: "前2日上涨概率"为60%，意思是：在所有解除限售事件中，有60%的事件在生效后前2个交易日的上涨比率>=50%

**重要提示**: 所有统计计算已完成，你无需再次计算，请直接使用统计数据中的数值进行分析。

## 统计数据
{json.dumps(statistics, ensure_ascii=False, indent=2)}

## 历史基金份额解除限售详细记录（共{len(detailed_records)}条）
"""

    # 添加详细记录（最多展示8条）
    max_display = min(8, len(detailed_records))
    for i, item in enumerate(detailed_records[:max_display], 1):
        prompt += f"\n{i}. 生效日期: {item['event_date']}\n"
        prompt += f"   详细数据: {json.dumps(item['event_data'], ensure_ascii=False)}\n"

    if len(detailed_records) > max_display:
        prompt += f"\n（省略剩余 {len(detailed_records) - max_display} 条记录）\n"

    prompt += """

## 分析要求

你正在为投资者提供决策支持。假设**刚刚发布了一份基金份额解除限售公告**，请基于历史数据给出专业分析和操作建议：

### 1. 历史规律总结
- 该基金历史上发生基金份额解除限售的次数
- 样本数量是否充足（至少需要3个样本才有参考价值，请明确指出样本是否充足）
- 解除限售生效后的典型价格走势特征

### 2. 价格影响的时间窗口分析
- **生效当日（前1日）**:
  * 上涨概率是多少？
  * 是否存在明显的方向性？
  * 市场反应是积极还是消极？

- **短期影响（前2-3日）**:
  * 短期内上涨概率如何？
  * 生效后价格是持续调整还是快速恢复？

- **中期影响（前5日）**:
  * 5日内的上涨概率如何？
  * 解除限售的负面影响是否持续？还是市场很快消化？

### 3. 涨跌幅度特征分析
- 根据历史"价格数据"，分析：
  * 生效当日通常涨跌幅度如何？
  * 是否出现过单日大跌（跌幅>2%或>3%）的情况？频率如何？
  * 是否出现过逆势上涨的情况？什么原因？
  * 后续交易日的波动特征如何？

### 4. 事件对价格的短期影响评估
- **整体影响倾向**: 解除限售是利空、中性还是利好？基于数据判断
- **影响持续时间**: 负面影响（如果有）通常持续多久？
- **市场消化能力**: 市场对这类事件的反应是迅速消化还是长期压制？

### 5. 历史规律的可靠性评估 ⭐⭐⭐（重要）
- **样本充分性**: 样本数量是否足够支撑结论？如果样本不足，请明确警示
- **一致性检验**: 历史表现是否稳定？还是每次差异很大？
- **异常值识别**: 是否存在显著偏离常规的案例？需特别说明
- **置信度评级**: 对历史规律的可信度评级（高/中/低），并说明理由

### 6. 投资操作建议 ⭐⭐⭐⭐⭐（核心）

假设**今天刚发布了基金份额解除限售公告**（或即将生效），请根据历史数据给出操作建议：

#### 6.1 持有者的操作建议
- **是否应该提前卖出**？基于历史数据判断生效后下跌的概率和幅度
- **卖出时机选择**:
  * 公告日当天卖出 vs 等到生效日前卖出 vs 观察生效后再决定？
  * 理由：基于历史价格反应的时间特征
- **继续持有的风险**: 如果选择继续持有，潜在的最大回撤是多少？

#### 6.2 潜在买入者的操作建议
- **是否存在买入机会**？解除限售是否会造成价格下跌从而形成买入机会？
- **最佳买入时点**:
  * 生效日前 vs 生效日当天 vs 生效后观察2-3天？
  * 理由：基于历史上价格下跌和恢复的时间特征
- **预期收益与风险**: 基于历史数据，抄底的胜率和预期收益如何？

#### 6.3 具体操作建议
- **一句话总结**: 基金份额解除限售对该基金价格的历史影响特征
- **核心建议**:
  * 持有者应该___（继续持有/提前卖出/观察后决定），时点建议___
  * 潜在买入者应该___（等待买入机会/不建议参与/观察市场反应）
- **信心等级**: 对上述建议的信心程度（高/中/低），基于历史数据的可靠性和样本充分性

### 7. 关键风险警示
- 样本数量不足的警示（如果适用）
- 历史规律不稳定的警示（如果适用）
- 特殊情况说明（如某次异常波动及其可能原因）
- **重要声明**: 历史表现不代表未来，解除限售的价格影响还受解除规模、市场环境、基金基本面等多重因素影响

---

## 输出要求
1. **数据驱动**: 所有结论必须基于统计数据，引用具体数字
2. **客观诚实**: 如实指出历史规律的不足之处，不要过度自信
3. **可操作性强**: 给出明确的操作建议，而不是模糊的可能性
4. **风险提示充分**: 诚实告知历史数据的局限性，样本不足时明确警示
5. **语言简洁专业**: 避免冗长描述，突出关键决策点，使用要点式表述
6. **结构清晰**: 按照上述部分组织内容，便于投资者快速理解

请给出专业、客观、可操作的分析报告。
"""

    return prompt


def build_generic_prompt(
    statistics: Dict[str, Any],
    detailed_records: List[Dict],
    doc_type_2: str,
    fund_code: str
) -> str:
    """构建通用事件类型的提示词"""

    prompt = f"""你是一位专业的REITs投资分析师。现在需要你分析基金代码 {fund_code} 的"{doc_type_2}"类型事件对价格的历史影响。

## 统计数据
{json.dumps(statistics, ensure_ascii=False, indent=2)}

## 历史事件详细记录（共{len(detailed_records)}条）
"""

    for i, item in enumerate(detailed_records[:10], 1):
        prompt += f"\n{i}. 事件日期: {item.get('event_date', 'N/A')}\n"
        prompt += f"   详细数据: {json.dumps(item.get('event_data', {}), ensure_ascii=False)}\n"

    prompt += """

请基于上述数据分析该类型事件对价格的历史影响，并给出投资建议。
"""

    return prompt


# ============================================================================
# 主函数
# ============================================================================
def analyze_event_impact(
    fund_code: str,
    date: str,
    doc_type_2: str,
    return_reasoning: bool = False,
    log_level: str = 'simple'
) -> Dict[str, Any]:
    """
    分析特定类型事件对价格的历史影响

    Args:
        fund_code: 基金代码
        date: 当前日期 (格式: YYYY-MM-DD)
        doc_type_2: 事件类型（如"分红公告"、"基金份额解除限售"）
        return_reasoning: 是否返回LLM推理过程（默认False）
        log_level: 日志级别，'detailed'（详细版）或 'simple'（简单版），默认'simple'

    Returns:
        Dict: {
            'status': 'success' / 'error' / 'no_data',
            'analysis': LLM分析结果,
            'reasoning': LLM推理过程（如果return_reasoning=True）,
            'message': 提示信息
        }
    """
    # 重新配置logger
    global logger
    logger = setup_logger(log_level)

    logger.info("=" * 80)
    logger.info("开始执行事件影响分析")
    logger.info(f"输入参数 - fund_code: {fund_code}, date: {date}, doc_type_2: {doc_type_2}")
    logger.info(f"配置参数 - return_reasoning: {return_reasoning}, log_level: {log_level}")

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
        records = query_event_data(fund_code, date, doc_type_2)

        if not records:
            warning_msg = f"未找到符合条件的历史事件记录"
            logger.warning(warning_msg)
            result = {
                'status': 'no_data',
                'message': warning_msg,
                'analysis': '暂无历史数据可供分析，无法给出投资建议。建议积累更多历史数据后再进行分析。'
            }
            if return_reasoning:
                result['reasoning'] = ''

            logger.info(f"传出参数 - status: {result['status']}, analysis: {result['analysis']}")
            return result

        # 3. 根据事件类型计算统计数据
        if doc_type_2 == "分红公告":
            statistics, detailed_records = calculate_dividend_statistics(records)
        elif doc_type_2 == "基金份额解除限售":
            statistics, detailed_records = calculate_unlock_statistics(records)
        else:
            # 通用处理（可根据需要扩展）
            logger.warning(f"未识别的事件类型: {doc_type_2}，使用通用处理")
            statistics = {'总样本数': len(records)}
            detailed_records = [
                {
                    'event_date': str(record['event_date']),
                    'event_data': parse_json_field(record.get('event_price_change', ''))
                }
                for record in records
            ]

        # 4. 在详细日志中记录统计数据
        if logger.level == logging.DEBUG:
            logger.debug(f"传递给LLM的统计数据: {json.dumps(statistics, ensure_ascii=False, indent=2)}")

        # 5. 调用 LLM 分析
        analysis, reasoning = call_deepseek_reasoner(
            statistics, detailed_records, doc_type_2, fund_code
        )

        # 6. 返回结果
        result = {
            'status': 'success',
            'analysis': analysis
        }

        # 根据参数决定是否返回推理过程
        if return_reasoning:
            result['reasoning'] = reasoning

        # 记录传出参数
        logger.info(f"传出参数 - status: {result['status']}, analysis长度: {len(analysis)} 字符")
        if return_reasoning:
            logger.info(f"传出参数 - reasoning长度: {len(reasoning) if reasoning else 0} 字符")

        logger.info("事件影响分析完成")
        logger.info("=" * 80)

        return result

    except Exception as e:
        error_msg = f"执行过程中发生错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'message': error_msg
        }
