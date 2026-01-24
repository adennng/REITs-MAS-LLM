#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向预测模块
调用LLM（deepseek-reasoner）进行方向预测
"""

import sys
import os
import json
import logging
from typing import Dict, Any, Tuple
from openai import OpenAI

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from config.model_config import MODEL_CONFIG  # noqa: E402
from agent_direction import config  # noqa: E402

# 获取logger
logger = logging.getLogger(__name__)


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


def validate_probability_sum(prediction_json: Dict[str, Any]) -> Tuple[bool, str]:
    """
    验证direction_judgment中所有期限的概率和是否等于1.0

    Args:
        prediction_json: 预测结果的JSON（已解析为dict）

    Returns:
        Tuple[bool, str]: (验证是否通过, 错误信息)
    """
    try:
        # 检查是否有predictions字段
        if 'predictions' not in prediction_json:
            return False, "缺少predictions字段"

        predictions = prediction_json['predictions']

        # 定义需要检查的期限
        periods_to_check = []

        # 检查T+1和T+5
        if 'T+1' in predictions:
            periods_to_check.append(('T+1', predictions['T+1']))
        if 'T+5' in predictions:
            periods_to_check.append(('T+5', predictions['T+5']))

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

    except Exception as e:
        return False, f"验证过程发生错误: {str(e)}"


def validate_dominant_probability(prediction_json: Dict[str, Any], min_dominant_prob: float = 0.5) -> Tuple[bool, str]:
    """
    验证每个时间维度的主导方向概率是否大于等于指定阈值

    Args:
        prediction_json: 预测结果的JSON（已解析为dict）
        min_dominant_prob: 主导方向的最小概率阈值，默认0.5

    Returns:
        Tuple[bool, str]: (验证是否通过, 错误信息)
    """
    try:
        # 检查是否有predictions字段
        if 'predictions' not in prediction_json:
            return False, "缺少predictions字段"

        predictions = prediction_json['predictions']

        # 定义需要检查的期限
        periods_to_check = []

        # 检查T+1、T+5和T+20
        if 'T+1' in predictions:
            periods_to_check.append(('T+1', predictions['T+1']))
        if 'T+5' in predictions:
            periods_to_check.append(('T+5', predictions['T+5']))
        if 'T+20' in predictions:
            periods_to_check.append(('T+20', predictions['T+20']))

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

    except Exception as e:
        return False, f"主导概率验证过程发生错误: {str(e)}"


def build_direction_prompt(
    fund_code: str,
    date: str,
    four_experts_results: Dict[str, Any],
    price_context: Dict[str, Any],
    volatility_threshold: float = None
) -> str:
    """
    构建方向预测的prompt

    Args:
        fund_code: 基金代码
        date: 分析日期
        four_experts_results: 四个专家的结果
        price_context: 价格上下文
        volatility_threshold: 动态波动阈值

    Returns:
        str: 完整的prompt
    """
    # 解析市场专家的象限信息
    current_quadrant = "xx象限"  # 默认值
    try:
        if 'market' in four_experts_results:
            market_expert = four_experts_results['market']
            # 尝试解析JSON字符串
            if isinstance(market_expert, str):
                market_data = json.loads(market_expert)
            else:
                market_data = market_expert

            # 提取象限信息
            quadrant_info = market_data.get('detailed_analysis', {}).get('quadrant_info', {})
            quadrant_raw = quadrant_info.get('current_quadrant', '')

            # 从字符串中提取象限标识（可能的格式："第III象限。xxx" 或 "III。xxx" 或 "过渡区。xxx"）
            if quadrant_raw:
                # 提取第一个句号前的内容
                quadrant_part = quadrant_raw.split('。')[0].strip() if '。' in quadrant_raw else quadrant_raw.strip()
                if quadrant_part:
                    current_quadrant = quadrant_part
    except Exception as e:
        logger.warning(f"解析市场专家象限信息失败: {e}，使用默认值")

    # 格式化四专家输出
    experts_section = "## 四个专家的分析结果\n\n"

    expert_names = {
        'announcement': '公告专家',
        'market': '市场专家',
        'price': '价格动量专家',
        'event': '事件专家'
    }

    for expert_key in ['announcement', 'market', 'price', 'event']:
        if expert_key in four_experts_results:
            expert_data = four_experts_results[expert_key]
            expert_name = expert_names.get(expert_key, expert_key)

            experts_section += f"### {expert_name}\n\n"
            experts_section += f"```json\n{json.dumps(expert_data, ensure_ascii=False, indent=2)}\n```\n\n"

    # 构建价格与交易上下文部分
    # 第一部分：最近20个交易日价格数据（JSON格式）
    recent_20d_prices = []
    recent_dates = price_context.get('recent_dates', [])
    recent_close = price_context.get('recent_close', [])

    # 构建每个交易日的数据（包含日期、收盘价、涨跌幅）
    for i, (date, close) in enumerate(zip(recent_dates, recent_close)):
        # 计算相对前一日的涨跌幅
        change_pct = None
        if i > 0:
            prev_close = recent_close[i - 1]
            change_pct = (close - prev_close) / prev_close

        recent_20d_prices.append({
            "date": date,
            "close": round(close, 3),  # 保留3位小数
            "change_pct": round(change_pct, 4) if change_pct is not None else None  # 保留4位小数
        })

    # 获取涨跌幅数据
    price_change_5d = price_context.get('price_change_5d')
    price_change_20d = price_context.get('price_change_20d')

    # 构建JSON字符串（格式化输出）
    price_json_str = json.dumps({
        "recent_20d_prices": recent_20d_prices,
        "price_change_5d": round(price_change_5d, 4) if price_change_5d is not None else None,
        "price_change_20d": round(price_change_20d, 4) if price_change_20d is not None else None
    }, ensure_ascii=False, indent=2)

    # 第二部分：整盘判断专项说明
    # 计算最近5个交易日的涨跌幅和突破情况
    threshold_pct = volatility_threshold * 100 if volatility_threshold else 0.38
    recent_5d_details = []
    breakthrough_count = 0
    consolidation_count = 0

    # 只取最近5个交易日
    recent_5_dates = recent_dates[-5:] if len(recent_dates) >= 5 else recent_dates
    recent_5_close = recent_close[-5:] if len(recent_close) >= 5 else recent_close

    for i in range(len(recent_5_dates)):
        date = recent_5_dates[i]
        close = recent_5_close[i]

        # 计算涨跌幅
        if i == 0 and len(recent_close) >= 6:
            # 第一天相对于前一天（即recent_close[-6]）
            prev_idx = len(recent_close) - 6
            change_pct = (close - recent_close[prev_idx]) / recent_close[prev_idx]
        elif i > 0:
            # 后续几天相对于前一天
            change_pct = (close - recent_5_close[i - 1]) / recent_5_close[i - 1]
        else:
            # 数据不足，无法计算
            change_pct = None

        # 判断是否突破整盘阈值
        if change_pct is not None:
            is_breakthrough = abs(change_pct) > (threshold_pct / 100)
            if is_breakthrough:
                breakthrough_count += 1
                if change_pct > 0:
                    status = f"突破整盘阈值{threshold_pct:.2f}%，显著上涨"
                else:
                    status = f"突破整盘阈值{threshold_pct:.2f}%，显著下跌"
            else:
                consolidation_count += 1
                status = f"未突破整盘阈值{threshold_pct:.2f}%"

            recent_5d_details.append(f"- {date}: 涨跌幅 {change_pct*100:+.2f}%（{status}）")
        else:
            recent_5d_details.append(f"- {date}: 涨跌幅 无数据")

    recent_5d_summary = f"最近5个交易日中，{breakthrough_count}天突破整盘阈值，{consolidation_count}天处于整盘状态"

    price_section = f"""## 价格与交易上下文

```json
{price_json_str}
```

**说明**：
- `recent_20d_prices`: 最近20个交易日收盘价和每日涨跌幅（从旧到新）
  - `date`: 交易日期
  - `close`: 收盘价
  - `change_pct`: 相对前一日的涨跌幅
- `price_change_5d`: 最近5日涨跌幅
- `price_change_20d`: 最近20日涨跌幅


## 整盘判断专项说明

### 一、动态阈值计算方法
本系统使用每日动态阈值方案判断价格变动显著性（即是否处于整盘状态）：
- 基于历史波动率、自适应乘数和边界约束计算

### 二、当前动态阈值
当前阈值：{threshold_pct:.2f}%
- 预计日收益率绝对值 > {threshold_pct:.2f}% → 显著变动（上涨或下跌）
- 预计日收益率绝对值 ≤ {threshold_pct:.2f}% → 横盘震荡（整盘）

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
{chr(10).join(recent_5d_details)}

小结：{recent_5d_summary}

### 六、本次预测的整盘判断条件

**不同时间维度的整盘判断标准：**

1. **T+1（次日预测）**：
   - 判断标准：预计次日收益率绝对值 ≤ θₜ（当前动态阈值 {threshold_pct:.2f}%）
   - 即：|次日收益率| ≤ {threshold_pct:.2f}% → 判断为整盘

2. **T+5（未来5个交易日）**：
   - 阈值扩展系数：ε₅ = √5 · θₜ = {threshold_pct * config.THRESHOLD_MULTIPLIER_T5:.2f}%
   - 判断标准：预计未来5个交易日的累计波动幅度 ≤ ε₅
   - 或等价于：未来5日的平均日收益率绝对值 ≤ θₜ
   - 解释：基于价格随机游走假设，多日波动率按时间的平方根缩放

3. **T+20（未来20个交易日）**：
   - 阈值扩展系数：ε₂₀ = √20 · θₜ = {threshold_pct * config.THRESHOLD_MULTIPLIER_T20:.2f}%
   - 判断标准：预计未来20个交易日的累计波动幅度 ≤ ε₂₀
   - 或等价于：未来20日的平均日收益率绝对值 ≤ θₜ
   - 解释：同样基于波动率的时间平方根缩放规律

**重要说明：**
- 整盘不等于不确定，是正常的市场状态，代表价格走势不显著
- 当判断为整盘时，应给予合理的主导概率和合理的置信度
- 不要因为预判整盘就系统性降低置信度

"""

    # 构建完整prompt
    prompt = f"""你是一位专业的REITs投资分析师，**不是结果整理专家**，而是需要基于全部原始信息进行独立判断的投资分析专家！

基金 {fund_code} 在 {date} 时点，你需要综合分析四个领域专家提供的**全部**信息（包括结果和详细分析内容）、近期价格行为，预测未来短期价格方向。

## 背景

这是多agent投资决策系统，你负责方向预测，决策agent会综合你的预测和风控信息做出最终交易策略。

## 你的角色定位

**核心原则：独立思考，深度分析，不完全依赖现成结果**

## 分析立场

1.  **避免过度保守，敢于判断**：根据各维度信号强度给出相应的概率和置信度，不要因为害怕错误而系统性降低数值。请记住：
   **市场特性要求更果断的判断**：
   - 公募REITs波动率远低于股票
   - 整盘（side）是公募REITs常见且正常的状态，判断为整盘时也应给予合理的主导概率和置信度

   **投资者的策略依赖你的预测质量**：
   - 策略目标：在回撤可控的前提下最大化收益
   - 你的角色：提供清晰、有置信度的方向预测，作为交易决策的关键输入
   - "模糊安全"倾向（总是选择中间值或较低值），会显著降低策略的收益潜力
   - 因此在有明确信号（无论上涨、下跌还是整盘）时请给予较高的概率和置信度

   **注意**：这并不意味着盲目乐观，而是要求你基于充分的证据做出匹配的判断强度

---

{experts_section}

{price_section}

---

## 核心分析框架与步骤

### 第一步：解构与验证

1.  **深度审查 `detailed_analysis`**：逐一阅读四位专家的详细分析 (`detailed_analysis`)。这部分内容是你的主要信息来源，比他们给出的直接结论 (`direction_judgment`) 更重要
2.  **寻找关键驱动因素**：从每个 `detailed_analysis` 中提炼出影响未来价格走势的核心信息、关键数据和主要逻辑
3.  **批判性评估**：基于你提取的关键信息，独立判断每位专家给出的 `direction_probs` 的主导方向是否合理
4. 了解近20个交易日价格变动整体情况，以及整盘判断专项说明中的关键信息（含当前动态阈值、最近5日整盘突破情况、预测条件、公募reits价格波动特点等）

### 第二步：综合与权衡

1.  **识别主要矛盾和一致性**：找出不同专家分析之间的冲突点和一致点
2.  **评估影响权重**：综合考虑关键信息的类型、专家置信度水平、信息重要程度、同时结合历史价格变动情况和整盘数据等因素，判断在不同时间维度下，各类信息的影响力主次。你需要为 T+1，T+5，T+20 三个周期分别权衡
  - T+1(超短期):价格动量、市场情绪和突发事件/公告的影响权重可能相对高
  - T+5(短期):是短期情绪与宏观环境/基本面信息开始交织的时期。需判断技术面趋势能否持续，以及公告/事件的影响是否开始发酵
  - T+20(中期):基本面(财报、运营数据)、宏观环境的影响占据主导地位。短期技术扰动和市场情绪的权重降低
3.  **形成核心观点**：综合所有信息，处理完矛盾和权重后，形成你自己对未来走势的核心判断

### 第三步：量化与输出

1.  **确定主导方向**：基于你的核心观点，确定 `T+1` 、 `T+5` 、`T+20`的主导方向（up, down, or side）。
2.  **分配概率 (`direction_probs`)**：将你的判断转化为具体的概率。请注意，各专家的 direction_probs 的具体数值**仅供参考**，请不要完全依赖，你需要根据分析重新打分
3.  **设定置信度 (`confidence`)**：根据你分析过程中的确定性来设定。请注意，各专家的 confidence 的具体数值**仅供参考**，请不要完全依赖，你需要根据分析重新打分
4.  **撰写核心理由 (`rationale`)**：用一句话高度概括你的核心观点

### 第四步：检查与调整

1.  **检查概率与置信度**：检查 `direction_probs` 和 `confidence` 是否符合字段填写说明的要求，如果发现不符合要求，请根据分析重新打分

---

## 分析任务

你必须**严格按照以下JSON格式**输出预测结果。输出内容必须是**纯JSON格式**。

### 输出格式示例：

{{
  "predictions": {{
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
    }},
  "meta": {{
    "analysis_date": "{date}",
    "fund_code": "{fund_code}"
  }}
}}

---

## 字段填写说明——严格遵守

### 1. predictions（方向预测）

#### direction_probs（方向概率）
- **必须满足**：`up + down + side = 1.0`
- **主导方向概率范围**：[0.50, 0.90]，主导方向的概率必须大于另外两个方向，不要出现两个相同的最大概率
- **避免极端概率**：任何单一方向概率不应低于0.05
- **含义**：
  - `up`: 上涨概率
  - `down`: 下跌概率
  - `side`: 横盘概率
-  **整盘的判断**：由于公募REITs波动率较低，本系统使用每日价格波动是否超过动态阈值来判断是否处于整盘状态，请充分考虑这一点，在确定主导方向时除了分析关键信息以外还需结合动态阈值、最近整盘突破值等来综合判断

**强信号**（多项指标共振、方向一致，或者存在矛盾但影响最大的维度信号明确）：
- 主导方向概率：≥ 0.70
- 主导概率与次方向差距：≥ 0.45

**中等信号**（主要指标明确但有少量矛盾）：
- 主导方向概率：0.60 - 0.65
- 主导概率与次方向差距：≥ 0.25

**弱信号**（信号模糊或矛盾较多）：
- 主导方向概率：0 - 0.55
- 主导概率与次方向差距：≥ 0.10

#### confidence（置信度）
反映你对该判断的整体信心（0.00-1.00）：

**强信号**（多项指标共振）：
- 置信度：≥ 0.70
- 情况：多个维度的detailed_analysis都支持同一结论，四专家高度一致，或者存在矛盾但影响最大的维度信号明确

**中等信号**（主要指标明确但有矛盾）：
- 置信度：0.60 - 0.65
- 情况：主要维度支持，四专家中度一致，但存在次要矛盾或不确定性

**弱信号**（信号模糊）：
- 置信度：0.50 - 0.55
- 情况：信号模糊或存在分歧，四专家分歧程度一般

**极弱信号**（矛盾严重）：
- 置信度：< 0.5
- 情况：信息不足、矛盾较多、四专家分歧较大、或市场处于关键转折点

#### rationale（理由）
用1句话（50字以内）说明核心理由，需引用具体的四专家观点
- 引用专家的分析结果或detailed_analysis中的**具体数据/事实**
- 说明你的核心判断依据

**好的示例**：
- "公告披露Q3净利增25%，价格动量MACD金叉，市场情绪偏暖，短期上行概率大"
- "虽四专家偏多，但分红率仅2.8%低于行业均值，价格已涨15%透支预期，谨慎看平"

**差的示例**：
- "四专家一致看多" （只是复述结果，没有独立分析）

### 2. meta（元数据）

包含分析日期和基金代码，保持不变。

## 输出要求

1. **必须输出有效的JSON格式**，确保所有括号、引号、逗号正确
2. **概率之和必须=1.0**，系统会自动检查
3. **控制字数**：严格遵守各字段的字数限制

请开始分析，输出JSON格式结果。
"""

    return prompt


def predict_direction(
    fund_code: str,
    date: str,
    four_experts_results: Dict[str, Any],
    price_context: Dict[str, Any],
    volatility_threshold: float = None,
    max_retries: int = 3,
    show_reasoning: bool = True,
    min_dominant_prob: float = 0.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    调用LLM进行方向预测

    Args:
        fund_code: 基金代码
        date: 分析日期
        four_experts_results: 四个专家的结果
        price_context: 价格上下文
        volatility_threshold: 动态波动阈值
        max_retries: 最大重试次数
        show_reasoning: 是否显示推理过程
        min_dominant_prob: 主导方向的最小概率阈值，默认0.5

    Returns:
        Tuple[bool, Dict]: (是否成功, 结果或错误信息)
            成功时返回: {
                'prediction': {...},  # 预测JSON
                'reasoning': '...'     # 推理过程（如果show_reasoning=True）
            }
            失败时返回: {
                'error': '错误信息'
            }
    """
    logger.info(f"开始调用LLM进行方向预测 - 基金: {fund_code}, 日期: {date}")

    # 获取模型配置
    model_config = MODEL_CONFIG['deepseek']['deepseek-reasoner']

    # 初始化客户端
    client = OpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )

    # 构建prompt
    prompt = build_direction_prompt(
        fund_code=fund_code,
        date=date,
        four_experts_results=four_experts_results,
        price_context=price_context,
        volatility_threshold=volatility_threshold
    )

    logger.debug(f"Prompt长度: {len(prompt)}字符")

    # 最多尝试max_retries次
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
                temperature=0.6
            )

            # 提取推理过程和最终内容
            reasoning_content = ""
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning_content = response.choices[0].message.reasoning_content or ""

            final_content = response.choices[0].message.content

            # 显示推理过程
            if show_reasoning and reasoning_content:
                logger.info("=" * 80)
                logger.info("LLM推理过程：")
                logger.info("=" * 80)
                logger.info(reasoning_content)
                logger.info("=" * 80)

            logger.info("  LLM调用成功，开始验证概率和...")

            # 清理final_content，去除markdown代码块标记
            cleaned_content = clean_json_response(final_content)

            # 解析JSON
            try:
                prediction_json = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.error(f"  ✗ JSON解析失败: {e}")
                last_error = f"JSON解析失败: {e}"
                if attempt < max_retries:
                    logger.info(f"  准备进行第 {attempt + 1} 次尝试...")
                    continue
                else:
                    return False, {'error': last_error}

            # 验证概率和
            is_valid_sum, error_msg_sum = validate_probability_sum(prediction_json)

            if not is_valid_sum:
                logger.warning(f"  ✗ 概率和验证失败（第 {attempt} 次尝试）: {error_msg_sum}")
                last_error = f"概率和验证失败: {error_msg_sum}"

                if attempt < max_retries:
                    logger.info(f"  准备进行第 {attempt + 1} 次尝试...")
                    continue
                else:
                    logger.error(f"  已达最大重试次数（{max_retries}次），概率和验证仍未通过")
                    return False, {'error': last_error}

            # 验证主导概率
            is_valid_dominant, error_msg_dominant = validate_dominant_probability(prediction_json, min_dominant_prob)

            if not is_valid_dominant:
                logger.warning(f"  ✗ 主导概率验证失败（第 {attempt} 次尝试）: {error_msg_dominant}")
                last_error = f"主导概率验证失败: {error_msg_dominant}"

                if attempt < max_retries:
                    logger.info(f"  准备进行第 {attempt + 1} 次尝试...")
                    continue
                else:
                    logger.error(f"  已达最大重试次数（{max_retries}次），主导概率验证仍未通过")
                    return False, {'error': last_error}

            # 所有验证通过
            logger.info(f"  ✓ 概率和验证通过（第 {attempt} 次尝试）")
            logger.info(f"  ✓ 主导概率验证通过（阈值={min_dominant_prob}，第 {attempt} 次尝试）")
            result = {
                'prediction': prediction_json,
                'reasoning': reasoning_content,  # 始终返回，供数据库写入
                'llm_input': prompt  # 返回完整的prompt，供数据库写入
            }
            return True, result

        except Exception as e:
            logger.error(f"  第 {attempt} 次调用LLM失败: {e}")
            last_error = str(e)

            if attempt < max_retries:
                logger.info(f"  准备进行第 {attempt + 1} 次尝试...")
            else:
                logger.error(f"  已达最大重试次数（{max_retries}次）")
                return False, {'error': last_error}

    # 理论上不会到达这里
    return False, {'error': last_error}
