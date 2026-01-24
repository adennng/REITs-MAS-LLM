#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示词构建模块
根据state_t构建决策专家的完整提示词
"""

import os
import sys
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.core.signal_processor import get_signal_description
from agent_trading.utils.helpers import format_percentage, format_currency


def get_direction_description(pred_dir: str) -> str:
    """
    获取主导方向的中文描述

    Args:
        pred_dir: 主导方向 (up/down/side)

    Returns:
        str: 中文描述
    """
    direction_map = {
        'up': '上涨',
        'down': '下跌',
        'side': '盘整'
    }
    return direction_map.get(pred_dir, '未知')


def format_direction_judgment(direction_judgment: Dict[str, Any]) -> str:
    """
    格式化方向判断信息

    Args:
        direction_judgment: 方向判断字典

    Returns:
        str: 格式化后的字符串
    """
    if not direction_judgment:
        return "无方向判断数据"

    result = []

    # 处理短期判断（T+1, T+5）
    short_term = direction_judgment.get('short_term', {})
    for period in ['T+1', 'T+5']:
        if period in short_term:
            data = short_term[period]
            probs = data.get('direction_probs', {})
            result.append(
                f"**{period}**: 上涨{probs.get('up', 0):.2f} / 下跌{probs.get('down', 0):.2f} / "
                f"盘整{probs.get('side', 0):.2f}, "
                f"置信度{data.get('confidence', 0):.2f}, "
                f"理由: {data.get('rationale', '无')}"
            )

    # 处理T+20（如果有）
    if 'T+20' in direction_judgment:
        data = direction_judgment['T+20']
        probs = data.get('direction_probs', {})
        result.append(
            f"**T+20**: 上涨{probs.get('up', 0):.2f} / 下跌{probs.get('down', 0):.2f} / "
            f"盘整{probs.get('side', 0):.2f}, "
            f"置信度{data.get('confidence', 0):.2f}, "
            f"理由: {data.get('rationale', '无')}"
        )

    return '\n'.join(result)


def format_announcement_expert(expert_data: Dict[str, Any]) -> str:
    """
    格式化公告专家分析结果

    Args:
        expert_data: 公告专家输出的完整数据

    Returns:
        str: 格式化后的字符串
    """
    if not expert_data:
        return "无数据"

    # 检查是否是特殊情况（无数据或中性）
    if expert_data.get('no_data_or_neutral'):
        return f"**说明**: {expert_data.get('message', '无有效信息')}"

    # 正常情况：展示方向判断和综合分析
    result = []

    # 1. 方向判断
    if 'direction_judgment' in expert_data:
        result.append("**方向判断**:")
        result.append(format_direction_judgment(expert_data['direction_judgment']))

    # 2. 综合分析
    detailed = expert_data.get('detailed_analysis', {})
    if 'comprehensive_analysis' in detailed:
        result.append(f"\n**综合分析**: {detailed['comprehensive_analysis']}")

    return '\n'.join(result) if result else "无有效数据"


def format_market_expert(expert_data: Dict[str, Any]) -> str:
    """
    格式化市场专家分析结果

    Args:
        expert_data: 市场专家输出的完整数据

    Returns:
        str: 格式化后的字符串
    """
    if not expert_data:
        return "无数据"

    result = []

    # 1. 方向判断
    if 'direction_judgment' in expert_data:
        result.append("**方向判断**:")
        result.append(format_direction_judgment(expert_data['direction_judgment']))

    # 2. 综合分析
    detailed = expert_data.get('detailed_analysis', {})
    if 'comprehensive_analysis' in detailed:
        result.append(f"\n**综合分析**: {detailed['comprehensive_analysis']}")

    return '\n'.join(result) if result else "无有效数据"


def format_price_expert(expert_data: Dict[str, Any]) -> str:
    """
    格式化价格动量专家分析结果

    Args:
        expert_data: 价格动量专家输出的完整数据

    Returns:
        str: 格式化后的字符串
    """
    if not expert_data:
        return "无数据"

    result = []

    # 1. 方向判断
    if 'direction_judgment' in expert_data:
        result.append("**方向判断**:")
        result.append(format_direction_judgment(expert_data['direction_judgment']))

    # 2. 综合分析
    detailed = expert_data.get('detailed_analysis', {})
    if 'technical_summary' in detailed:
        result.append(f"\n**综合分析**: {detailed['technical_summary']}")

    return '\n'.join(result) if result else "无有效数据"


def format_event_expert(expert_data: Dict[str, Any]) -> str:
    """
    格式化事件专家分析结果

    Args:
        expert_data: 事件专家输出的完整数据

    Returns:
        str: 格式化后的字符串
    """
    if not expert_data:
        return "无数据"

    result = []

    # 1. 方向判断
    if 'direction_judgment' in expert_data:
        result.append("**方向判断**:")
        result.append(format_direction_judgment(expert_data['direction_judgment']))

    # 2. 综合分析
    detailed = expert_data.get('detailed_analysis', {})
    if 'comprehensive_analysis' in detailed:
        result.append(f"\n**综合分析**: {detailed['comprehensive_analysis']}")

    return '\n'.join(result) if result else "无有效数据"


def format_direction_prediction(prediction_data: Dict[str, Any]) -> str:
    """
    格式化预测专家的预测结果

    Args:
        prediction_data: 预测专家输出的完整数据

    Returns:
        str: 格式化后的字符串
    """
    if not prediction_data:
        return "无数据"

    # 提取predictions字段
    predictions = prediction_data.get('predictions', {})
    if not predictions:
        return "无预测数据"

    result = []

    # 格式化每个时间维度的预测
    for period in ['T+1', 'T+5', 'T+20']:
        if period in predictions:
            data = predictions[period]
            probs = data.get('direction_probs', {})
            result.append(
                f"**{period}**: 上涨{probs.get('up', 0):.2f} / 下跌{probs.get('down', 0):.2f} / "
                f"盘整{probs.get('side', 0):.2f}, "
                f"置信度{data.get('confidence', 0):.2f}, "
                f"理由: {data.get('rationale', '无')}"
            )

    return '\n'.join(result) if result else "无有效预测数据"


def build_trading_prompt(state: Dict[str, Any]) -> str:
    """
    构建决策专家的完整提示词

    Args:
        state: 完整的state_t

    Returns:
        str: 完整的提示词
    """
    meta = state['meta_info']
    four_experts = state['four_experts_output']
    direction = state['direction_output']
    price_ctx = state['price_context']
    account = state['account_state']
    risk = state['risk_config']
    phase = state['phase_info']

    # 构建提示词各部分
    prompt = f"""你是一个公募REITs单资产策略的"决策专家"。

每个交易日收盘后，你会收到以下信息，需要为下一交易日给出调仓决策。

---

## 一、元信息

- **分析日期**: {meta['analysis_date']}
- **基金代码**: {meta['fund_code']}
- **策略运行天数**: {account['days_since_start']}
- **是否建仓期**: {'是' if phase['is_building_phase'] else '否'}

---

## 二、当前账户状态

### 账户总体情况

- **当前仓位**: {format_percentage(account['position'])}
- **现金**: {format_currency(account['cash'])}
- **持有份额**: {account['shares']:.4f}
- **账户总净值**: {format_currency(account['nav'])}
- **账户总收益率**: {format_percentage(account['total_return'])}
- **历史最高净值**: {format_currency(account['peak_nav'])}
- **账户最大回撤**: {format_percentage(account['max_drawdown'])}
"""

    # 添加本轮持仓信息（如果有持仓）
    if account['position'] > 0.001:
        round_entry = account.get('round_entry_price')
        round_return = account.get('round_return')
        round_peak = account.get('round_peak_nav')
        round_dd = account.get('round_max_drawdown')
        round_days = account.get('round_holding_days', 0)

        entry_str = f"{round_entry:.4f}" if round_entry else '无'
        return_str = format_percentage(round_return) if round_return is not None else '无'
        peak_str = f"{round_peak:.2f}" if round_peak else '无'
        dd_str = format_percentage(round_dd) if round_dd is not None else '无'

        prompt += f"""

### 本轮持仓信息

- **本轮平均成本**: {entry_str}
- **本轮收益率**: {return_str}
- **本轮最高净值**: {peak_str}
- **本轮最大回撤**: {dd_str}
- **本轮持仓天数**: {round_days}天
"""

    prompt += f"""

---

## 三、四个专家的分析结果

以下是四个专家从各自专业角度的分析。注意它们的分工：

**公告专家**：偏短期，关注公司公告、分红、资产运营等；
**市场专家**：偏宏观/中长期，关注政策、利率、市场情绪等；
**价格动量专家**：偏短期技术面，关注价格趋势、成交量等；
**事件专家**：关注突发事件、重大新闻、资产基本面。

### 公告专家分析结果：

{format_announcement_expert(four_experts.get('announcement'))}

### 市场专家分析结果：

{format_market_expert(four_experts.get('market'))}

### 价格动量专家分析结果：

{format_price_expert(four_experts.get('price'))}

### 事件专家分析结果：

{format_event_expert(four_experts.get('event'))}

---

## 四、预测专家的综合判断

预测专家基于上述四个专家的`detailed_analysis`字段进行综合分析后给出的预测结果：

{format_direction_prediction(direction.get('direction_prediction'))}

### 信号分档

**重要**：以下信号档位已根据概率、置信度自动离散化，**决策时优先按档位理解**，而不是纠结于具体概率数值。

**信号档位共7档**：强多、中多、弱多、中性、弱空、中空、强空

**T+1信号**:
- 信号档位: {get_signal_description(direction['T1']['signal_level'])}
- 主导方向: {get_direction_description(direction['T1']['pred_dir'])}
- 概率分布: 上涨{direction['T1']['prob_up']:.2f} / 下跌{direction['T1']['prob_down']:.2f} / 盘整{direction['T1']['prob_side']:.2f}
- 置信度: {direction['T1']['confidence']:.2f}

**T+5信号**:
- 信号档位: {get_signal_description(direction['T5']['signal_level'])}
- 主导方向: {get_direction_description(direction['T5']['pred_dir'])}
- 概率分布: 上涨{direction['T5']['prob_up']:.2f} / 下跌{direction['T5']['prob_down']:.2f} / 盘整{direction['T5']['prob_side']:.2f}
- 置信度: {direction['T5']['confidence']:.2f}

**T+20信号**:
- 信号档位: {get_signal_description(direction['T20']['signal_level'])}
- 主导方向: {get_direction_description(direction['T20']['pred_dir'])}
- 概率分布: 上涨{direction['T20']['prob_up']:.2f} / 下跌{direction['T20']['prob_down']:.2f} / 盘整{direction['T20']['prob_side']:.2f}
- 置信度: {direction['T20']['confidence']:.2f}

---

## 五、价格与盘整上下文

"""

    # 添加最近N日详情（动态显示实际天数）
    recent_days = price_ctx.get('recent_5_days', [])
    num_days = len(recent_days)

    if num_days > 0:
        prompt += f"### 最近{num_days}个交易日情况\n\n"
        for i, day in enumerate(recent_days, 1):
            change_str = format_percentage(day['change_pct']) if day['change_pct'] is not None else '无数据'
            threshold_str = format_percentage(day['volatility_threshold']) if day['volatility_threshold'] is not None else '无数据'
            ratio_str = f"{day['ratio_t']:.2f}" if day['ratio_t'] is not None else '无数据'
            regime_str = day['daily_regime'] if day['daily_regime'] else '无数据'
            delta_str = str(day['delta_steps']) if day['delta_steps'] is not None else '无数据'

            prompt += f"""**第{i}日 ({day['date']})**:
- 收盘价: {day['close']}
- 涨跌幅: {change_str}
- 波动阈值: {threshold_str}
- ratio_t: {ratio_str}
- 盘整状态: {regime_str}
- 当日决策: {delta_str}

"""
        # 在所有交易日数据展示完后，统一说明整盘判断规则
        prompt += "\n**整盘判断**：日波动幅度小于波动阈值即视为整盘，代表价格走势不显著。公募reits波动率较低，历史约33%交易日处于整盘状态\n\n"
    else:
        prompt += "### 暂无历史交易日情况\n\n（首次交易日，无历史决策记录可供参考）\n\n"

    # 提取象限信息（仅象限定位）
    market_expert = four_experts.get('market', {})
    detailed_analysis = market_expert.get('detailed_analysis', {})
    quadrant_info = detailed_analysis.get('quadrant_info', {})
    current_quadrant = quadrant_info.get('current_quadrant', '未知')

    prompt += f"""
---

## 六、风险约束与策略参数

### 可用调仓指令（必须从以下7个值中选择）

你必须输出以下**7个调仓指令**之一：

- **-10**：清仓指令，立即清仓至0%
- **-2**：强减仓，减少{format_percentage(2 * risk['step'])}仓位
- **-1**：减仓，减少{format_percentage(risk['step'])}仓位
- **0**：保持不变
- **1**：加仓，增加{format_percentage(risk['step'])}仓位
- **2**：强加仓，增加{format_percentage(2 * risk['step'])}仓位
- **10**：满仓指令，立即满仓至{format_percentage(risk['position_max'])}


**重要**：只能输出上述7个值之一，不能输出其他数值

---

### 硬约束（必须严格遵守）

- **最大仓位**: {format_percentage(risk['position_max'])} - 目标仓位不会超过此值
- **最小调仓步长**: {format_percentage(risk['step'])} - 每步对应{format_percentage(risk['step'])}仓位变动
- **建仓期目标仓位**: {format_percentage(risk['building_target_position'])} - 建仓期内仓位上限
- **建仓期最长天数**: {risk['D_build_max']}天
- **硬止损阈值**: {format_percentage(risk['hard_dd_limit'])} - 本轮最大回撤触及此值时，必须使用 `delta_steps = -10` 清仓

**目标仓位将由系统计算**：如果你的`delta_steps`导致仓位突破硬性要求的上下限（仓位1或0），系统会自动将仓位修正为1或0，所以**无需担心仓位超出上下限**。

### 行为偏好（非强制，建议参考，但可根据具体情况调整）

- **止盈参考水平**: {format_percentage(risk['take_profit_level'])} - 本轮收益率达到此值时，可考虑部分止盈
- **止盈前最小持仓天数**: {risk['min_holding_days_before_tp']}天 - 避免过早止盈

---

## 七、REITs特性与策略目标

### REITs市场特性

公募REITs波动率远低于股票：
- 日涨跌幅通常±1%以内，超过±2%即为显著变动，日最高涨跌停±10%
- **低波动=低风险**：下跌幅度小，风控容易；**低波动≠低收益**，可用更高仓位捕捉趋势
- **整盘是常态**：日波动幅度小于波动阈值即视为整盘，历史约33%交易日处于整盘状态（neutral信号），整盘不等于不确定，是价格走势不显著的正常状态

### 策略目标

**首要目标：收益最大化**

**仓位原则**：
- 上涨区间 → 满仓
- 下跌区间 → 空仓或低仓位

**操作原则**：以核心操作表为操作指南，强多信号时果断加仓至满仓，强空信号时逐步减仓至低仓位或空仓，整体弱信号或中性信号时尽量保持不动，以减少调仓频率。避免长期空仓（除非是下跌区间）

**风控底线**：严格遵守硬止损阈值（本轮回撤达{format_percentage(risk['hard_dd_limit'])}时强制清仓），其他约束可根据信号强度灵活调整

---

## 八、决策框架

### 三个时间维度的协同机制

| 维度 | 角色定位 | 主要作用 |
|------|----------|----------|
| **T+5** | **主决策依据** | 决定加减仓方向和幅度 |
| **T+20** | 仓位上限与风格 | 控制最大仓位和持有耐心 |
| **T+1** | 择时优化 | **双向作用**：<br>① 防守：避免短期不利时点大幅调仓<br>② 进攻：强多时可作为加仓积极信号 |

---

### 核心操作表（基于T+5信号）

| T+5档位 | 标准操作 | T+20影响 | T+1影响 |
|---------|----------|----------|---------|
| **强多** | **2或10**（强力加仓） | T+20强空时降为1<br>T+20强多时用10| T+1强空时可降为1<br>T+1强多时用10 |
| **中多** | **1或2** | T+20强多/中多/弱多时用2<br>T+20强空/中空/弱空时用1 | T+1强多/中多/弱多时用2<br>T+1强空/中空/弱空时用1 |
| **弱多** | **1或0**（试探加仓） | T+20强空时用0<br>T+20强多时用1 | T+1强空时用0<br>T+1强多时用1 |
| **中性** | **默认0** | T+20强多/中多/弱多+0仓→**可升为1** | T+1强多/中多/弱多+0仓→**可升为1** |
| **弱空** | **默认0** | T+20强空时降为-1 | T+1强空时降为-1 |
| **中空** | **默认0** | T+20强空时降为-1 | T+1强空时降为-1 |
| **强空** | **-1或-2**（减仓）| T+20强空时用-2 | T+1强空时用-2 |

**说明**：

1. 表格结构
- 标准操作：基于T+5信号给出的基础建议
- T+20影响：中长期趋势对操作幅度的调整建议
- T+1影响：超短期信号对操作幅度的调整建议

2. 使用逻辑
- 决策时优先参照标准操作
- 若当前行情满足 T+20影响 或 T+1影响 中任一条件，则按该条件调整操作幅度
- 若均不满足，则执行标准操作
- 若仍有多个可选操作，可结合历史价格走势与四个专家的【综合分析】中的关键信息进一步确定

---

### 建仓期特殊规则（建仓期适用）
**建仓期目标**：快速建立底仓

| T+5档位 | 0仓操作 | 已有小仓操作 |
|---------|---------|--------------|
| **强多** | **2** | **1或2** |
| **中多** | **1** | **1或0** |
| **弱多** | **0或1**（试探） | **0** |
| **中性** | T+20看多时**1**，否则**0** | **0** |
| **弱空及以下** | **0** | **-1或0** |

**建仓期说明**：
- 允许始终保持0仓（等待明确信号）
- 但强多+T+20看多时可积极建仓
- 建仓期仓位不超过{format_percentage(risk['building_target_position'])}

---

## 九、决策流程

请严格按以下步骤进行决策：

---

### 第一步：关键信息回顾

1. 查看元信息中的分析日期、基金代码、是否属于建仓期，以及当前账户状态中的当前仓位，和第七节的策略目标

---

### 第二步：查看信号档位和专家分析结果

**查看内容**：
1. **信号档位**（第四节）：T+5、T+20、T+1的信号档位
2. **预测专家分析**（第四节）：预测专家的对于各维度判断的理由
3. **四个专家分析**（第三节）：各专家的【综合分析】中的关键信息

---

### 第三步：建仓期操作

**适用范围**：本步骤仅适用于建仓期，非建仓期的跳过本步骤

1. 查看第八节"建仓期特殊规则"初步确定delta_steps的范围
2. 查看价格与盘整上下文，分析近期价格走势、整盘突破情况**辅助判断**：
- 若最近几日连续上涨且突破阈值 + T+5 强多 → 增强加仓信心
- 若最近几日连续下跌且突破阈值 + T+5 强空 → 增强减仓信心
3. 最终确定结果。符合建仓期的，执行完上述步骤直接输出结果跳过后续步骤

---

### 第四步：查看四象限仓位或上限要求

**当前市场所处象限**：{current_quadrant}

- 如果当前象限为象限II，则为最危险，双重挤压，则仓位上限为70%
- 如果当前象限属于I/III/IV/过渡区，则无限制
- 根据当前仓位和上述仓位上限要求，初步确定本次可调仓位的最大范围

---

### 第五步：根据核心操作表确定标准操作

1. 在第八节"核心操作表"中找到T+5对应的行
2. 查看"标准操作"列，确定基础操作
3. 分别查看"T+20影响"列和"T+1影响"列，调整操作（满足条件之一即可调整）
4. 若不满足调整操作，则执行标准操作
5. 初步确定delta_steps的范围。若有多个可选操作，可结合后续步骤的历史价格走势与四个专家的【综合分析】中的关键信息进一步确定

---

### 第六步：查看近期价格和专家分析结果进行辅助判断

1. 查看价格与盘整上下文内容（第五节）：
- **最近N日价格走势**：收盘价变动趋势
- **波动阈值与盘整状态**：最近交易日的波动阈值、最近交易日的盘整状态、ratio_t值
- **历史决策效果**：最近几日的delta_steps、观察之前决策的效果

**辅助判断**：
- 若最近几日连续上涨且突破阈值  → 增强加仓信心
- 若最近几日连续下跌且突破阈值  → 增强减仓信心

2. 如果上述信息仍不能形成明确的判断可以进一步结合四个专家【综合分析】中的关键信息确定

3. **判断原则**：请先严格执行核心操作表，以核心操作表为准。只有存在不确定时，结合历史价格走势与四个专家的【综合分析】进行判断，判断原则是：强多时果断加仓至满仓，强空时逐步减仓至低仓位或空仓，整体弱信号或中性信号时尽量保持不动，以减少调仓频率

---

### 第七步：检查账户状态与风控

**检查内容**：

1. **当前仓位状态**：
   - 当前仓位：{format_percentage(account['position'])}

2. **本轮盈亏情况**：
   - 本轮收益率：{format_percentage(account.get('round_return')) if account.get('round_return') is not None else '无'}
   - 本轮最大回撤：{format_percentage(account.get('round_max_drawdown')) if account.get('round_max_drawdown') is not None else '无'}

3. **风控触发检查**：
   - **止盈条件**：本轮收益率≥{format_percentage(risk['take_profit_level'])} 且 T+5转中空/强空
     → 如信号明确可以考虑减仓（-1）

   - **硬止损条件**：本轮回撤超过{format_percentage(risk['hard_dd_limit'])}
     → 系统强制清仓（-10）

4. **硬约束检查**（建仓期适用）：
   - 建仓期仓位是否≤{format_percentage(risk['building_target_position'])}

**调整逻辑**：
- 若触发止盈/止损条件，优先执行风控操作

### 第八步：最终确定结果

**确定最终输出结果**：

综合前面所有步骤，按以下原则确定最终delta_steps：

1. **风控优先**：若触发硬止损 → 使用 `-10`（清仓）；若触发软止盈 → 可以考虑使用 `-1`（减仓）
2. **信号主导**：执行第五步根据核心操作表确定标准操作 → （如果）存在不确定/多个可选操作时，以第六步判断为辅，确定具体操作
3. **仓位状态检查**：执行任何操作前，检查当前仓位是否允许执行目标操作，例如
- 如果当前仓位已经处于下限（0仓），则忽略任何减仓/做空信号，应选择 0（维持空仓）
- 如果当前仓位已经处于上限（满仓），则忽略任何加仓/做多信号，应选择 0（维持满仓）
4. **确定结果**：最终确定delta_steps的值（必须从第六节的7个值 (-10, -2, -1, 0, 1, 2, 10) 中选择）

---

### 第九步：决策完成检查

在输出结果前，请检查：

- ✅ 建仓期是否遵从了建仓期操作要求？
- ✅ 非建仓期是否遵从了四象限仓位或上限要求？并检查了当前仓位情况？
- ✅ 非建仓期是否应用了核心操作表的标准操作？
- ✅ 非建仓期是否参考了价格实际走势和专家的综合分析？
- ✅ 非建仓期是否检查了账户状态和风控条件？
- ✅ 是否符合策略目标的整体要求？
- ✅ delta_steps是否是第六节列出的7个值之一（-10, -2, -1, 0, 1, 2, 10）？

**记住策略目标**：利用REITs低波动特性，在要求回撤范围内，收益最大化。严格遵从核心操作表的要求，存在不确定时考虑：强多信号时果断加仓至满仓，强空信号时逐步减仓至低仓位或空仓，整体弱信号或中性信号时尽量保持不动，以减少调仓频率。避免长期空仓（除非是下跌区间）

---

## 十、输出格式

请严格输出以下JSON结构：

```json
{{
  "delta_steps": 0,
  "rationale_core": "一句话说明本次决策核心原因，50字以内"
}}
```

**注意**：
- delta_steps必须是第六节列出的7个值之一（-10, -2, -1, 0, 1, 2, 10）
- rationale_core需引用具体的信号档位、价格走势、仓位要求或风控条件等信息支持你的决策
- 只输出JSON，不要有其他内容

请开始分析并输出决策。
"""

    return prompt
