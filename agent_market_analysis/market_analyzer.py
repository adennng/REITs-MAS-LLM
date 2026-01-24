"""
市场分析Agent主模块
调用LLM进行市场分析
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Tuple
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from config.model_config import MODEL_CONFIG
from agent_market_analysis.data_fetcher import DataFetcher
from agent_market_analysis.indicators_calculator import IndicatorsCalculator
from agent_market_analysis.utils import (
    interpret_indicators,
    generate_market_summary,
    determine_quadrant,
    format_output_for_decision_agent
)
from agent_price.daily_threshold_calculator import calculate_daily_volatility_threshold

logger = logging.getLogger(__name__)


def format_recent_5d_changes(recent_5d_changes: list, volatility_threshold: float) -> str:
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
        # 提供原始内容的预览（前200个字符）
        preview = analysis_json_str[:200] if len(analysis_json_str) > 200 else analysis_json_str
        return False, f"JSON解析失败: {str(e)} | 原始内容预览: {preview}"
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
        preview = analysis_json_str[:200] if len(analysis_json_str) > 200 else analysis_json_str
        return False, f"JSON解析失败: {str(e)} | 原始内容预览: {preview}"
    except Exception as e:
        return False, f"主导概率验证过程发生错误: {str(e)}"


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class MarketAnalyzer:
    """市场分析Agent"""

    def __init__(self):
        """初始化"""
        self.data_fetcher = DataFetcher()
        self.calculator = IndicatorsCalculator()
        self.llm_client = None

    def _init_llm_client(self):
        """初始化LLM客户端"""
        if self.llm_client is None:
            # 使用DeepSeek配置
            deepseek_config = MODEL_CONFIG['deepseek']['deepseek-reasoner']
            self.llm_client = OpenAI(
                api_key=deepseek_config['api_key'],
                base_url=deepseek_config['base_url']
            )
            logger.info("LLM客户端初始化成功")

    # 注释：_build_system_prompt 方法已不再使用
    # 原系统提示词内容已合并到 _build_user_prompt 方法中
    # 保留此方法定义以防需要回滚，但不再被调用
    # def _build_system_prompt(self):
    #     """构建系统提示词"""
    #     return """你是一位资深的公募REITs市场分析专家..."""

    def _build_user_prompt(self, market_data, fund_code, fund_name, sector,
                           volatility_threshold, recent_5d_changes, recent_20d_data):
        """构建用户提示词（已包含原系统提示词内容）"""
        prompt = f"""你是一位资深的公募REITs市场分析专家，拥有超过10年的固收+投资经验。你的任务是基于提供的公募reits市场及相关资产数据，从**市场整体角度**分析当前reits整体情况，进行深度分析并给出明确的投资建议。你的建议将与其他方面分析专家（如新闻、特定产品的价格走势、事件分析等）一同给到投资者，由投资者进行综合分析并进行决策。

## 【核心投资框架】

### 1. REITs市场本质认知
- **市场定位**: REITs = 70%债券属性 + 30%股票属性
- **最强影响因素**: 10年期国债利率
- **次强影响因素**: 高股息/红利板块的资金跷跷板效应

### 2. 四象限决策模型（核心！）

基于**利率趋势**（横轴）和**股市状态**（纵轴）的四象限模型：

**标准四象限**：

**象限I（利率下行 + 股市上涨）**:
- 状态：谨慎乐观
- 风险：需警惕股市持续强势导致资金分流

**象限II（利率上行 + 股市上涨）**:
- 状态：最危险（双重挤压）
- 风险：极高，REITs可能暴跌

**象限III（利率下行 + 股市下跌/震荡）**:
- 状态：最佳配置期
- 机会：双重利好，最确定的上涨窗口

**象限IV（利率上行 + 股市下跌）**:
- 状态：谨慎防御
- 风险：利率压制明显

**过渡区（利率横盘或边界情况）**:
- 利率横盘 + 股市上涨：中性偏负（股市分流资金）
- 利率横盘 + 股市震荡：中性
- 利率横盘 + 股市下跌：中性偏正（避险需求）
- 利率上行 + 股市震荡：偏不利（利率压制）

### 3. 关键判断标准

**利率趋势判断**：
- 明确下行：近20日利率变化 < -20bp
- 缓慢下行：近20日利率变化 -20 到 -5bp
- 横盘震荡：近20日利率变化 -5 到 +5bp
- 缓慢上行：近20日利率变化 +5 到 +20bp
- 明确上行：近20日利率变化 > +20bp

**股市状态判断**：
- 牛市：近20日涨幅>5% 且 RSI>60
- 震荡偏强：近20日涨幅0-5% 且 RSI 50-60
- 震荡：近20日涨幅-2到2% 且 RSI 40-60
- 震荡偏弱：近20日涨幅-5到0% 且 RSI 40-50
- 熊市：近20日涨幅<-5% 且 RSI<40

**估值水平判断**（基于分位数）：
- 极低：历史分位数<20
- 偏低：历史分位数20-40
- 中性：历史分位数40-60
- 偏高：历史分位数60-80
- 极高：历史分位数>80

**市场情绪判断**（基于换手率分位数）：
- 极低（冰点）：换手率分位数<10
- 偏低：换手率分位数10-30
- 正常：换手率分位数30-70
- 偏高（活跃）：换手率分位数70-90
- 极高（过热）：换手率分位数>90

### 4. 投资纪律

**必须遵守的原则**：
- ✅ 关注利率走势
- ✅ 关注红利指数走势，警惕资金分流效应
- ✅ 不追高：避免在历史高位（分位数>80）追涨
- ✅ 右侧交易：等待趋势确认后介入
- ✅ 量价配合：放量上涨健康，缩量上涨需警惕
- ❌ 避免左侧抄底
- ❌ 避免在双重挤压（象限II）时重仓
- ❌ 避免忽视换手率萎缩信号

## 【分析要求】

### 分析维度（按重要性排序）
1. **四象限定位**（权重30%）：当前处于哪个象限？该象限的典型策略是什么？
2. **利率环境分析**（权重20%）：利率趋势如何？对REITs的影响？
3. **股市环境分析**（权重20%）：股市情绪如何？红利板块是否对REITs形成压制？跷跷板效应是否明显？
4. **所属REITs板块技术面分析**（权重10%）：REITs自身的趋势、动量、支撑阻力如何？
4. **REITs市场技术面分析**（权重10%）：REITs自身的趋势、动量、支撑阻力如何？
5. **市场情绪分析**（权重10%）：换手率、量价关系反映什么信号？

### 分析深度
- 不仅要描述"是什么"，更要分析"为什么"
- 关注指标之间的相互印证或矛盾
- 识别市场的主要矛盾和次要矛盾

## 【特别注意】
- 你收到的数据已经过预处理和分类，包含三层信息：摘要层、解释层、原始指标层
- 如果认为摘要层或解释层不合理，可以基于原始指标层自行判断分析
- 所有判断必须基于数据，不能主观臆断
- 对于矛盾的信号，要权衡主要矛盾

---

# 市场分析任务

## 基本信息
- 分析日期：{market_data['meta']['analysis_date']}
- 目标基金代码：{fund_code}
- 目标基金名称：{fund_name or '未知'}
- 所属板块：{sector or '未知'}

## 市场数据

{json.dumps(market_data, ensure_ascii=False, indent=2, cls=NumpyEncoder)}

## 整盘判断专项说明

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
- 整盘的判断应基于技术指标的综合分析

## 价格形态

### 最近20个交易日价格和涨跌幅
以下是目标基金（{fund_code}）最近20个交易日的价格走势：

"""
        # 添加最近20日数据表格
        if recent_20d_data:
            for idx, item in enumerate(recent_20d_data, 1):
                prompt += f"- 第{idx}天 ({item['date']}): 收盘价 {item['price']:.3f}元, 涨跌幅 {item['change_pct']:+.2f}%\n"
        else:
            prompt += "- 数据不足，无法显示最近20日价格数据\n"

        prompt += f"""
### 价格形态说明
- 以上数据展示了目标基金最近20个交易日的价格走势
- 可以帮助你判断：
  - 价格趋势（上涨/下跌/横盘）
  - 波动幅度
  - 连续涨跌情况
  - 是否处于整盘状态

## 分析任务

你必须**严格按照以下JSON格式**输出分析结果。输出内容必须是**纯JSON格式**。

### 输出格式示例：

{{
  "expert_type": "market",
  "analysis_date": "{market_data['meta']['analysis_date']}",
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
    "quadrant_info": {{
      "current_quadrant": "I/II/III/IV/过渡区。用一句话介绍下该区域的普遍特点",
      "interest_trend": "明确下行/缓慢下行/横盘/缓慢上行/明确上行",
      "stock_market_status": "牛市/震荡偏强/震荡/震荡偏弱/熊市",
      "quadrant_strategy": "最佳配置期/谨慎防御/最危险/谨慎乐观等"
    }},
    "interest_environment": {{
      "analysis_target": "10年期国债利率",
      "current_level": 2.15,
      "percentile": 25,
      "level_assessment": "极低/偏低/中性/偏高/极高",
      "interest_change_20d": -30,
      "trend_signal": "利好/中性/利空"
      "general_impact_mechanism": "利率下行时,REITs的分派率相对吸引力提升,资金流入推动价格上涨;利率上行时,固定收益类资产吸引力增强,REITs面临资金流出压力,价格承压。REITs与10年国债利率相关性非常高。"
    }},
    "stock_environment": {{
      "analysis_target": "上证指数、红利指数",
      "shanghai_index_status": "强势/震荡/弱势",
      "dividend_index_status": "强势/震荡/弱势",
      "seesaw_effect": "明显/一般/不明显",
      "capital_flow_signal": "流入REITs/中性/流出REITs"
      "general_impact_mechanism": "股市(尤其高股息、红利板块)与REITs存在资金跷跷板效应。股市强势时资金分流至股票,REITs性价比下降;股市下跌或震荡时,REITs作为稳健替代品种吸引避险资金流入。最危险情况是股市大涨叠加利率上行的双重挤压。"
    }},
    "reits_technical": {{
      "analysis_target": "中证REITs全收益指数",
      "price_percentile": 45,
      "valuation_level": "极低/偏低/中性/偏高/极高",
      "trend_signal": "上涨/横盘/下跌",
      "momentum_signal": "强/中/弱"
    }},
    "market_sentiment": {{
      "analysis_target": "REITs全市场的成交量和换手率",
      "turnover_percentile": 35,
      "sentiment_level": "冰点/偏低/正常/活跃/过热",
      "volume_trend": "放量/缩量/平稳",
      "price_volume_relation": "量价配合/量价背离"
    }},
    "sector_performance": {{
      "analysis_target": "目标基金所属的REITs板块指数",
      "sector_name": "{sector or '未知'}",
      "relative_strength": "强于大盘/同步/弱于大盘",
      "sector_rank": "1-3/4-6/7-9/10-12",
      "suitability": "适合/一般/不适合"
    }},
    "scores": {{
      "interest_environment": 8,
      "stock_environment": 6,
      "reits_technical": 7,
      "market_sentiment": 5,
      "sector_analysis": 7
    }},
    "comprehensive_analysis": "综合上述各维度的分析，当前市场状态、投资逻辑和建议（300字以内，不使用换行符）",
    "risk_alerts": ["风险点1", "风险点2", "风险点3"]
  }}
}}

---

## 字段填写说明——必须遵守！！！

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
反映你对该判断的整体信心，市场信号的清晰程度（0.00-1.00），按信号强度限定：

**强信号**（四象限定位明确，多项指标共振）：
- 置信度：≥ 0.70
- 情况：处于象限III（双重利好）或象限II（双重挤压），利率和股市信号明确，REITs技术面配合等

**中等信号**（主要指标明确但有矛盾）：
- 置信度：0.60 - 0.65
- 情况：四象限定位清晰但存在次要矛盾，如利率利好但股市分流

**弱信号**（信号模糊）：
- 置信度：0.50 - 0.55
- 情况：处于四象限过渡区，利空利好相互抵消，市场方向不明

**极弱信号**（矛盾严重）：
- 置信度：≤ 0.45
- 情况：各项指标相互矛盾，市场处于关键转折期

**重要**：
- 在公募REITs低波动背景下，市场指标的微小变化也具有指导意义

#### rationale（理由）
用1句话（50字以内）说明核心理由

---

### 2. detailed_analysis（详细分析）

#### quadrant_info（四象限信息）
- **current_quadrant**: 当前所处象限（I/II/III/IV/过渡区）+用一句话介绍下该区域的普遍特点
- **interest_trend**: 利率趋势（明确下行/缓慢下行/横盘/缓慢上行/明确上行）
- **stock_market_status**: 股市状态（牛市/震荡偏强/震荡/震荡偏弱/熊市）
- **quadrant_strategy**: 该象限的典型策略（如"最佳配置期"、"谨慎防御"等）

#### interest_environment（利率环境）
- **analysis_target**: 直接原句抄下来
- **current_level**: 当前10年期国债利率（如2.15）
- **percentile**: 历史分位数（0-100）
- **level_assessment**: 水平评估（极低/偏低/中性/偏高/极高）
- **interest_change_20d**: 近20日利率变化（单位：bp，如-30表示下降30bp）
- **trend_signal**: 对REITs的信号（利好/中性/利空）
- **general_impact_mechanism**: 直接原句抄下来

#### stock_environment（股市环境）
- **analysis_target**: 直接原句抄下来
- **shanghai_index_status**: 上证指数状态（强势/震荡/弱势）
- **dividend_index_status**: 红利指数状态（强势/震荡/弱势）
- **seesaw_effect**: 跷跷板效应程度（明显/一般/不明显）
- **capital_flow_signal**: 资金流向信号（流入REITs/中性/流出REITs）
- **general_impact_mechanism**: 直接原句抄下来

#### reits_technical（REITs技术面）
- **analysis_target**: 直接原句抄下来
- **price_percentile**: REITs指数价格历史分位数（0-100）
- **valuation_level**: 估值水平（极低/偏低/中性/偏高/极高）
- **trend_signal**: 趋势信号（上涨/横盘/下跌）
- **momentum_signal**: 动量信号（强/中/弱）

#### market_sentiment（市场情绪）
- **analysis_target**: 直接原句抄下来
- **turnover_percentile**: 换手率历史分位数（0-100）
- **sentiment_level**: 情绪等级（冰点/偏低/正常/活跃/过热）
- **volume_trend**: 成交量趋势（放量/缩量/平稳）
- **price_volume_relation**: 量价关系（量价配合/量价背离）

#### sector_performance（板块表现）
- **analysis_target**: 直接原句抄下来
- **sector_name**: 板块名称（如"产业园"、"高速公路"等）
- **relative_strength**: 相对大盘表现（强于大盘/同步/弱于大盘）
- **sector_rank**: 在所有板块中的排名区间（1-3/4-6/7-9/10-12）
- **suitability**: 当前环境对该板块的适合度（适合/一般/不适合）

#### scores（评分）
对各维度进行1-10分的量化评分：
- **interest_environment**: 利率环境评分
- **stock_environment**: 股市环境评分
- **reits_technical**: REITs技术面评分
- **market_sentiment**: 市场情绪评分
- **sector_analysis**: 板块分析评分

#### comprehensive_analysis（综合分析）
综合上述各维度，分析当前市场状态、主要影响因素、主要矛盾、次要矛盾等等，给出投资逻辑和建议（300字以内）

**重要**：
- 使用连贯的段落形式，**不要使用换行符**
- 可以使用逗号、句号等标点符号组织内容
- 基于四象限定位和各项指标综合判断
- 提供必要的具体数据支撑
- 权衡主要矛盾和次要矛盾

#### risk_alerts（风险提示）
数组格式，列出1-3个当前最需要关注的风险点，每个风险点用一句话表述

---

## 重要提醒

1. **必须输出有效的JSON格式**，确保所有括号、引号、逗号正确
2. **概率之和必须=1.0**，系统会自动检查
3. **象限判断与概率一致**：quadrant_info应与概率倾向一致
4. **评分与置信度呼应**：如果多项评分都很低，置信度不应过高
5. **短期vs中长期**：短期可能受技术面影响偏离中长期方向
6. **控制字数**：按照字数限制编写
7. **不使用换行符**：comprehensive_analysis必须是连贯的段落
8. **数据支撑**：在阐述观点时，使用具体数据支撑
9. **基于四象限模型**：这是最核心的分析框架，所有判断应与之一致
10. **避免过度保守，敢于判断**：根据各维度信号强度给出相应的概率和置信度，不要因为害怕错误而系统性降低数值。请记住：
   **市场特性要求更果断的判断**：
   - 公募REITs波动率远低于股票
   - 整盘（side）是公募REITs常见且正常的状态，判断为整盘时也应给予合理的主导概率和置信度

   **投资者的策略依赖你的预测质量**：
   - 策略目标：在回撤可控的前提下最大化收益
   - 你的角色：提供清晰、有置信度的方向预测，作为交易决策的关键输入
   - "模糊安全"倾向（总是选择中间值或较低值），会显著降低策略的收益潜力
   - 因此在有明确信号（无论上涨、下跌还是整盘）时请给予较高的概率和置信度

   **注意**：这并不意味着盲目乐观，而是要求你基于充分的证据做出匹配的判断强度
11. **检查概率与置信度**：返回前检查 `direction_probs` 和 `confidence` 是否符合字段填写说明的要求，如果发现不符合要求，请根据分析重新打分

请开始你的分析，输出JSON格式结果。
"""
        return prompt

    def analyze(self, fund_code, analysis_date, fund_name=None, min_dominant_prob=0.5):
        """
        执行市场分析

        Args:
            fund_code: 基金代码
            analysis_date: 分析日期（字符串'YYYY-MM-DD'或date对象）
            fund_name: 基金名称（可选）
            min_dominant_prob: 主导方向的最小概率阈值，默认0.5

        Returns:
            dict: {
                'status': 'success' | 'error',
                'message': 错误信息（仅错误时）,
                'analysis_result': LLM返回的分析内容,
                'reasoning_process': LLM的推理过程,
                'raw_indicators': 原始指标数据,
                'metadata': 元数据
            }
        """
        logger.info(f"开始分析：基金{fund_code}，日期{analysis_date}")

        try:
            # 1. 获取数据
            logger.info("Step 1: 获取数据")
            raw_data = self.data_fetcher.fetch_all_data(fund_code, analysis_date)

            # 2. 计算指标
            logger.info("Step 2: 计算指标")
            raw_indicators = self.calculator.calculate_all_indicators(raw_data)

            # 3. 解释指标
            logger.info("Step 3: 解释指标")
            interpreted_metrics = interpret_indicators(raw_indicators)

            # 4. 生成市场摘要
            logger.info("Step 4: 生成市场摘要")
            market_summary = generate_market_summary(raw_indicators, interpreted_metrics)

            # 5. 确定四象限
            logger.info("Step 5: 确定四象限")
            quadrant_info = determine_quadrant(raw_indicators)
            market_summary['quadrant'] = quadrant_info

            # 5.5. 计算动态阈值和最近5日涨跌幅（用于整盘判断）
            logger.info("Step 5.5: 计算动态阈值和最近5日涨跌幅")
            fund_price_df = raw_data['fund_price']

            # 计算动态阈值
            if len(fund_price_df) >= 2:
                price_list = fund_price_df['close'].tolist()
                volatility_threshold = calculate_daily_volatility_threshold(price_list)
            else:
                volatility_threshold = None
                logger.warning("基金价格数据不足，无法计算动态阈值")

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
            else:
                logger.warning("基金价格数据不足，无法计算最近5日涨跌幅")

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
            else:
                logger.warning("基金价格数据不足，无法计算最近20日数据")

            # 6. 组装LLM输入
            logger.info("Step 6: 组装LLM输入")
            market_data = {
                'meta': {
                    'analysis_date': str(analysis_date),
                    'fund_code': fund_code,
                    'fund_name': fund_name or '',
                    'sector': raw_data['sector'] or '未知',
                    'analysis_version': 'v1.0'
                },
                'market_summary': market_summary,
                'interpreted_metrics': interpreted_metrics,
                'raw_indicators': raw_indicators
            }

            # 7. 调用LLM分析
            logger.info("Step 7: 调用LLM分析（最多尝试3次）")
            self._init_llm_client()

            # 构建用户提示词（已包含原系统提示词内容）
            user_prompt = self._build_user_prompt(
                market_data,
                fund_code,
                fund_name,
                raw_data['sector'],
                volatility_threshold,
                recent_5d_changes,
                recent_20d_data
            )

            # 记录传递给LLM的提示词
            logger.info("=" * 80)
            logger.info("传递给LLM的User Prompt（已包含原系统提示词内容）:")
            logger.info("=" * 80)
            logger.info(user_prompt)
            logger.info("=" * 80)
            logger.info("")

            # 最多尝试3次
            max_attempts = 3
            last_error = None
            reasoning_content = ""
            final_content = ""

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(f"第 {attempt} 次调用DeepSeek推理模型...")

                    # 调用DeepSeek推理模型（只使用user role，与其他专家保持一致）
                    response = self.llm_client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.6,  # 从0.7降低到0.6，与其他专家保持一致
                    )

                    # 提取推理过程和最终内容
                    if hasattr(response.choices[0].message, 'reasoning_content'):
                        reasoning_content = response.choices[0].message.reasoning_content or ""
                        if reasoning_content:
                            logger.info("=" * 80)
                            logger.info("LLM推理过程：")
                            logger.info("=" * 80)
                            logger.info(reasoning_content)
                            logger.info("=" * 80)
                            # 同时打印到终端
                            print("\n" + "=" * 80)
                            print("LLM推理过程：")
                            print("=" * 80)
                            print(reasoning_content)
                            print("=" * 80 + "\n")

                    final_content = response.choices[0].message.content

                    # 记录LLM返回的原始内容（用于调试）
                    logger.info("=" * 80)
                    logger.info("LLM返回的原始内容（final_content）：")
                    logger.info("=" * 80)
                    logger.info(final_content)
                    logger.info("=" * 80)
                    logger.info("")

                    logger.info("LLM调用成功，开始验证概率和...")

                    # 验证概率和
                    is_valid_sum, error_msg_sum = validate_probability_sum(final_content)

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
                    is_valid_dominant, error_msg_dominant = validate_dominant_probability(final_content, min_dominant_prob)

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
                    break  # 验证通过，跳出循环

                except ValueError:
                    # 重新抛出概率验证失败的异常
                    raise
                except Exception as e:
                    logger.error(f"第 {attempt} 次调用LLM失败: {e}")
                    last_error = str(e)

                    if attempt < max_attempts:
                        logger.info(f"准备进行第 {attempt + 1} 次尝试...")
                    else:
                        logger.error(f"已达最大重试次数（{max_attempts}次）")
                        raise Exception(f"调用LLM失败（尝试{max_attempts}次）: {last_error}")

            logger.info("LLM分析完成")

            # 清理final_content，去除markdown代码块标记
            cleaned_final_content = clean_json_response(final_content)

            # 8. 格式化输出
            result = {
                'status': 'success',
                'analysis_result': cleaned_final_content,  # 使用清理后的内容
                'reasoning_process': reasoning_content,
                'raw_indicators': raw_indicators,
                'metadata': {
                    'expert_type': 'market',
                    'fund_code': fund_code,
                    'fund_name': fund_name or '',
                    'sector': raw_data['sector'] or '未知',
                    'analysis_date': str(analysis_date),
                    'timestamp': datetime.now().isoformat()
                }
            }

            # 记录最终传出的内容（用于验证）
            logger.info("\n" + "=" * 80)
            logger.info("最终传出的内容（返回给调用方）:")
            logger.info("=" * 80)
            logger.info(f"status: {result['status']}")
            logger.info(f"analysis_result 长度: {len(result['analysis_result'])} 字符")
            logger.info(f"reasoning_process 长度: {len(result['reasoning_process'])} 字符")
            logger.info(f"metadata: {result['metadata']}")
            logger.info("=" * 80)

            logger.info("市场分析完成")

            return result

        except Exception as e:
            logger.error(f"市场分析失败: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'metadata': {
                    'expert_type': 'market',
                    'fund_code': fund_code,
                    'fund_name': fund_name or '',
                    'analysis_date': str(analysis_date),
                    'timestamp': datetime.now().isoformat()
                }
            }

    def analyze_for_decision(self, fund_code, analysis_date, fund_name=None):
        """
        为决策Agent执行分析（返回简化格式）

        Args:
            fund_code: 基金代码
            analysis_date: 分析日期
            fund_name: 基金名称（可选）

        Returns:
            dict: 决策Agent需要的格式
        """
        # 执行完整分析
        full_result = self.analyze(fund_code, analysis_date, fund_name)

        # 格式化为决策Agent需要的格式
        decision_input = format_output_for_decision_agent(full_result)

        return decision_input
