"""
工具函数模块
提供指标解释、市场摘要生成等功能
"""

import logging

logger = logging.getLogger(__name__)


def interpret_indicators(raw_indicators):
    """
    解释原始指标，生成Layer 2解释层数据

    Args:
        raw_indicators: 原始计算出的指标

    Returns:
        dict: 解释后的指标
    """
    logger.info("开始解释指标")

    interpreted = {
        'reits_market': {},
        'interest_rate': {},
        'stock_market': {},
        'market_sentiment': {},
        'sector_analysis': {}
    }

    try:
        # REITs市场解释
        reits = raw_indicators.get('reits_market', {})
        if reits:
            interpreted['reits_market'] = {
                'price_position': _interpret_price_position(reits),
                'momentum': _interpret_momentum(reits),
                'volatility': _interpret_volatility(reits),
                'trend': _interpret_trend(reits)
            }

        # 利率环境解释
        rate = raw_indicators.get('interest_rate', {})
        if rate:
            interpreted['interest_rate'] = {
                'rate_level': _interpret_rate_level(rate),
                'rate_trend': _interpret_rate_trend(rate),
                'correlation': _interpret_rate_correlation(rate),
                'overall_impact': _interpret_rate_impact(rate)
            }

        # 股市环境解释
        stock = raw_indicators.get('stock_market', {})
        if stock:
            interpreted['stock_market'] = {
                'shanghai_index': _interpret_shanghai_index(stock),
                'dividend_index': _interpret_dividend_index(stock),
                'vs_reits': _interpret_stock_vs_reits(stock),
                'risk_appetite': _interpret_risk_appetite(stock),
                'overall_impact': _interpret_stock_impact(stock)
            }

        # 市场情绪解释
        sentiment = raw_indicators.get('market_sentiment', {})
        if sentiment:
            interpreted['market_sentiment'] = {
                'turnover_rate': _interpret_turnover_rate(sentiment),
                'turnover_amount': _interpret_turnover_amount(sentiment),
                'volume_price': _interpret_volume_price(sentiment, reits),
                'overall_sentiment': _interpret_overall_sentiment(sentiment)
            }

        # 板块分析解释
        sector = raw_indicators.get('sector_analysis', {})
        if sector:
            interpreted['sector_analysis'] = _interpret_sector(sector, raw_indicators)

        logger.info("指标解释完成")

    except Exception as e:
        logger.error(f"指标解释失败: {e}", exc_info=True)

    return interpreted


def _interpret_price_position(reits):
    """解释价格位置"""
    percentile = reits.get('percentile_full', 50)

    if percentile < 20:
        level = "极低"
        interpretation = "处于历史极低位置，估值便宜"
    elif percentile < 40:
        level = "偏低"
        interpretation = "处于历史偏低位置，估值合理偏低"
    elif percentile < 60:
        level = "中性"
        interpretation = "处于历史中等位置"
    elif percentile < 80:
        level = "偏高"
        interpretation = "处于历史偏高位置，估值偏贵"
    else:
        level = "极高"
        interpretation = "处于历史极高位置，估值昂贵"

    return {
        'percentile_full': percentile,
        'percentile_1y': reits.get('percentile_1y'),
        'interpretation': interpretation,
        'level': level,
        'vs_high': reits.get('drawdown_from_high'),
        'vs_low': reits.get('rally_from_low'),
        'near_support_resistance': {
            'support_20d': reits.get('distance_to_20d_low'),
            'resistance_20d': reits.get('distance_to_20d_high'),
            'support_60d': reits.get('distance_to_60d_low'),
            'resistance_60d': reits.get('distance_to_60d_high')
        }
    }


def _interpret_momentum(reits):
    """解释动量"""
    change_20d = reits.get('change_20d', 0)

    if abs(change_20d or 0) < 2:
        strength = "弱"
        interpretation = "短期动量较弱，横盘震荡"
    elif change_20d > 5:
        strength = "强"
        interpretation = "短期动量强劲，上涨趋势明显"
    elif change_20d > 0:
        strength = "中"
        interpretation = "短期动量偏强，温和上涨"
    elif change_20d < -5:
        strength = "强（下跌）"
        interpretation = "短期动量较差，下跌趋势明显"
    else:
        strength = "中（下跌）"
        interpretation = "短期动量偏弱，温和下跌"

    return {
        'change_1d': reits.get('change_1d'),
        'change_5d': reits.get('change_5d'),
        'change_20d': change_20d,
        'change_60d': reits.get('change_60d'),
        'interpretation': interpretation,
        'strength': strength,
        'up_ratio_20d': reits.get('up_ratio_20d'),
        'consecutive_days': reits.get('consecutive_days')
    }


def _interpret_volatility(reits):
    """解释波动率"""
    vol_percentile = reits.get('volatility_percentile', 50)

    if vol_percentile < 20:
        level = "极低"
        interpretation = "波动率极低，市场平静"
    elif vol_percentile < 40:
        level = "偏低"
        interpretation = "波动率偏低，市场相对平静"
    elif vol_percentile < 60:
        level = "正常"
        interpretation = "波动率正常，市场情绪平稳"
    elif vol_percentile < 80:
        level = "偏高"
        interpretation = "波动率偏高，市场情绪波动"
    else:
        level = "极高"
        interpretation = "波动率极高，市场恐慌或过热"

    return {
        'vol_20d': reits.get('volatility_20d'),
        'vol_percentile': vol_percentile,
        'interpretation': interpretation,
        'level': level
    }


def _interpret_trend(reits):
    """解释趋势"""
    ma5 = reits.get('ma5')
    ma10 = reits.get('ma10')
    ma20 = reits.get('ma20')
    ma60 = reits.get('ma60')

    bullish = reits.get('bullish_alignment', False)
    bearish = reits.get('bearish_alignment', False)

    if bullish:
        pattern = "多头排列"
        ma_status = "价格在所有均线上方，多头排列"
    elif bearish:
        pattern = "空头排列"
        ma_status = "价格在所有均线下方，空头排列"
    else:
        pattern = "混乱"
        ma_status = "均线系统混乱，方向不明"

    rsi = reits.get('rsi_14', 50)
    if rsi > 70:
        level_rsi = "超买"
        interpretation_rsi = f"RSI={rsi:.1f}，超买区域，需警惕回调"
    elif rsi > 60:
        level_rsi = "偏强"
        interpretation_rsi = f"RSI={rsi:.1f}，偏强区域"
    elif rsi > 40:
        level_rsi = "中性"
        interpretation_rsi = f"RSI={rsi:.1f}，中性区域"
    elif rsi > 30:
        level_rsi = "偏弱"
        interpretation_rsi = f"RSI={rsi:.1f}，偏弱区域"
    else:
        level_rsi = "超卖"
        interpretation_rsi = f"RSI={rsi:.1f}，超卖区域，可能反弹"

    return {
        'ma_status': ma_status,
        'pattern': pattern,
        'interpretation': f"技术面{'强势' if bullish else '弱势' if bearish else '混乱'}",
        'macd': {
            'dif': reits.get('macd_dif'),
            'dea': reits.get('macd_dea'),
            'macd_bar': reits.get('macd_bar')
        },
        'rsi_14': rsi,
        'interpretation_rsi': interpretation_rsi,
        'level_rsi': level_rsi
    }


def _interpret_rate_level(rate):
    """解释利率水平（基于近1年数据）"""
    percentile = rate.get('rate_percentile_1y', 50)  # 使用近1年分位数

    if percentile < 20:
        level = "极低"
        interpretation = "利率处于近1年极低位，对REITs高度有利"
    elif percentile < 40:
        level = "偏低"
        interpretation = "利率处于近1年偏低位，对REITs有利"
    elif percentile < 60:
        level = "中性"
        interpretation = "利率处于近1年中位，影响中性"
    elif percentile < 80:
        level = "偏高"
        interpretation = "利率处于近1年偏高位，对REITs不利"
    else:
        level = "极高"
        interpretation = "利率处于近1年极高位，对REITs高度不利"

    return {
        'current': rate.get('current_rate'),
        'percentile': percentile,  # 只保留近1年分位数
        'interpretation': interpretation,
        'level': level,
        'vs_1.8': rate.get('distance_to_1.8')
    }


def _interpret_rate_trend(rate):
    """解释利率趋势"""
    trend = rate.get('rate_trend', "横盘震荡")
    change_20d = rate.get('rate_change_20d', 0)

    if "下行" in trend:
        impact = "有利"
        if abs(change_20d or 0) > 20:
            strength = "强"
        elif abs(change_20d or 0) > 10:
            strength = "中"
        else:
            strength = "弱"
    elif "上行" in trend:
        impact = "不利"
        if abs(change_20d or 0) > 20:
            strength = "强"
        elif abs(change_20d or 0) > 10:
            strength = "中"
        else:
            strength = "弱"
    else:
        impact = "中性"
        strength = "无"

    return {
        'change_20d': change_20d,
        'trend': trend,
        'strength': strength,
        'interpretation': f"利率{trend}，对REITs{impact}",
        'vs_ma20': rate.get('rate_vs_ma20'),
        'down_ratio_20d': rate.get('rate_down_ratio_20d')
    }


def _interpret_rate_correlation(rate):
    """解释利率相关性"""
    corr_60d = rate.get('corr_rate_reits_60d')
    corr_20d = rate.get('corr_rate_reits_20d')

    if corr_20d is None:
        return {}

    if abs(corr_20d) > 0.7:
        strength = "强"
    elif abs(corr_20d) > 0.4:
        strength = "中"
    else:
        strength = "弱"

    interpretation = f"利率与REITs相关系数{corr_20d:.2f}，相关性{strength}"

    return {
        'corr_60d': corr_60d,
        'corr_20d': corr_20d,
        'interpretation': interpretation,
        'strength': strength
    }


def _interpret_rate_impact(rate):
    """综合评估利率影响（基于近1年数据）"""
    level = rate.get('rate_percentile_1y', 50)  # 改用近1年分位数
    trend = rate.get('rate_trend', "横盘震荡")

    # 计算得分
    score = 5.0  # 基准分

    # 利率水平贡献（基于近1年分位数）
    if level < 20:
        score += 3
    elif level < 40:
        score += 1.5
    elif level > 80:
        score -= 3
    elif level > 60:
        score -= 1.5

    # 利率趋势贡献
    if "明确下行" in trend:
        score += 2
        impact = "强烈正面"
    elif "缓慢下行" in trend:
        score += 1
        impact = "正面"
    elif "明确上行" in trend:
        score -= 2
        impact = "强烈负面"
    elif "缓慢上行" in trend:
        score -= 1
        impact = "负面"
    else:
        impact = "中性"

    score = max(1, min(10, score))  # 限制在1-10

    return {
        'impact': impact,
        'score': round(score, 1),
        'reasoning': f"利率处于{'低' if level < 40 else '高' if level > 60 else '中'}位（近1年）且{trend}"
    }


def _interpret_shanghai_index(stock):
    """解释上证指数"""
    trend = stock.get('sh_trend', "震荡")
    change_20d = stock.get('sh_change_20d', 0)

    return {
        'current': stock.get('sh_index'),
        'percentile': stock.get('sh_percentile_1y'),
        'change_20d': change_20d,
        'trend': trend,
        'rsi_14': stock.get('sh_rsi_14'),
        'interpretation': f"上证指数{trend}，近20日{'上涨' if change_20d > 0 else '下跌'}{abs(change_20d or 0):.1f}%"
    }


def _interpret_dividend_index(stock):
    """解释红利指数"""
    change_20d = stock.get('dividend_change_20d', 0)
    consecutive = stock.get('dividend_consecutive_up', 0)

    if change_20d > 5:
        strength = "强势"
    elif change_20d > 2:
        strength = "偏强"
    elif change_20d < -5:
        strength = "弱势"
    elif change_20d < -2:
        strength = "偏弱"
    else:
        strength = "平稳"

    return {
        'current': stock.get('dividend_index'),
        'percentile': stock.get('dividend_percentile_1y'),
        'change_20d': change_20d,
        'consecutive_up': consecutive,
        'interpretation': f"红利指数{strength}，近20日{'上涨' if change_20d > 0 else '下跌'}{abs(change_20d or 0):.1f}%",
        'vs_sh_20d': stock.get('dividend_vs_sh_20d')
    }


def _interpret_stock_vs_reits(stock):
    """解释股市与REITs对比"""
    reits_vs_div = stock.get('reits_vs_dividend_20d')
    seesaw_strong = stock.get('seesaw_strong', False)

    if seesaw_strong:
        interpretation = "强跷跷板效应：红利强势上涨，REITs明显下跌，资金分流明显"
        impact = "负面"
    elif reits_vs_div is not None and reits_vs_div < -5:
        interpretation = "REITs明显跑输红利指数，可能存在资金分流"
        impact = "负面"
    elif reits_vs_div is not None and reits_vs_div > 5:
        interpretation = "REITs明显跑赢红利指数，可能有资金回流"
        impact = "正面"
    else:
        interpretation = "REITs与红利指数表现相当"
        impact = "中性"

    return {
        'reits_vs_sh_20d': stock.get('reits_vs_sh_20d'),
        'reits_vs_dividend_20d': reits_vs_div,
        'interpretation': interpretation,
        'seesaw_effect': {
            'strong_seesaw': seesaw_strong,
            'impact': impact
        },
        'correlation': {
            'corr_60d': stock.get('corr_dividend_reits_60d'),
            'corr_20d': stock.get('corr_dividend_reits_20d')
        }
    }


def _interpret_risk_appetite(stock):
    """解释风险偏好"""
    # 这里简化处理，可以根据实际需要扩展
    sh_change = stock.get('sh_change_20d', 0)
    vol = stock.get('sh_volatility_20d', 20)

    if sh_change > 3 and vol < 25:
        risk_on = True
        interpretation = "Risk on：股市稳定上涨，风险偏好高"
        impact = "中性偏负，股市吸引力上升"
    elif sh_change < -3 or vol > 30:
        risk_on = False
        interpretation = "Risk off：股市下跌或高波动，风险偏好低"
        impact = "中性偏正，避险需求上升"
    else:
        risk_on = None
        interpretation = "风险偏好中性"
        impact = "中性"

    return {
        'vol_20d': vol,
        'risk_on': risk_on,
        'interpretation': interpretation,
        'impact_on_reits': impact
    }


def _interpret_stock_impact(stock):
    """综合评估股市影响"""
    reits_vs_div = stock.get('reits_vs_dividend_20d', 0)
    seesaw = stock.get('seesaw_strong', False)

    score = 5.0

    # 跷跷板效应
    if seesaw:
        score -= 3
        impact = "强烈负面"
    elif reits_vs_div < -5:
        score -= 1.5
        impact = "负面"
    elif reits_vs_div > 5:
        score += 1.5
        impact = "正面"
    else:
        impact = "中性"

    score = max(1, min(10, score))

    return {
        'impact': impact,
        'score': round(score, 1),
        'reasoning': f"{'强跷跷板效应' if seesaw else 'REITs相对表现' + ('较好' if reits_vs_div > 0 else '较差')}"
    }


def _interpret_turnover_rate(sentiment):
    """解释换手率"""
    rate = sentiment.get('turnover_rate')
    percentile = sentiment.get('turnover_rate_percentile', 50)
    level = sentiment.get('turnover_rate_level', "正常")

    interpretation = f"换手率{rate:.2f}%，处于{level}水平（{percentile:.0f}分位）"

    return {
        'current': rate,
        'percentile': percentile,
        'level': level,
        'interpretation': interpretation,
        'ma5': sentiment.get('turnover_rate_ma5'),
        'ma20': sentiment.get('turnover_rate_ma20'),
        'vs_ma20': sentiment.get('turnover_rate_vs_ma20')
    }


def _interpret_turnover_amount(sentiment):
    """解释成交额"""
    amount = sentiment.get('turnover_amount')
    vs_yesterday = sentiment.get('turnover_vs_yesterday', 0)
    vs_last_week = sentiment.get('turnover_vs_last_week', 0)

    if vs_last_week > 20:
        trend = "大幅放量"
    elif vs_last_week > 10:
        trend = "明显放量"
    elif vs_last_week < -20:
        trend = "大幅萎缩"
    elif vs_last_week < -10:
        trend = "明显萎缩"
    else:
        trend = "基本持平"

    return {
        'current': amount,
        'vs_yesterday': vs_yesterday,
        'vs_last_week': vs_last_week,
        'interpretation': f"成交额{amount:.1f}亿，较上周{trend}"
    }


def _interpret_volume_price(sentiment, reits):
    """解释量价关系"""
    turnover_rate = sentiment.get('turnover_rate', 0)
    turnover_ma20 = sentiment.get('turnover_rate_ma20', turnover_rate)
    change_1d = reits.get('change_1d', 0) if reits else 0

    if change_1d > 0.5 and turnover_rate > turnover_ma20:
        pattern = "放量上涨"
        healthy = "健康"
        interpretation = "放量上涨，买盘积极，健康信号"
    elif change_1d > 0.5 and turnover_rate < turnover_ma20:
        pattern = "缩量上涨"
        healthy = "需警惕"
        interpretation = "缩量上涨，上涨乏力，需警惕"
    elif change_1d < -0.5 and turnover_rate > turnover_ma20:
        pattern = "放量下跌"
        healthy = "恐慌"
        interpretation = "放量下跌，抛压较大"
    elif change_1d < -0.5 and turnover_rate < turnover_ma20:
        pattern = "缩量下跌"
        healthy = "中性偏正面"
        interpretation = "缩量下跌，抛压释放，可能为后续反弹蓄势"
    else:
        pattern = "震荡"
        healthy = "中性"
        interpretation = "量价关系正常"

    return {
        'pattern': pattern,
        'interpretation': interpretation,
        'healthy': healthy
    }


def _interpret_overall_sentiment(sentiment):
    """综合评估市场情绪"""
    percentile = sentiment.get('turnover_rate_percentile', 50)

    if percentile < 10:
        sentiment_level = "极低（冰点）"
        score = 2
    elif percentile < 30:
        sentiment_level = "偏低"
        score = 4
    elif percentile < 70:
        sentiment_level = "正常"
        score = 5
    elif percentile < 90:
        sentiment_level = "偏高（活跃）"
        score = 7
    else:
        sentiment_level = "极高（过热）"
        score = 9

    return {
        'sentiment': sentiment_level,
        'score': score,
        'reasoning': f"换手率分位数{percentile:.0f}%，市场情绪{sentiment_level}"
    }


def _interpret_sector(sector, raw_indicators):
    """解释板块情况"""
    change_20d = sector.get('sector_change_20d')
    vs_market = sector.get('sector_vs_market_20d')
    rank = sector.get('sector_rank')
    total = sector.get('sector_total_count', 8)

    if vs_market and vs_market > 2:
        performance = "明显跑赢"
    elif vs_market and vs_market > 0:
        performance = "小幅跑赢"
    elif vs_market and vs_market < -2:
        performance = "明显跑输"
    elif vs_market:
        performance = "小幅跑输"
    else:
        performance = "持平"

    return {
        'performance': {
            'change_20d': change_20d,
            'vs_market_20d': vs_market,
            'rank': f"{rank}/{total}" if rank else "未知",
            'interpretation': f"近20日{performance}大盘"
        }
    }


def generate_market_summary(raw_indicators, interpreted_metrics):
    """
    生成市场摘要（Layer 3）

    Args:
        raw_indicators: 原始指标
        interpreted_metrics: 解释后的指标

    Returns:
        dict: 市场摘要
    """
    logger.info("开始生成市场摘要")

    summary = {
        'overall_state': {},
        'key_signals': {
            'positive': [],
            'negative': [],
            'neutral': []
        },
        'risk_level': {}
    }

    try:
        # 整体状态
        reits_trend = _summarize_reits_trend(interpreted_metrics.get('reits_market', {}))
        rate_trend = interpreted_metrics.get('interest_rate', {}).get('rate_trend', {}).get('trend', '未知')
        stock_trend = interpreted_metrics.get('stock_market', {}).get('shanghai_index', {}).get('trend', '未知')
        sentiment = interpreted_metrics.get('market_sentiment', {}).get('overall_sentiment', {}).get('sentiment', '未知')

        summary['overall_state'] = {
            'reits_trend': reits_trend,
            'interest_rate_trend': rate_trend,
            'stock_market_trend': stock_trend,
            'sentiment': sentiment
        }

        # 关键信号
        _collect_key_signals(summary['key_signals'], raw_indicators, interpreted_metrics)

        # 风险等级
        summary['risk_level'] = _assess_risk_level(raw_indicators, interpreted_metrics)

        logger.info("市场摘要生成完成")

    except Exception as e:
        logger.error(f"市场摘要生成失败: {e}", exc_info=True)

    return summary


def _summarize_reits_trend(reits_market):
    """总结REITs趋势"""
    if not reits_market:
        return "未知"

    trend = reits_market.get('trend', {})
    pattern = trend.get('pattern', '混乱')

    if pattern == "多头排列":
        return "上涨趋势"
    elif pattern == "空头排列":
        return "下跌趋势"
    else:
        momentum = reits_market.get('momentum', {})
        change_20d = momentum.get('change_20d', 0)
        if change_20d > 2:
            return "震荡偏强"
        elif change_20d < -2:
            return "震荡偏弱"
        else:
            return "震荡"


def _collect_key_signals(signals, raw_indicators, interpreted_metrics):
    """收集关键信号"""
    # 正面信号
    rate_impact = interpreted_metrics.get('interest_rate', {}).get('overall_impact', {})
    if rate_impact.get('score', 5) >= 7:
        signals['positive'].append(f"利率环境有利（{rate_impact.get('reasoning', '')}）")

    # 负面信号
    stock_impact = interpreted_metrics.get('stock_market', {}).get('overall_impact', {})
    if stock_impact.get('score', 5) <= 4:
        signals['negative'].append(f"股市环境不利（{stock_impact.get('reasoning', '')}）")

    seesaw = raw_indicators.get('stock_market', {}).get('seesaw_strong', False)
    if seesaw:
        signals['negative'].append("强跷跷板效应，红利板块分流资金明显")

    # 中性信号
    vol_level = interpreted_metrics.get('reits_market', {}).get('volatility', {}).get('level', '')
    if vol_level == "正常":
        signals['neutral'].append("波动率处于正常水平")


def _assess_risk_level(raw_indicators, interpreted_metrics):
    """评估风险等级"""
    risk_score = 0  # 风险分数，越高越危险

    # 利率风险
    rate_trend = raw_indicators.get('interest_rate', {}).get('rate_trend', '')
    if "上行" in rate_trend:
        risk_score += 2

    # 股市风险
    seesaw = raw_indicators.get('stock_market', {}).get('seesaw_strong', False)
    if seesaw:
        risk_score += 2

    # 情绪风险
    turnover_percentile = raw_indicators.get('market_sentiment', {}).get('turnover_rate_percentile', 50)
    if turnover_percentile < 10:
        risk_score += 1

    # 确定等级
    if risk_score >= 4:
        level = "高"
        alert_color = "🔴 红色"
    elif risk_score >= 2:
        level = "中等"
        alert_color = "🟡 黄色"
    else:
        level = "低"
        alert_color = "🟢 绿色"

    return {
        'level': level,
        'alert_color': alert_color,
        'score': risk_score
    }


def determine_quadrant(raw_indicators):
    """
    确定四象限位置

    Args:
        raw_indicators: 原始指标

    Returns:
        dict: 象限信息
    """
    logger.info("开始确定四象限")

    # 利率趋势
    rate_trend = raw_indicators.get('interest_rate', {}).get('rate_trend', '横盘震荡')
    rate_down = "下行" in rate_trend
    rate_up = "上行" in rate_trend

    # 股市状态
    stock_trend = raw_indicators.get('stock_market', {}).get('sh_trend', '震荡')
    # 细化股市判断
    stock_up = stock_trend in ["牛市", "震荡偏强"]
    stock_down = stock_trend in ["熊市", "震荡偏弱"]
    stock_neutral = stock_trend == "震荡"

    # 确定象限
    if rate_down and stock_up:
        # 象限I：利率下行 + 股市上涨
        quadrant = "象限I"
        description = f"利率下行 + 股市上涨（{stock_trend}）"
        favorable_level = "谨慎乐观"
        recommended_position = "70%"

    elif rate_up and stock_up:
        # 象限II：利率上行 + 股市上涨
        quadrant = "象限II"
        description = f"利率上行 + 股市上涨（{stock_trend}）"
        favorable_level = "最危险（双重挤压）"
        recommended_position = "30%"

    elif rate_down and (stock_down or stock_neutral):
        # 象限III：利率下行 + 股市下跌/震荡
        quadrant = "象限III"
        description = f"利率下行 + 股市{'下跌' if stock_down else '震荡'}（{stock_trend}）"
        favorable_level = "最佳配置期"
        recommended_position = "90-100%"

    elif rate_up and stock_down:
        # 象限IV：利率上行 + 股市下跌
        quadrant = "象限IV"
        description = f"利率上行 + 股市下跌（{stock_trend}）"
        favorable_level = "谨慎防御"
        recommended_position = "50%"

    elif rate_up and stock_neutral:
        # 利率上行 + 股市震荡：偏向象限IV
        quadrant = "过渡区（偏象限IV）"
        description = f"利率上行 + 股市震荡（{stock_trend}）"
        favorable_level = "偏不利（利率压制）"
        recommended_position = "50-60%"

    else:
        # 利率横盘的各种情况
        quadrant = "过渡区"
        description = f"利率横盘 + 股市{stock_trend}"

        if stock_up:
            # 利率横盘 + 股市上涨：需警惕资金分流
            favorable_level = "中性偏负（股市分流资金）"
            recommended_position = "50-60%"
        elif stock_down:
            # 利率横盘 + 股市下跌：有避险需求
            favorable_level = "中性偏正（避险需求）"
            recommended_position = "60-70%"
        else:
            # 利率横盘 + 股市震荡
            favorable_level = "中性"
            recommended_position = "60-70%"

    logger.info(f"四象限定位：{quadrant}")

    return {
        'position': quadrant,
        'description': description,
        'favorable_level': favorable_level,
        'recommended_position': recommended_position,
        'reasoning': f"利率{rate_trend}，股市{stock_trend}"
    }


def format_output_for_decision_agent(full_result):
    """
    格式化输出给决策Agent

    Args:
        full_result: 完整分析结果

    Returns:
        dict: 决策Agent需要的格式
    """
    logger.info("格式化输出给决策Agent")

    # 这里简化处理，返回主要内容
    # 实际可以进一步解析LLM的输出，提取结构化信息

    return {
        'analysis_type': 'market_overall_analysis',
        'analysis_date': full_result['metadata']['analysis_date'],
        'fund_info': {
            'fund_code': full_result['metadata']['fund_code'],
            'fund_name': full_result['metadata']['fund_name'],
            'sector': full_result['metadata']['sector']
        },
        'analysis_result': full_result['analysis_result'],
        'metadata': full_result['metadata']
    }
