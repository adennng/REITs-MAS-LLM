#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REITs价格动量分析Agent
从价格走势、技术指标、成交量等维度分析投资机会，为决策提供支持
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pymysql
import pandas as pd
import numpy as np
from openai import OpenAI

# 导入配置（使用相对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 以下导入依赖于上面的 sys.path 设置，必须在此之后导入
from config.db_config import get_db_reits_config  # noqa: E402
from config.model_config import MODEL_CONFIG  # noqa: E402
from agent_direction import config as direction_config  # noqa: E402
from agent_price.daily_threshold_calculator import calculate_daily_volatility_threshold  # noqa: E402


# ============================================================================
# 配置参数 - 可手动修改
# ============================================================================
# 获取历史数据的天数（需要足够的数据计算60日均线等长期指标）
HISTORY_DAYS = 150  # 获取150个自然日的数据，约100个交易日

# 日志模式
LOG_MODE = 'detailed'  # 'detailed' 或 'simple'
# detailed: 记录全过程(入参、数据库查询结果、指标计算、LLM调用、推理过程等)
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
    logger = logging.getLogger('PriceAnalyzer')
    logger.setLevel(logging.DEBUG if log_mode == 'detailed' else logging.INFO)

    # 清除已有的处理器（避免重复）
    logger.handlers.clear()

    # 确定日志目录（相对路径：../log/）
    log_dir = os.path.join(parent_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = 'price_analyzer.log'
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


def get_price_data(fund_code: str, end_date: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    获取指定基金的历史价格数据

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


def get_dividend_data(fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定基金在指定时间范围内的分红数据

    Args:
        fund_code: 基金代码
        start_date: 起始日期 (格式: YYYY-MM-DD)
        end_date: 截止日期 (格式: YYYY-MM-DD)

    Returns:
        DataFrame: 包含 ex_dividend_date, dividend_per_share 的数据框，按日期升序排列
    """
    logger.debug(f"开始查询基金 {fund_code} 的分红数据，时间范围：{start_date} 至 {end_date}")

    conn = get_db_connection()
    try:
        sql = """
            SELECT ex_dividend_date, dividend_per_share
            FROM dividend
            WHERE fund_code = %s
              AND ex_dividend_date >= %s
              AND ex_dividend_date <= %s
            ORDER BY ex_dividend_date ASC
        """

        df = pd.read_sql(sql, conn, params=(fund_code, start_date, end_date))

        logger.debug(f"查询到 {len(df)} 条分红记录")
        if logger.level == logging.DEBUG and len(df) > 0:
            for idx, row in df.iterrows():
                logger.debug(f"  分红{idx+1}: {row['ex_dividend_date']} - 每份{row['dividend_per_share']}元")

        return df

    finally:
        conn.close()


def calculate_adjusted_price(price_df: pd.DataFrame, dividend_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算前复权价格（支持多次分红）

    前复权算法：
    1. 从最新日期往历史回溯
    2. 遇到除息日，将该日期之前的所有收盘价减去分红金额
    3. 当前价格保持真实，历史价格被调整

    Args:
        price_df: 价格数据框（包含 trade_date, close, vol）
        dividend_df: 分红数据框（包含 ex_dividend_date, dividend_per_share）

    Returns:
        DataFrame: 添加了 adjusted_close 列的价格数据框
    """
    logger.debug("开始计算前复权价格")

    # 复制数据框，避免修改原数据
    df = price_df.copy()
    df['adjusted_close'] = df['close'].copy()

    if len(dividend_df) == 0:
        logger.debug("无分红记录，复权价格等于原价格")
        return df

    # 将日期转换为 datetime 类型
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    dividend_df = dividend_df.copy()
    dividend_df['ex_dividend_date'] = pd.to_datetime(dividend_df['ex_dividend_date'])

    # 按除息日期降序排列（从最新往历史回溯）
    dividend_df = dividend_df.sort_values('ex_dividend_date', ascending=False)

    # 累计调整金额（从最新日期开始累计）
    cumulative_adjustment = 0.0

    # 从最新除息日开始，逐步向历史调整
    for idx, div_row in dividend_df.iterrows():
        ex_date = div_row['ex_dividend_date']
        dividend_amount = float(div_row['dividend_per_share'])

        logger.debug(f"处理除息日 {ex_date.date()}，分红金额：{dividend_amount}元")

        # 将除息日之前的所有价格减去当前分红金额
        mask = df['trade_date'] < ex_date
        df.loc[mask, 'adjusted_close'] -= dividend_amount

        cumulative_adjustment += dividend_amount

    logger.debug(f"复权处理完成，累计调整金额：{cumulative_adjustment}元")
    logger.debug(f"复权后价格范围：{df['adjusted_close'].min():.3f} - {df['adjusted_close'].max():.3f}")

    return df


def calculate_ma(series: pd.Series, window: int) -> pd.Series:
    """计算移动平均线"""
    return series.rolling(window=window, min_periods=window).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI（相对强弱指标）

    Args:
        series: 价格序列
        period: 计算周期

    Returns:
        RSI值序列
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD指标

    Args:
        series: 价格序列
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期

    Returns:
        (DIF, DEA, MACD) 三个序列
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = dif - dea

    return dif, dea, macd


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带

    Args:
        series: 价格序列
        window: 计算周期
        num_std: 标准差倍数

    Returns:
        (上轨, 中轨, 下轨) 三个序列
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算ATR（平均真实波幅）

    注意：由于我们只有收盘价，这里用简化版本
    使用收盘价的波动范围作为替代

    Args:
        high: 最高价序列（这里用收盘价的滚动最大值替代）
        low: 最低价序列（这里用收盘价的滚动最小值替代）
        close: 收盘价序列
        period: 计算周期

    Returns:
        ATR值序列
    """
    # 简化版本：使用收盘价的变化范围
    tr = close.diff().abs()
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_volatility_threshold(price_history: List[float]) -> Optional[float]:
    """
    计算基于历史波动的动态阈值（用于判断价格变动显著性）

    使用新的每日动态阈值计算方案：
    1. 为每个交易日独立计算阈值
    2. 只使用该日之前的历史数据
    3. 包含波动率计算、自适应乘数、边界约束等步骤

    根据新方案的核心逻辑：
    - 边界约束通过 max(lower_bound, min(dynamic_threshold, upper_bound)) 保证阈值合理性
    - lower_bound 本身来自历史分位数，已经是合理的下限
    - 不需要固定的最小值兜底

    这个阈值用于判断价格变动是否显著：
    - 收益率 > threshold → 显著上涨
    - 收益率 < -threshold → 显著下跌
    - |收益率| ≤ threshold → 横盘震荡

    Args:
        price_history: 价格序列（从旧到新，长度>=2）

    Returns:
        Optional[float]: 动态阈值（百分比形式，如 0.015 表示 1.5%），如果数据不足或计算失败返回 None
    """
    # 使用新的每日动态阈值计算模块
    return calculate_daily_volatility_threshold(price_history)


def analyze_ma_alignment(current_price: float, ma5: float, ma10: float, ma20: float, ma60: float) -> str:
    """
    分析均线排列状态

    Args:
        current_price: 当前价格
        ma5, ma10, ma20, ma60: 各周期均线值

    Returns:
        均线排列状态：'多头排列'、'空头排列'、'混乱'
    """
    # 多头排列：价格 > MA5 > MA10 > MA20 > MA60
    if current_price > ma5 > ma10 > ma20 > ma60:
        return '多头排列'

    # 空头排列：价格 < MA5 < MA10 < MA20 < MA60
    if current_price < ma5 < ma10 < ma20 < ma60:
        return '空头排列'

    return '混乱'


def analyze_volume_price_relationship(price_change: float, volume_ratio: float) -> str:
    """
    分析量价关系

    Args:
        price_change: 价格变化百分比
        volume_ratio: 量比（当日量/均量）

    Returns:
        量价关系描述
    """
    # 定义阈值
    price_up = price_change > 0.1  # 涨幅超过0.1%
    price_down = price_change < -0.1  # 跌幅超过0.1%
    volume_up = volume_ratio > 1.2  # 量比超过1.2
    volume_down = volume_ratio < 0.8  # 量比低于0.8

    if price_up and volume_up:
        return '价涨量增（健康上涨）'
    elif price_up and volume_down:
        return '价涨量缩（上涨乏力）'
    elif price_down and volume_up:
        return '价跌量增（恐慌下跌）'
    elif price_down and volume_down:
        return '价跌量缩（下跌减弱）'
    else:
        return '价量平稳'


def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    计算所有技术指标

    Args:
        df: 包含 adjusted_close 和 vol 的数据框

    Returns:
        包含所有技术指标的字典
    """
    logger.debug("开始计算技术指标")

    if len(df) < 60:
        logger.warning(f"数据量不足（仅{len(df)}条），可能无法计算部分长期指标")

    # 使用复权价格进行计算
    close = df['adjusted_close']
    volume = df['vol']

    # 获取当前（最新）数据
    current_price = close.iloc[-1]
    current_volume = volume.iloc[-1]
    prev_price = close.iloc[-2] if len(close) > 1 else current_price

    # ========== 1. 趋势指标 ==========
    ma5 = calculate_ma(close, 5)
    ma10 = calculate_ma(close, 10)
    ma20 = calculate_ma(close, 20)
    ma60 = calculate_ma(close, 60)

    # 当前各均线值
    current_ma5 = ma5.iloc[-1] if not ma5.isna().iloc[-1] else None
    current_ma10 = ma10.iloc[-1] if not ma10.isna().iloc[-1] else None
    current_ma20 = ma20.iloc[-1] if not ma20.isna().iloc[-1] else None
    current_ma60 = ma60.iloc[-1] if not ma60.isna().iloc[-1] else None

    # 均线偏离度
    ma5_deviation = ((current_price - current_ma5) / current_ma5 * 100) if current_ma5 else None
    ma10_deviation = ((current_price - current_ma10) / current_ma10 * 100) if current_ma10 else None
    ma20_deviation = ((current_price - current_ma20) / current_ma20 * 100) if current_ma20 else None
    ma60_deviation = ((current_price - current_ma60) / current_ma60 * 100) if current_ma60 else None

    # 均线排列状态
    ma_alignment = '数据不足'
    if all([current_ma5, current_ma10, current_ma20, current_ma60]):
        ma_alignment = analyze_ma_alignment(current_price, current_ma5, current_ma10, current_ma20, current_ma60)

    # 价格趋势（涨跌幅）
    price_change_1d = ((current_price - prev_price) / prev_price * 100) if prev_price else 0
    price_change_5d = ((current_price - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 5 else None
    price_change_20d = ((current_price - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 20 else None
    price_change_60d = ((current_price - close.iloc[-61]) / close.iloc[-61] * 100) if len(close) > 60 else None

    # ========== 2. 动量指标 ==========
    rsi6 = calculate_rsi(close, 6)
    rsi12 = calculate_rsi(close, 12)
    rsi24 = calculate_rsi(close, 24)

    current_rsi6 = rsi6.iloc[-1] if not rsi6.isna().iloc[-1] else None
    current_rsi12 = rsi12.iloc[-1] if not rsi12.isna().iloc[-1] else None
    current_rsi24 = rsi24.iloc[-1] if not rsi24.isna().iloc[-1] else None

    # RSI状态判断
    rsi6_status = '数据不足'
    if current_rsi6 is not None:
        if current_rsi6 > 70:
            rsi6_status = '超买'
        elif current_rsi6 < 30:
            rsi6_status = '超卖'
        else:
            rsi6_status = '正常'

    # MACD
    dif, dea, macd_hist = calculate_macd(close)
    current_dif = dif.iloc[-1] if not dif.isna().iloc[-1] else None
    current_dea = dea.iloc[-1] if not dea.isna().iloc[-1] else None
    current_macd = macd_hist.iloc[-1] if not macd_hist.isna().iloc[-1] else None

    # MACD信号判断
    macd_signal = '数据不足'
    if len(dif) > 1 and len(dea) > 1:
        prev_dif = dif.iloc[-2]
        prev_dea = dea.iloc[-2]
        if not pd.isna(prev_dif) and not pd.isna(prev_dea) and not pd.isna(current_dif) and not pd.isna(current_dea):
            # 金叉：DIF上穿DEA
            if prev_dif <= prev_dea and current_dif > current_dea:
                macd_signal = '金叉'
            # 死叉：DIF下穿DEA
            elif prev_dif >= prev_dea and current_dif < current_dea:
                macd_signal = '死叉'
            else:
                macd_signal = '无明显信号'

    # 动量值（10日）
    momentum_10d = current_price - close.iloc[-11] if len(close) > 10 else None

    # ========== 3. 波动率指标 ==========
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)

    current_bb_upper = bb_upper.iloc[-1] if not bb_upper.isna().iloc[-1] else None
    current_bb_middle = bb_middle.iloc[-1] if not bb_middle.isna().iloc[-1] else None
    current_bb_lower = bb_lower.iloc[-1] if not bb_lower.isna().iloc[-1] else None

    # 布林带位置
    bb_position = '数据不足'
    bb_position_pct = None
    if all([current_bb_upper, current_bb_middle, current_bb_lower]):
        bb_width = current_bb_upper - current_bb_lower
        if bb_width > 0:
            bb_position_pct = ((current_price - current_bb_middle) / (bb_width / 2) * 100)
            if current_price > current_bb_upper:
                bb_position = '突破上轨'
            elif current_price < current_bb_lower:
                bb_position = '跌破下轨'
            elif abs(bb_position_pct) < 30:
                bb_position = '居中'
            elif bb_position_pct > 0:
                bb_position = '偏上'
            else:
                bb_position = '偏下'

    # 价格波动率（20日标准差）
    volatility_20d = close.rolling(window=20).std().iloc[-1] if len(close) >= 20 else None
    volatility_20d_pct = (volatility_20d / current_price * 100) if volatility_20d else None

    # ATR（简化版）
    atr_14d = calculate_atr(close, close, close, 14).iloc[-1] if len(close) >= 14 else None

    # ========== 4. 成交量指标 ==========
    vol_ma5 = calculate_ma(volume, 5)
    vol_ma10 = calculate_ma(volume, 10)
    vol_ma20 = calculate_ma(volume, 20)

    current_vol_ma5 = vol_ma5.iloc[-1] if not vol_ma5.isna().iloc[-1] else None
    current_vol_ma10 = vol_ma10.iloc[-1] if not vol_ma10.isna().iloc[-1] else None
    current_vol_ma20 = vol_ma20.iloc[-1] if not vol_ma20.isna().iloc[-1] else None

    # 量比
    volume_ratio = (current_volume / current_vol_ma5) if current_vol_ma5 and current_vol_ma5 > 0 else None

    # 量价关系
    volume_price_relationship = '数据不足'
    if volume_ratio is not None:
        volume_price_relationship = analyze_volume_price_relationship(price_change_1d, volume_ratio)

    # ========== 5. 支撑压力位 ==========
    # 近期高低点
    high_5d = close.rolling(window=5).max().iloc[-1] if len(close) >= 5 else None
    low_5d = close.rolling(window=5).min().iloc[-1] if len(close) >= 5 else None
    high_10d = close.rolling(window=10).max().iloc[-1] if len(close) >= 10 else None
    low_10d = close.rolling(window=10).min().iloc[-1] if len(close) >= 10 else None
    high_20d = close.rolling(window=20).max().iloc[-1] if len(close) >= 20 else None
    low_20d = close.rolling(window=20).min().iloc[-1] if len(close) >= 20 else None
    high_60d = close.rolling(window=60).max().iloc[-1] if len(close) >= 60 else None
    low_60d = close.rolling(window=60).min().iloc[-1] if len(close) >= 60 else None

    # ========== 6. 价格形态与统计 ==========
    # 连续涨跌天数
    price_changes = close.diff()
    consecutive_days = 0
    for i in range(len(price_changes) - 1, 0, -1):
        if pd.isna(price_changes.iloc[i]):
            break
        if i == len(price_changes) - 1:
            consecutive_days = 1
            continue
        # 判断方向是否一致
        if (price_changes.iloc[i] > 0 and price_changes.iloc[i-1] > 0) or \
           (price_changes.iloc[i] < 0 and price_changes.iloc[i-1] < 0):
            consecutive_days += 1
        else:
            break

    consecutive_direction = '上涨' if price_changes.iloc[-1] > 0 else '下跌' if price_changes.iloc[-1] < 0 else '平盘'

    # 近20日涨跌统计
    recent_20d_changes = price_changes.tail(20) if len(price_changes) >= 20 else price_changes
    up_days = (recent_20d_changes > 0).sum()
    down_days = (recent_20d_changes < 0).sum()

    # 振幅统计（使用收盘价的日变化幅度）
    daily_amplitude = close.pct_change().abs() * 100
    avg_amplitude_20d = daily_amplitude.tail(20).mean() if len(daily_amplitude) >= 20 else None
    max_amplitude_20d = daily_amplitude.tail(20).max() if len(daily_amplitude) >= 20 else None

    # 最近5日每日涨跌幅（用于判断整盘状态）
    recent_5d_changes = []
    if len(close) >= 6:
        for i in range(-5, 0):
            daily_change = ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * 100)
            recent_5d_changes.append({
                'date': str(df['trade_date'].iloc[i]),
                'change_pct': daily_change
            })
    else:
        recent_5d_changes = []

    # ========== 7. 动态波动阈值 ==========
    # 使用新的每日动态阈值计算方案
    # 新方案会自动使用所有可用历史数据，并确保无未来信息泄露
    # 计算最后一个交易日（当前日）的阈值
    price_for_threshold = close.tolist()
    volatility_threshold = calculate_volatility_threshold(price_for_threshold)
    # 如果计算失败返回 None，边界约束已经保证了阈值的合理性，不需要额外兜底

    # ========== 汇总所有指标 ==========
    indicators = {
        # 基础信息
        'current_price': current_price,
        'prev_price': prev_price,
        'price_change_1d': price_change_1d,

        # 趋势指标
        'ma5': current_ma5,
        'ma10': current_ma10,
        'ma20': current_ma20,
        'ma60': current_ma60,
        'ma5_deviation': ma5_deviation,
        'ma10_deviation': ma10_deviation,
        'ma20_deviation': ma20_deviation,
        'ma60_deviation': ma60_deviation,
        'ma_alignment': ma_alignment,
        'price_change_5d': price_change_5d,
        'price_change_20d': price_change_20d,
        'price_change_60d': price_change_60d,

        # 动量指标
        'rsi6': current_rsi6,
        'rsi12': current_rsi12,
        'rsi24': current_rsi24,
        'rsi6_status': rsi6_status,
        'dif': current_dif,
        'dea': current_dea,
        'macd': current_macd,
        'macd_signal': macd_signal,
        'momentum_10d': momentum_10d,

        # 波动率指标
        'bb_upper': current_bb_upper,
        'bb_middle': current_bb_middle,
        'bb_lower': current_bb_lower,
        'bb_position': bb_position,
        'bb_position_pct': bb_position_pct,
        'volatility_20d': volatility_20d,
        'volatility_20d_pct': volatility_20d_pct,
        'atr_14d': atr_14d,
        'volatility_threshold': volatility_threshold,

        # 成交量指标
        'current_volume': current_volume,
        'vol_ma5': current_vol_ma5,
        'vol_ma10': current_vol_ma10,
        'vol_ma20': current_vol_ma20,
        'volume_ratio': volume_ratio,
        'volume_price_relationship': volume_price_relationship,

        # 支撑压力位
        'high_5d': high_5d,
        'low_5d': low_5d,
        'high_10d': high_10d,
        'low_10d': low_10d,
        'high_20d': high_20d,
        'low_20d': low_20d,
        'high_60d': high_60d,
        'low_60d': low_60d,

        # 价格形态
        'consecutive_days': consecutive_days,
        'consecutive_direction': consecutive_direction,
        'up_days_20d': up_days,
        'down_days_20d': down_days,
        'avg_amplitude_20d': avg_amplitude_20d,
        'max_amplitude_20d': max_amplitude_20d,
        'recent_5d_changes': recent_5d_changes,
    }

    logger.debug("技术指标计算完成")

    return indicators


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


def build_analysis_prompt(indicators: Dict[str, Any], fund_code: str, current_date: str) -> str:
    """
    构建分析提示词

    Args:
        indicators: 计算好的技术指标字典
        fund_code: 基金代码
        current_date: 当前日期

    Returns:
        完整的提示词
    """
    # ========== 开头说明 ==========
    intro = f"""你是一位专业的REITs投资分析师，现在需要从价格动量角度分析基金 {fund_code} 在 {current_date} 的投资价值。

背景：这是多agent投资决策系统，你负责从价格走势和技术指标维度分析，其他agent负责基本面、市场面、事件面等。决策agent会综合所有信息做出最终交易策略。

以下是已经计算好的技术指标和数据，请基于这些信息进行分析：
"""

    # ========== 基础信息 ==========
    basic_info = f"""
{'='*60}
【基础信息】
{'='*60}
基金代码：{fund_code}
分析日期：{current_date}
当前收盘价：{indicators['current_price']:.3f}元（前复权价格）
前一交易日收盘价：{indicators['prev_price']:.3f}元
涨跌幅：{indicators['price_change_1d']:+.2f}%
"""

    # ========== 趋势指标 ==========
    trend_section = f"""
{'='*60}
【趋势指标】
{'='*60}
移动平均线：
- MA5: {indicators['ma5']:.3f}元（偏离度: {indicators['ma5_deviation']:+.2f}%）
- MA10: {indicators['ma10']:.3f}元（偏离度: {indicators['ma10_deviation']:+.2f}%）
- MA20: {indicators['ma20']:.3f}元（偏离度: {indicators['ma20_deviation']:+.2f}%）
- MA60: {indicators['ma60']:.3f}元（偏离度: {indicators['ma60_deviation']:+.2f}%）

均线排列状态：{indicators['ma_alignment']}

价格趋势：
- 短期（5日）涨跌幅：{indicators['price_change_5d']:+.2f}%
- 中期（20日）涨跌幅：{indicators['price_change_20d']:+.2f}%
- 长期（60日）涨跌幅：{indicators['price_change_60d']:+.2f}%
"""

    # ========== 动量指标 ==========
    momentum_section = f"""
{'='*60}
【动量指标】
{'='*60}
RSI指标：
- RSI(6): {indicators['rsi6']:.1f}（状态：{indicators['rsi6_status']}）
- RSI(12): {indicators['rsi12']:.1f}
- RSI(24): {indicators['rsi24']:.1f}

MACD指标：
- DIF: {indicators['dif']:.4f}
- DEA: {indicators['dea']:.4f}
- MACD柱: {indicators['macd']:.4f}
- 信号：{indicators['macd_signal']}

动量值(10日)：{indicators['momentum_10d']:.3f}
"""

    # ========== 波动率指标 ==========
    volatility_section = f"""
{'='*60}
【波动率指标】
{'='*60}
布林带（20日，2倍标准差）：
- 上轨: {indicators['bb_upper']:.3f}元
- 中轨: {indicators['bb_middle']:.3f}元
- 下轨: {indicators['bb_lower']:.3f}元
- 当前位置：{indicators['bb_position']}（距中轨{indicators['bb_position_pct']:+.1f}%）

价格波动率（20日）：{indicators['volatility_20d_pct']:.2f}%
ATR（14日）：{indicators['atr_14d']:.3f}

动态波动阈值：{indicators['volatility_threshold']*100:.2f}%
- 计算方法：使用每日动态阈值计算方案，基于历史波动率、自适应乘数和边界约束计算
- 用途：在预测价格趋势是否为整盘的参考阈值，日收益率变动超过±{indicators['volatility_threshold']*100:.2f}%视为显著变动，否则视为横盘震荡
"""

    # ========== 整盘判断专项说明 ==========
    sideways_section = f"""
{'='*60}
【整盘判断专项说明】
{'='*60}

一、动态阈值计算方法
本系统使用每日动态阈值方案判断价格变动显著性（即是否处于整盘状态）：
- 基于历史波动率、自适应乘数和边界约束计算
- 每个交易日独立计算

二、当前动态阈值
当前阈值：{indicators['volatility_threshold']*100:.2f}%
- 预计日收益率绝对值 > {indicators['volatility_threshold']*100:.2f}% → 显著变动（上涨或下跌）
- 预计日收益率绝对值 ≤ {indicators['volatility_threshold']*100:.2f}% → 横盘震荡（整盘）

三、公募REITs波动率特性
- 日涨跌幅通常在1%以内
- 超过2%的变动即为显著变动
- 日最高涨跌幅不超过10%
- 整体波动率远低于股票市场

四、基于动态阈值的历史整盘分布统计
根据动态阈值参数设定，历史回测显示：
- 约33%的交易日处于整盘状态
- 整盘是正常的市场状态，代表价格走势不显著

五、最近5个交易日涨跌幅明细
（用于判断最近是否处于整盘状态）
{format_recent_5d_changes(indicators['recent_5d_changes'], indicators['volatility_threshold'])}

六、本次预测的整盘判断条件

**不同时间维度的整盘判断标准：**

1. **T+1（次日预测）**：
   - 判断标准：预计次日收益率绝对值 ≤ θₜ（当前动态阈值 {indicators['volatility_threshold']*100:.2f}%）
   - 即：|次日收益率| ≤ {indicators['volatility_threshold']*100:.2f}% → 判断为整盘

2. **T+5（未来5个交易日）**：
   - 阈值扩展系数：ε₅ = √5 · θₜ = {(indicators['volatility_threshold'] * (5**0.5))*100:.2f}%
   - 判断标准：预计未来5个交易日的累计波动幅度 ≤ ε₅
   - 或等价于：未来5日的平均日收益率绝对值 ≤ θₜ
   - 解释：基于价格随机游走假设，多日波动率按时间的平方根缩放

3. **T+20（未来20个交易日）**：
   - 阈值扩展系数：ε₂₀ = √20 · θₜ = {(indicators['volatility_threshold'] * (20**0.5))*100:.2f}%
   - 判断标准：预计未来20个交易日的累计波动幅度 ≤ ε₂₀
   - 或等价于：未来20日的平均日收益率绝对值 ≤ θₜ
   - 解释：同样基于波动率的时间平方根缩放规律

**重要说明：**
- 整盘不等于"不确定"，是正常的市场状态，代表价格走势不显著
- 当判断为整盘时，应给予合理的主导概率和合理的置信度
- 整盘的判断应基于技术指标的综合分析
"""

    # ========== 成交量分析 ==========
    volume_section = f"""
{'='*60}
【成交量分析】
{'='*60}
成交量：
- 当日成交量：{int(indicators['current_volume'])}手
- 5日均量：{int(indicators['vol_ma5'])}手（量比：{indicators['volume_ratio']:.2f}）
- 10日均量：{int(indicators['vol_ma10'])}手
- 20日均量：{int(indicators['vol_ma20'])}手

量价关系：{indicators['volume_price_relationship']}
"""

    # ========== 支撑压力位 ==========
    # 智能选择支撑位和压力位
    support_levels = []
    resistance_levels = []

    # 支撑位候选（按强度排序）
    if indicators['ma20'] and indicators['ma20'] < indicators['current_price']:
        support_levels.append((indicators['ma20'], 'MA20'))
    if indicators['low_20d'] and indicators['low_20d'] < indicators['current_price']:
        support_levels.append((indicators['low_20d'], '20日低点'))
    if indicators['bb_lower'] and indicators['bb_lower'] < indicators['current_price']:
        support_levels.append((indicators['bb_lower'], '布林带下轨'))
    if indicators['ma60'] and indicators['ma60'] < indicators['current_price']:
        support_levels.append((indicators['ma60'], 'MA60'))

    # 压力位候选
    if indicators['high_20d'] and indicators['high_20d'] > indicators['current_price']:
        resistance_levels.append((indicators['high_20d'], '20日高点'))
    if indicators['ma60'] and indicators['ma60'] > indicators['current_price']:
        resistance_levels.append((indicators['ma60'], 'MA60'))
    if indicators['bb_upper'] and indicators['bb_upper'] > indicators['current_price']:
        resistance_levels.append((indicators['bb_upper'], '布林带上轨'))

    # 排序（支撑位从高到低，压力位从低到高）
    support_levels.sort(reverse=True)
    resistance_levels.sort()

    # 选择最重要的几个位置
    support_text = ""
    for i, (price, desc) in enumerate(support_levels[:3], 1):
        support_text += f"- 支撑位{i}：{price:.3f}元（{desc}）\n"

    if not support_text:
        support_text = "- 暂无明显支撑位\n"

    resistance_text = ""
    for i, (price, desc) in enumerate(resistance_levels[:3], 1):
        resistance_text += f"- 压力位{i}：{price:.3f}元（{desc}）\n"

    if not resistance_text:
        resistance_text = "- 暂无明显压力位\n"

    support_resistance_section = f"""
{'='*60}
【支撑压力位】
{'='*60}
主要支撑位：
{support_text}主要压力位：
{resistance_text}"""

    # ========== 价格形态 ==========
    pattern_section = f"""
{'='*60}
【价格形态】
{'='*60}
连续走势：连续{indicators['consecutive_direction']}{indicators['consecutive_days']}天
近20日涨跌统计：上涨{indicators['up_days_20d']}天，下跌{indicators['down_days_20d']}天
近20日平均振幅：{indicators['avg_amplitude_20d']:.2f}%
近20日最大单日振幅：{indicators['max_amplitude_20d']:.2f}%
"""

    # ========== 分析任务要求 ==========
    task_section = f"""
{'='*60}
【分析任务要求】
{'='*60}

你必须**严格按照以下JSON格式**输出分析结果。输出内容必须是**纯JSON格式**。

### 输出格式示例：

{{{{
  "expert_type": "price",
  "analysis_date": "{current_date}",
  "fund_code": "{fund_code}",

  "direction_judgment": {{{{
    "short_term": {{{{
      "T+1": {{{{
        "direction_probs": {{{{
          "up": 0.00,
          "down": 0.00,
          "side": 0.00
        }}}},
        "confidence": 0.00,
        "rationale": "核心理由(1句话，50字以内)"
      }}}},
      "T+5": {{{{
        "direction_probs": {{{{
          "up": 0.00,
          "down": 0.00,
          "side": 0.00
        }}}},
        "confidence": 0.00,
        "rationale": "核心理由(1句话，50字以内)"
      }}}}
    }}}},
    "T+20": {{{{
      "direction_probs": {{{{
        "up": 0.00,
        "down": 0.00,
        "side": 0.00
      }}}},
      "confidence": 0.00,
      "rationale": "核心理由(1句话，50字以内)"
    }}}}
  }}}},

  "detailed_analysis": {{{{
    "trend_analysis": {{{{
      "ma_alignment": "多头排列/空头排列/混乱",
      "current_trend": "上涨/下跌/横盘",
      "trend_strength": "强/中/弱",
      "key_support": 2.50,
      "key_resistance": 2.80
    }}}},
    "momentum_analysis": {{{{
      "rsi_status": "超买/正常/超卖",
      "rsi_value": 65.5,
      "macd_signal": "金叉/死叉/无明显信号",
      "momentum_direction": "增强/减弱/平稳"
    }}}},
    "volume_analysis": {{{{
      "volume_ratio": 1.20,
      "volume_price_relation": "价涨量增/价涨量缩/价跌量增/价跌量缩/价量平稳",
      "volume_signal": "健康/异常"
    }}}},
    "volatility_analysis": {{{{
      "bb_position": "突破上轨/偏上/居中/偏下/跌破下轨",
      "volatility_level": "高/中/低"
    }}}},
    "technical_summary": "综合上述技术指标，分析当前价格动量状态和可能走势（200字以内，不使用换行符）",
    "risk_alerts": ["风险点1", "风险点2"]
  }}}}
}}}}

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

#### 短期vs中长期的概率差异（技术面特有）
- **T+1**: 极短期受技术形态、量价关系、超买超卖影响最大
- **T+5**: 短期末，技术信号延续性，但需警惕反转
- **T+20**: 更多受趋势方向、均线系统、长期动量影响
- 在公募REITs低波动背景下，小幅技术信号也具有参考意义

#### confidence（置信度）
反映技术信号的清晰程度和可靠性（0.00-1.00），按信号强度限定：

**强信号**（多项技术指标共振）：
- 置信度：≥ 0.70
- 情况：金叉+多头排列+放量上涨+RSI正常区
- 或：死叉+空头排列+放量下跌+RSI正常区
- 或：多项指标确认整盘（价格在布林带中轨附近+MA混乱+量能萎缩+RSI中性）

**中等信号**（主要指标明确但有矛盾）：
- 置信度：0.60 - 0.65
- 情况：金叉但缩量，或多头排列但RSI超买，或个别指标矛盾

**弱信号**（信号模糊）：
- 置信度：0.50 - 0.55
- 情况：均线混乱，MACD无明显信号，量价背离

**极弱信号**（矛盾严重）：
- 置信度：≤ 0.45
- 情况：多项指标相互矛盾，处于关键变盘期

**重要**：
- 在公募REITs低波动背景下，技术指标的微小变化也具有指导意义

#### rationale（理由）
用1句话（50字以内）说明核心理由，需引用具体技术指标

---

### 2. detailed_analysis（详细分析）

#### trend_analysis（趋势分析）
- **ma_alignment**: 均线排列状态（多头排列/空头排列/混乱）
- **current_trend**: 当前趋势（上涨/下跌/横盘）
- **trend_strength**: 趋势强度（强/中/弱）
- **key_support**: 最关键的支撑位价格（数值）
- **key_resistance**: 最关键的压力位价格（数值）

#### momentum_analysis（动量分析）
- **rsi_status**: RSI状态（超买/正常/超卖）
- **rsi_value**: RSI(6)当前数值
- **macd_signal**: MACD信号（金叉/死叉/无明显信号）
- **momentum_direction**: 动量方向（增强/减弱/平稳）

#### volume_analysis（成交量分析）
- **volume_ratio**: 量比数值（当前成交量/5日均量）
- **volume_price_relation**: 量价关系（价涨量增/价涨量缩/价跌量增/价跌量缩/价量平稳）
- **volume_signal**: 成交量信号（健康/异常）

#### volatility_analysis（波动率分析）
- **bb_position**: 布林带位置（突破上轨/偏上/居中/偏下/跌破下轨）
- **volatility_level**: 波动率水平（高/中/低）

#### technical_summary（技术总结）
综合上述技术指标，分析当前价格动量状态和可能走势（200字以内）

**重要**：
- 使用连贯的段落形式，**不要使用换行符**
- 可以使用逗号、句号等标点符号组织内容
- 基于具体技术指标数值进行分析
- 说明各项指标之间的相互印证或矛盾
- 给出可能的走势判断

#### risk_alerts（风险提示）
数组格式，列出1-3个当前最需要关注的技术面风险点，每个风险点用一句话表述

---

## 重要提醒

1. **必须输出有效的JSON格式**，确保所有括号、引号、逗号正确
2. **概率之和必须=1.0**，系统会自动检查
3. **技术指标与概率一致**：概率判断应与技术指标状态一致
4. **使用具体数据**：rationale中必须引用具体技术指标
5. **控制字数**：遵从字数要求
6. **不使用换行符**：technical_summary必须是连贯的段落
7. **客观分析**：技术分析要客观，不要过度乐观或悲观
8. **多指标共振**：关注多项技术指标的相互印证
9. **置信度合理**：有矛盾信号时应降低置信度，但不要系统性偏低
10. **本分析仅从价格动量维度评估**，实际投资决策需结合其他维度
11. **避免过度保守，敢于判断**：根据各维度信号强度给出相应的概率和置信度，不要因为害怕错误而系统性降低数值。请记住：
   **市场特性要求更果断的判断**：
   - 公募REITs波动率远低于股票
   - 整盘（side）是公募REITs常见且正常的状态，判断为整盘时也应给予合理的主导概率和置信度

   **投资者的策略依赖你的预测质量**：
   - 策略目标：在回撤可控的前提下最大化收益
   - 你的角色：提供清晰、有置信度的方向预测，作为交易决策的关键输入
   - "模糊安全"倾向（总是选择中间值或较低值），会显著降低策略的收益潜力
   - 因此在有明确信号（无论上涨、下跌还是整盘）时请给予较高的概率和置信度

   **注意**：这并不意味着盲目乐观，而是要求你基于充分的证据做出匹配的判断强度
12. **检查概率与置信度**：返回前检查 `direction_probs` 和 `confidence` 是否符合字段填写说明的要求，如果发现不符合要求，请根据分析重新打分

请开始你的分析，输出JSON格式结果。
"""

    # ========== 组合完整提示词 ==========
    prompt = (intro + basic_info + trend_section + momentum_section +
              volatility_section + sideways_section + volume_section +
              support_resistance_section + pattern_section + task_section)

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


def analyze_price(fund_code: str, date: str, min_dominant_prob: float = 0.5) -> Dict[str, Any]:
    """
    价格动量分析主函数（可被其他脚本调用）

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
    logger.info(f"开始执行价格动量分析 - 基金: {fund_code}, 日期: {date}")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # 第1步：获取价格数据
        logger.info(f"步骤1: 获取历史价格数据（往前{HISTORY_DAYS}天）")
        price_df = get_price_data(fund_code, date, HISTORY_DAYS)

        if len(price_df) == 0:
            raise ValueError(f"未找到基金 {fund_code} 的价格数据")

        logger.info(f"✓ 查询到 {len(price_df)} 条价格数据，范围：{price_df['trade_date'].min()} 至 {price_df['trade_date'].max()}")

        # 第2步：获取分红数据
        logger.info("步骤2: 获取分红数据")
        start_date = price_df['trade_date'].min()
        dividend_df = get_dividend_data(fund_code, str(start_date), date)
        logger.info(f"✓ 查询到 {len(dividend_df)} 条分红记录")

        # 第3步：计算复权价格
        logger.info("步骤3: 计算前复权价格")
        adjusted_df = calculate_adjusted_price(price_df, dividend_df)
        logger.info(f"✓ 复权处理完成")

        # 第4步：计算技术指标
        logger.info("步骤4: 计算技术指标")
        indicators = calculate_technical_indicators(adjusted_df)
        logger.info("✓ 技术指标计算完成")

        # 记录关键指标
        if logger.level == logging.DEBUG:
            logger.debug("\n" + "=" * 80)
            logger.debug("关键技术指标:")
            logger.debug("=" * 80)
            logger.debug(f"当前价格: {indicators['current_price']:.3f}")
            logger.debug(f"MA5: {indicators['ma5']:.3f}, MA20: {indicators['ma20']:.3f}, MA60: {indicators['ma60']:.3f}")
            logger.debug(f"RSI(6): {indicators['rsi6']:.1f} ({indicators['rsi6_status']})")
            logger.debug(f"MACD信号: {indicators['macd_signal']}")
            logger.debug(f"量价关系: {indicators['volume_price_relationship']}")
            logger.debug("=" * 80)

        # 第5步：构建提示词
        logger.info("步骤5: 构建分析提示词")
        prompt = build_analysis_prompt(indicators, fund_code, date)

        if logger.level == logging.DEBUG:
            logger.debug("\n" + "=" * 80)
            logger.debug("完整提示词内容:")
            logger.debug("=" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)

        # 第6步：调用LLM分析
        logger.info("步骤6: 调用 deepseek-reasoner 模型进行分析")
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

        logger.info(f"\n✓ 价格动量分析完成，总耗时: {elapsed:.2f} 秒")
        logger.info("=" * 80)

        # 构建返回结果
        return_data = {
            'status': 'success',
            'analysis_result': result['content'],
            'reasoning_process': result['reasoning_content'],
            'metadata': {
                'expert_type': 'price',
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
        logger.error(f"价格动量分析执行失败: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'metadata': {
                'expert_type': 'price',
                'fund_code': fund_code,
                'analysis_date': date,
                'timestamp': datetime.now().isoformat()
            }
        }
