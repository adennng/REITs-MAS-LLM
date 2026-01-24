"""
指标计算模块
计算所有市场分析所需的指标
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class IndicatorsCalculator:
    """指标计算类"""

    def __init__(self):
        """初始化"""
        pass

    @staticmethod
    def _safe_divide(a, b, default=0):
        """安全除法，避免除零错误"""
        try:
            if b == 0 or pd.isna(b):
                return default
            return a / b
        except:
            return default

    @staticmethod
    def _calculate_percentile(series, value):
        """计算分位数"""
        if len(series) == 0 or pd.isna(value):
            return None
        return stats.percentileofscore(series, value, kind='rank')

    @staticmethod
    def _calculate_change(df, periods, price_col='close_price'):
        """计算涨跌幅"""
        if len(df) < periods + 1:
            return None
        try:
            current = df.iloc[-1][price_col]
            previous = df.iloc[-(periods + 1)][price_col]
            return (current - previous) / previous * 100
        except:
            return None

    @staticmethod
    def _calculate_ma(df, periods, price_col='close_price'):
        """计算移动平均"""
        if len(df) < periods:
            return None
        return df[price_col].rolling(window=periods).mean().iloc[-1]

    @staticmethod
    def _calculate_rsi(df, periods=14, price_col='close_price'):
        """计算RSI"""
        if len(df) < periods + 1:
            return None

        try:
            delta = df[price_col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return None

    @staticmethod
    def _calculate_volatility(df, periods, price_col='close_price'):
        """计算波动率（年化）"""
        if len(df) < periods + 1:
            return None

        try:
            returns = df[price_col].pct_change().dropna()
            if len(returns) < periods:
                return None
            vol = returns.tail(periods).std() * np.sqrt(252)
            return vol * 100  # 转为百分比
        except:
            return None

    @staticmethod
    def _calculate_macd(df, price_col='close_price'):
        """计算MACD"""
        if len(df) < 26:
            return None, None, None

        try:
            ema12 = df[price_col].ewm(span=12, adjust=False).mean()
            ema26 = df[price_col].ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd = (dif - dea) * 2

            return dif.iloc[-1], dea.iloc[-1], macd.iloc[-1]
        except:
            return None, None, None

    @staticmethod
    def _count_consecutive_days(df, price_col='close_price'):
        """计算连续上涨/下跌天数"""
        if len(df) < 2:
            return 0

        try:
            changes = df[price_col].diff().iloc[-20:]  # 只看最近20天
            count = 0

            for i in range(len(changes) - 1, -1, -1):
                if pd.isna(changes.iloc[i]):
                    break
                if count == 0:
                    if changes.iloc[i] > 0:
                        count = 1
                    elif changes.iloc[i] < 0:
                        count = -1
                    else:
                        continue
                else:
                    if (count > 0 and changes.iloc[i] > 0) or (count < 0 and changes.iloc[i] < 0):
                        count = count + 1 if count > 0 else count - 1
                    else:
                        break

            return count
        except:
            return 0

    def calculate_reits_indicators(self, df):
        """
        计算REITs市场指标

        Args:
            df: DataFrame with columns: trade_date, close_price

        Returns:
            dict of indicators
        """
        logger.info("开始计算REITs市场指标")

        if df is None or len(df) == 0:
            logger.error("REITs数据为空")
            return {}

        indicators = {}

        try:
            current_price = df.iloc[-1]['close_price']
            all_prices = df['close_price']

            # A. 价格位置指标
            indicators['percentile_full'] = self._calculate_percentile(all_prices, current_price)
            indicators['percentile_1y'] = self._calculate_percentile(all_prices.tail(252), current_price)
            indicators['percentile_6m'] = self._calculate_percentile(all_prices.tail(126), current_price)

            indicators['drawdown_from_high'] = (current_price - all_prices.max()) / all_prices.max() * 100
            indicators['rally_from_low'] = (current_price - all_prices.min()) / all_prices.min() * 100

            # 距离近期高低点
            if len(df) >= 20:
                high_20d = df.tail(20)['close_price'].max()
                low_20d = df.tail(20)['close_price'].min()
                indicators['distance_to_20d_high'] = (high_20d - current_price) / current_price * 100
                indicators['distance_to_20d_low'] = (current_price - low_20d) / current_price * 100

            if len(df) >= 60:
                high_60d = df.tail(60)['close_price'].max()
                low_60d = df.tail(60)['close_price'].min()
                indicators['distance_to_60d_high'] = (high_60d - current_price) / current_price * 100
                indicators['distance_to_60d_low'] = (current_price - low_60d) / current_price * 100

            if len(df) >= 250:
                high_250d = df.tail(250)['close_price'].max()
                indicators['distance_to_250d_high'] = (high_250d - current_price) / current_price * 100

            # B. 动量指标
            for period in [1, 3, 5, 10, 20, 60]:
                indicators[f'change_{period}d'] = self._calculate_change(df, period)

            # 上涨占比
            if len(df) >= 5:
                changes_5d = df.tail(5)['close_price'].diff()
                indicators['up_ratio_5d'] = (changes_5d > 0).sum() / len(changes_5d)

            if len(df) >= 10:
                changes_10d = df.tail(10)['close_price'].diff()
                indicators['up_ratio_10d'] = (changes_10d > 0).sum() / len(changes_10d)

            if len(df) >= 20:
                changes_20d = df.tail(20)['close_price'].diff()
                indicators['up_ratio_20d'] = (changes_20d > 0).sum() / len(changes_20d)

            indicators['consecutive_days'] = self._count_consecutive_days(df)

            # C. 波动率指标
            indicators['volatility_20d'] = self._calculate_volatility(df, 20)
            indicators['volatility_60d'] = self._calculate_volatility(df, 60)
            if indicators['volatility_20d'] is not None:
                all_vol_20 = [self._calculate_volatility(df.iloc[:i], 20) for i in range(20, len(df))]
                all_vol_20 = [v for v in all_vol_20 if v is not None]
                if all_vol_20:
                    indicators['volatility_percentile'] = self._calculate_percentile(all_vol_20, indicators['volatility_20d'])

            # D. 趋势指标
            indicators['ma5'] = self._calculate_ma(df, 5)
            indicators['ma10'] = self._calculate_ma(df, 10)
            indicators['ma20'] = self._calculate_ma(df, 20)
            indicators['ma60'] = self._calculate_ma(df, 60)

            # 价格相对均线
            for ma in [5, 10, 20, 60]:
                ma_value = indicators.get(f'ma{ma}')
                if ma_value:
                    indicators[f'price_vs_ma{ma}'] = (current_price - ma_value) / ma_value * 100

            # 均线排列
            ma5, ma10, ma20, ma60 = indicators.get('ma5'), indicators.get('ma10'), indicators.get('ma20'), indicators.get('ma60')
            if all([ma5, ma10, ma20, ma60]):
                indicators['bullish_alignment'] = ma5 > ma10 > ma20 > ma60
                indicators['bearish_alignment'] = ma5 < ma10 < ma20 < ma60
            else:
                indicators['bullish_alignment'] = False
                indicators['bearish_alignment'] = False

            # MACD
            dif, dea, macd_bar = self._calculate_macd(df)
            indicators['macd_dif'] = dif
            indicators['macd_dea'] = dea
            indicators['macd_bar'] = macd_bar

            # RSI
            indicators['rsi_14'] = self._calculate_rsi(df, 14)

            # E. 历史对比
            if len(df) >= 252:
                price_1y_ago = df.iloc[-252]['close_price']
                indicators['price_1y_ago'] = price_1y_ago
                indicators['yoy_change'] = (current_price - price_1y_ago) / price_1y_ago * 100

            # 年初至今
            current_year = df.iloc[-1]['trade_date'].year
            year_start_data = df[df['trade_date'].apply(lambda x: x.year) == current_year]
            if len(year_start_data) > 0:
                year_start_price = year_start_data.iloc[0]['close_price']
                indicators['ytd_change'] = (current_price - year_start_price) / year_start_price * 100

            # 低于当前价格的占比
            indicators['below_current_price_ratio'] = (all_prices < current_price).sum() / len(all_prices) * 100

            logger.info(f"REITs指标计算完成，共{len(indicators)}个指标")

        except Exception as e:
            logger.error(f"REITs指标计算失败: {e}", exc_info=True)

        return indicators

    def calculate_interest_rate_indicators(self, df, reits_df=None):
        """
        计算利率环境指标

        Args:
            df: DataFrame with columns: trade_date, rate
            reits_df: REITs数据，用于计算相关性

        Returns:
            dict of indicators
        """
        logger.info("开始计算利率指标")

        if df is None or len(df) == 0:
            logger.error("利率数据为空")
            return {}

        indicators = {}

        try:
            current_rate = df.iloc[-1]['rate']
            all_rates = df['rate']

            # A. 利率水平
            indicators['current_rate'] = float(current_rate)
            indicators['rate_percentile_full'] = self._calculate_percentile(all_rates, current_rate)
            indicators['rate_percentile_1y'] = self._calculate_percentile(all_rates.tail(252), current_rate)

            indicators['distance_to_1.8'] = (current_rate - 1.8) * 100  # bp
            if len(df) >= 252:
                avg_1y = all_rates.tail(252).mean()
                indicators['distance_to_1y_avg'] = (current_rate - avg_1y) * 100  # bp

            # B. 利率趋势
            for period in [1, 3, 5, 10, 20, 60]:
                change = self._calculate_change(df, period, 'rate')
                if change is not None:
                    # 利率变化用bp表示
                    indicators[f'rate_change_{period}d'] = change / 100 * current_rate * 100  # 转为bp

            # 趋势判断
            change_20d = indicators.get('rate_change_20d', 0)
            if change_20d < -20:
                indicators['rate_trend'] = "明确下行"
            elif -20 <= change_20d < -5:
                indicators['rate_trend'] = "缓慢下行"
            elif -5 <= change_20d <= 5:
                indicators['rate_trend'] = "横盘震荡"
            elif 5 < change_20d <= 20:
                indicators['rate_trend'] = "缓慢上行"
            else:
                indicators['rate_trend'] = "明确上行"

            # 利率移动平均
            indicators['rate_ma5'] = self._calculate_ma(df, 5, 'rate')
            indicators['rate_ma20'] = self._calculate_ma(df, 20, 'rate')
            indicators['rate_ma60'] = self._calculate_ma(df, 60, 'rate')

            if indicators.get('rate_ma20'):
                indicators['rate_vs_ma20'] = (current_rate - indicators['rate_ma20']) * 100  # bp

            # 下降天数占比
            if len(df) >= 20:
                rate_changes = df.tail(20)['rate'].diff()
                indicators['rate_down_ratio_20d'] = (rate_changes < 0).sum() / 20

            # C. 相关性（如果有REITs数据）
            if reits_df is not None and len(reits_df) > 0:
                # 合并数据
                merged = pd.merge(df, reits_df, on='trade_date', how='inner')

                if len(merged) >= 60:
                    rate_changes_60 = merged.tail(60)['rate'].pct_change()
                    reits_changes_60 = merged.tail(60)['close_price'].pct_change()
                    indicators['corr_rate_reits_60d'] = rate_changes_60.corr(reits_changes_60)

                if len(merged) >= 20:
                    rate_changes_20 = merged.tail(20)['rate'].pct_change()
                    reits_changes_20 = merged.tail(20)['close_price'].pct_change()
                    indicators['corr_rate_reits_20d'] = rate_changes_20.corr(reits_changes_20)

            logger.info(f"利率指标计算完成，共{len(indicators)}个指标")

        except Exception as e:
            logger.error(f"利率指标计算失败: {e}", exc_info=True)

        return indicators

    def calculate_stock_indicators(self, df, reits_df=None):
        """
        计算股市环境指标

        Args:
            df: DataFrame with columns: trade_date, sh_index, dividend_index
            reits_df: REITs数据，用于对比

        Returns:
            dict of indicators
        """
        logger.info("开始计算股市指标")

        if df is None or len(df) == 0:
            logger.error("股市数据为空")
            return {}

        indicators = {}

        try:
            # 上证指数
            sh_current = df.iloc[-1]['sh_index']
            sh_all = df['sh_index']

            indicators['sh_index'] = float(sh_current)
            indicators['sh_percentile_full'] = self._calculate_percentile(sh_all, sh_current)
            indicators['sh_percentile_1y'] = self._calculate_percentile(sh_all.tail(252), sh_current)

            for period in [1, 5, 10, 20, 60]:
                indicators[f'sh_change_{period}d'] = self._calculate_change(df, period, 'sh_index')

            indicators['sh_rsi_14'] = self._calculate_rsi(df, 14, 'sh_index')

            # 趋势判断 - 严格按照提示词定义的5个标准
            change_20d = indicators.get('sh_change_20d', 0) or 0
            rsi = indicators.get('sh_rsi_14', 50) or 50

            # 首先检查是否完全符合标准定义
            if change_20d > 5 and rsi > 60:
                # 牛市标准：涨幅>5% 且 RSI>60
                indicators['sh_trend'] = "牛市"
            elif 0 < change_20d <= 5 and 50 < rsi <= 60:
                # 震荡偏强标准：涨幅0-5% 且 RSI 50-60
                indicators['sh_trend'] = "震荡偏强"
            elif -2 < change_20d <= 2 and 40 < rsi <= 60:
                # 震荡标准：涨幅-2到2% 且 RSI 40-60
                indicators['sh_trend'] = "震荡"
            elif -5 < change_20d <= 0 and 40 < rsi <= 50:
                # 震荡偏弱标准：涨幅-5到0% 且 RSI 40-50
                indicators['sh_trend'] = "震荡偏弱"
            elif change_20d < -5 and rsi < 40:
                # 熊市标准：涨幅<-5% 且 RSI<40
                indicators['sh_trend'] = "熊市"

            # 处理不完全符合标准的边界情况（按最接近原则分类，严格区分涨跌）
            else:
                # 大涨（>5%）但动能不足
                if change_20d > 5:
                    if rsi > 50:
                        indicators['sh_trend'] = "震荡偏强"  # 接近牛市
                    elif rsi > 40:
                        indicators['sh_trend'] = "震荡"
                    else:
                        indicators['sh_trend'] = "震荡偏弱"  # 可能反转

                # 小涨（0-5%）但RSI不在标准区间
                elif 0 < change_20d <= 5:
                    if rsi > 60:
                        indicators['sh_trend'] = "震荡偏强"  # 动能强，接近牛市
                    elif rsi > 40:
                        indicators['sh_trend'] = "震荡"
                    else:
                        indicators['sh_trend'] = "震荡偏弱"

                # 小幅波动（-2到0%，轻微下跌）但RSI偏离标准区间
                elif -2 < change_20d <= 0:
                    if rsi > 60:
                        indicators['sh_trend'] = "震荡"  # 超买后小跌，属于震荡
                    elif rsi > 50:
                        indicators['sh_trend'] = "震荡"
                    else:
                        indicators['sh_trend'] = "震荡偏弱"

                # 横盘微涨（0-2%）但RSI偏离标准区间
                elif 0 < change_20d <= 2:
                    if rsi > 60:
                        indicators['sh_trend'] = "震荡偏强"
                    else:
                        indicators['sh_trend'] = "震荡"

                # 小跌（-5到-2%）但RSI不在标准区间
                elif -5 < change_20d <= -2:
                    if rsi > 60:
                        indicators['sh_trend'] = "震荡"  # 可能反转
                    elif rsi > 50:
                        indicators['sh_trend'] = "震荡偏弱"
                    elif rsi > 40:
                        indicators['sh_trend'] = "震荡偏弱"
                    else:
                        indicators['sh_trend'] = "熊市"

                # 大跌（<-5%）但动能未完全崩溃
                else:  # change_20d < -5
                    if rsi >= 50:
                        indicators['sh_trend'] = "震荡偏弱"  # 可能反弹
                    elif rsi >= 40:
                        indicators['sh_trend'] = "震荡偏弱"
                    else:
                        indicators['sh_trend'] = "熊市"

            # 距离关键整数位
            key_levels = [3000, 3300, 3500, 3700, 4000]
            distances = {level: abs(sh_current - level) for level in key_levels}
            nearest_level = min(distances, key=distances.get)
            indicators['sh_nearest_integer'] = nearest_level
            indicators['sh_distance_to_integer'] = (sh_current - nearest_level) / nearest_level * 100

            # 红利指数
            div_current = df.iloc[-1]['dividend_index']
            div_all = df['dividend_index']

            indicators['dividend_index'] = float(div_current)
            indicators['dividend_percentile_full'] = self._calculate_percentile(div_all, div_current)
            indicators['dividend_percentile_1y'] = self._calculate_percentile(div_all.tail(252), div_current)

            for period in [1, 5, 20, 60]:
                indicators[f'dividend_change_{period}d'] = self._calculate_change(df, period, 'dividend_index')

            indicators['dividend_consecutive_up'] = self._count_consecutive_days(df, 'dividend_index')

            # 红利vs上证
            sh_change_20 = indicators.get('sh_change_20d', 0) or 0
            div_change_20 = indicators.get('dividend_change_20d', 0) or 0
            indicators['dividend_vs_sh_20d'] = div_change_20 - sh_change_20

            # 对比REITs
            if reits_df is not None:
                merged = pd.merge(df, reits_df, on='trade_date', how='inner')

                if len(merged) >= 20:
                    reits_change_20 = self._calculate_change(merged, 20, 'close_price')
                    if reits_change_20 is not None:
                        indicators['reits_vs_sh_20d'] = reits_change_20 - sh_change_20
                        indicators['reits_vs_dividend_20d'] = reits_change_20 - div_change_20

                # 跷跷板效应
                if len(merged) >= 5:
                    div_change_5 = self._calculate_change(merged.tail(5), 5, 'dividend_index')
                    reits_change_5 = self._calculate_change(merged.tail(5), 5, 'close_price')

                    if div_change_5 and reits_change_5:
                        indicators['seesaw_strong'] = div_change_5 > 2 and reits_change_5 < -1
                        indicators['seesaw_weak'] = div_change_5 < -2 and reits_change_5 > 1
                    else:
                        indicators['seesaw_strong'] = False
                        indicators['seesaw_weak'] = False

                # 相关性
                if len(merged) >= 60:
                    div_changes_60 = merged.tail(60)['dividend_index'].pct_change()
                    reits_changes_60 = merged.tail(60)['close_price'].pct_change()
                    indicators['corr_dividend_reits_60d'] = div_changes_60.corr(reits_changes_60)

                if len(merged) >= 20:
                    div_changes_20 = merged.tail(20)['dividend_index'].pct_change()
                    reits_changes_20 = merged.tail(20)['close_price'].pct_change()
                    indicators['corr_dividend_reits_20d'] = div_changes_20.corr(reits_changes_20)

            # 风险偏好
            indicators['sh_volatility_20d'] = self._calculate_volatility(df, 20, 'sh_index')

            logger.info(f"股市指标计算完成，共{len(indicators)}个指标")

        except Exception as e:
            logger.error(f"股市指标计算失败: {e}", exc_info=True)

        return indicators

    def calculate_sentiment_indicators(self, turnover_df):
        """
        计算市场情绪指标

        Args:
            turnover_df: DataFrame with columns: trade_date, turnover_amount, total_market_value

        Returns:
            dict of indicators
        """
        logger.info("开始计算市场情绪指标")

        if turnover_df is None or len(turnover_df) == 0:
            logger.error("成交数据为空")
            return {}

        indicators = {}

        try:
            # 计算换手率
            turnover_df = turnover_df.copy()
            turnover_df['turnover_rate'] = turnover_df['turnover_amount'] / turnover_df['total_market_value'] * 100

            current = turnover_df.iloc[-1]
            indicators['turnover_rate'] = float(current['turnover_rate'])
            indicators['turnover_amount'] = float(current['turnover_amount'])
            indicators['total_market_value'] = float(current['total_market_value'])

            # 换手率分位数
            all_rates = turnover_df['turnover_rate'].dropna()
            indicators['turnover_rate_percentile'] = self._calculate_percentile(all_rates, indicators['turnover_rate'])

            # 换手率水平
            percentile = indicators.get('turnover_rate_percentile', 50)
            if percentile < 10:
                indicators['turnover_rate_level'] = "极低"
            elif percentile < 30:
                indicators['turnover_rate_level'] = "偏低"
            elif percentile < 70:
                indicators['turnover_rate_level'] = "正常"
            elif percentile < 90:
                indicators['turnover_rate_level'] = "偏高"
            else:
                indicators['turnover_rate_level'] = "极高"

            # 换手率均值
            if len(turnover_df) >= 5:
                indicators['turnover_rate_ma5'] = turnover_df.tail(5)['turnover_rate'].mean()
            if len(turnover_df) >= 20:
                indicators['turnover_rate_ma20'] = turnover_df.tail(20)['turnover_rate'].mean()
                if indicators.get('turnover_rate_ma20'):
                    indicators['turnover_rate_vs_ma20'] = (indicators['turnover_rate'] - indicators['turnover_rate_ma20']) / indicators['turnover_rate_ma20'] * 100

            # 相对最低换手率
            min_rate = all_rates.min()
            if min_rate > 0:
                indicators['turnover_rate_vs_lowest'] = (indicators['turnover_rate'] - min_rate) / min_rate * 100

            # 连续缩量/放量
            if len(turnover_df) >= 2:
                rate_changes = turnover_df.tail(20)['turnover_rate'].diff()
                count = 0
                for i in range(len(rate_changes) - 1, -1, -1):
                    if pd.isna(rate_changes.iloc[i]):
                        break
                    if count == 0:
                        if rate_changes.iloc[i] < 0:
                            count = -1
                        elif rate_changes.iloc[i] > 0:
                            count = 1
                    else:
                        if (count < 0 and rate_changes.iloc[i] < 0) or (count > 0 and rate_changes.iloc[i] > 0):
                            count = count - 1 if count < 0 else count + 1
                        else:
                            break
                indicators['consecutive_volume_change'] = count

            # 成交额变化
            if len(turnover_df) >= 2:
                yesterday = turnover_df.iloc[-2]['turnover_amount']
                indicators['turnover_vs_yesterday'] = (indicators['turnover_amount'] - yesterday) / yesterday * 100

            if len(turnover_df) >= 5:
                last_week = turnover_df.iloc[-5]['turnover_amount']
                indicators['turnover_vs_last_week'] = (indicators['turnover_amount'] - last_week) / last_week * 100

            # 成交额均值
            if len(turnover_df) >= 5:
                indicators['turnover_amount_ma5'] = turnover_df.tail(5)['turnover_amount'].mean()
            if len(turnover_df) >= 20:
                indicators['turnover_amount_ma20'] = turnover_df.tail(20)['turnover_amount'].mean()

            logger.info(f"情绪指标计算完成，共{len(indicators)}个指标")

        except Exception as e:
            logger.error(f"情绪指标计算失败: {e}", exc_info=True)

        return indicators

    def calculate_sector_indicators(self, sector_df, reits_df, sector_name, sectors_ranking):
        """
        计算板块指标

        Args:
            sector_df: 板块指数数据
            reits_df: REITs全指数据
            sector_name: 板块名称
            sectors_ranking: 所有板块排名数据

        Returns:
            dict of indicators
        """
        logger.info(f"开始计算{sector_name}板块指标")

        if sector_df is None or len(sector_df) == 0:
            logger.warning("板块数据为空")
            return {}

        indicators = {}

        try:
            # 板块涨跌幅
            for period in [1, 5, 20, 60]:
                indicators[f'sector_change_{period}d'] = self._calculate_change(sector_df, period, 'index_value')

            # 相对表现
            if reits_df is not None:
                sector_change_20 = indicators.get('sector_change_20d')
                reits_change_20 = self._calculate_change(reits_df, 20, 'close_price')

                if sector_change_20 is not None and reits_change_20 is not None:
                    indicators['sector_vs_market_20d'] = sector_change_20 - reits_change_20

            # 板块排名
            if sectors_ranking:
                sector_rank_info = next((s for s in sectors_ranking if s['sector'] == sector_name), None)
                if sector_rank_info:
                    indicators['sector_rank'] = sector_rank_info['rank']
                    indicators['sector_total_count'] = len(sectors_ranking)
                    indicators['sector_strong_signal'] = sector_rank_info['rank'] <= 3
                    indicators['sector_weak_signal'] = sector_rank_info['rank'] >= len(sectors_ranking) - 2

            logger.info(f"板块指标计算完成，共{len(indicators)}个指标")

        except Exception as e:
            logger.error(f"板块指标计算失败: {e}", exc_info=True)

        return indicators

    def calculate_all_indicators(self, data):
        """
        计算所有指标

        Args:
            data: 从DataFetcher获取的数据字典

        Returns:
            dict包含所有计算好的指标
        """
        logger.info("开始计算所有指标")

        indicators = {
            'reits_market': {},
            'interest_rate': {},
            'stock_market': {},
            'market_sentiment': {},
            'sector_analysis': {}
        }

        try:
            # REITs市场指标
            indicators['reits_market'] = self.calculate_reits_indicators(data['reits_index'])

            # 利率指标
            indicators['interest_rate'] = self.calculate_interest_rate_indicators(
                data['interest_rate'],
                data['reits_index']
            )

            # 股市指标
            indicators['stock_market'] = self.calculate_stock_indicators(
                data['stock_index'],
                data['reits_index']
            )

            # 情绪指标
            indicators['market_sentiment'] = self.calculate_sentiment_indicators(
                data['market_turnover']
            )

            # 板块指标
            if data.get('sector_index') is not None:
                indicators['sector_analysis'] = self.calculate_sector_indicators(
                    data['sector_index'],
                    data['reits_index'],
                    data['sector'],
                    data.get('sectors_ranking', [])
                )

            logger.info("所有指标计算完成")

            return indicators

        except Exception as e:
            logger.error(f"指标计算失败: {e}", exc_info=True)
            raise
