"""
数据获取模块
从数据库获取所需的原始数据
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymysql
from config.db_config import get_db_reits_config

logger = logging.getLogger(__name__)


class DataFetcher:
    """数据获取类"""

    def __init__(self):
        """初始化数据库连接"""
        self.conn = None

    def connect(self):
        """建立数据库连接"""
        try:
            db_config = get_db_reits_config()
            self.conn = pymysql.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                charset=db_config['charset']
            )
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

    def _find_nearest_date_with_data(self, table_name, date_column, target_date, value_column, max_days_back=30):
        """
        向前查找最近的有数据的日期

        Args:
            table_name: 表名
            date_column: 日期字段名
            target_date: 目标日期
            value_column: 要检查的值字段
            max_days_back: 最多向前查找天数

        Returns:
            最近有数据的日期，如果找不到返回None
        """
        cursor = self.conn.cursor()

        # 向前查找
        for i in range(max_days_back + 1):
            check_date = target_date - timedelta(days=i)

            query = f"""
                SELECT {date_column}
                FROM {table_name}
                WHERE {date_column} <= %s
                  AND {value_column} IS NOT NULL
                ORDER BY {date_column} DESC
                LIMIT 1
            """

            cursor.execute(query, (check_date,))
            result = cursor.fetchone()

            if result:
                found_date = result[0]
                if i > 0:
                    logger.warning(f"目标日期{target_date}无数据，使用{found_date}的数据（向前{i}天）")
                return found_date

        cursor.close()
        logger.error(f"在{max_days_back}天内未找到{table_name}的有效数据")
        return None

    def get_reits_index_data(self, end_date, lookback_days=300):
        """
        获取中证REITs全收益指数历史数据

        Args:
            end_date: 截止日期
            lookback_days: 向前回溯天数

        Returns:
            DataFrame with columns: trade_date, close_price
        """
        import pandas as pd

        # 先找到有数据的日期
        actual_date = self._find_nearest_date_with_data(
            'index_price_data',
            'trade_date',
            end_date,
            '中证REITs全收益'
        )

        if not actual_date:
            raise ValueError(f"无法获取{end_date}的REITs指数数据")

        cursor = self.conn.cursor()

        query = """
            SELECT
                trade_date,
                `中证REITs全收益` as close_price
            FROM index_price_data
            WHERE trade_date <= %s
              AND `中证REITs全收益` IS NOT NULL
            ORDER BY trade_date DESC
            LIMIT %s
        """

        cursor.execute(query, (actual_date, lookback_days))
        results = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(results, columns=['trade_date', 'close_price'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 将 Decimal 转换为 float
        df['close_price'] = df['close_price'].astype(float)

        logger.info(f"获取REITs指数数据：{len(df)}条，日期范围{df['trade_date'].min()} - {df['trade_date'].max()}")

        return df

    def get_fund_price_data(self, fund_code, end_date, lookback_days=150):
        """
        获取指定基金的历史价格数据（用于计算动态阈值）

        Args:
            fund_code: 基金代码
            end_date: 截止日期
            lookback_days: 向前回溯天数

        Returns:
            DataFrame with columns: trade_date, close, vol
        """
        import pandas as pd

        cursor = self.conn.cursor()

        # 计算起始日期
        start_date = end_date - timedelta(days=lookback_days)

        query = """
            SELECT trade_date, close, vol
            FROM price_data
            WHERE fund_code = %s
              AND trade_date >= %s
              AND trade_date <= %s
            ORDER BY trade_date ASC
        """

        cursor.execute(query, (fund_code, start_date, end_date))
        results = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(results, columns=['trade_date', 'close', 'vol'])

        # 将 Decimal 转换为 float
        df['close'] = df['close'].astype(float)
        if 'vol' in df.columns:
            df['vol'] = df['vol'].astype(float)

        logger.info(f"获取基金{fund_code}价格数据：{len(df)}条，日期范围{df['trade_date'].min() if len(df) > 0 else 'N/A'} - {df['trade_date'].max() if len(df) > 0 else 'N/A'}")

        return df

    def get_interest_rate_data(self, end_date, lookback_days=300):
        """
        获取10年期国债收益率历史数据

        Args:
            end_date: 截止日期
            lookback_days: 向前回溯天数

        Returns:
            DataFrame with columns: trade_date, rate
        """
        import pandas as pd

        actual_date = self._find_nearest_date_with_data(
            'index_price_data',
            'trade_date',
            end_date,
            '10年期国债收益率'
        )

        if not actual_date:
            raise ValueError(f"无法获取{end_date}的国债收益率数据")

        cursor = self.conn.cursor()

        query = """
            SELECT
                trade_date,
                `10年期国债收益率` as rate
            FROM index_price_data
            WHERE trade_date <= %s
              AND `10年期国债收益率` IS NOT NULL
            ORDER BY trade_date DESC
            LIMIT %s
        """

        cursor.execute(query, (actual_date, lookback_days))
        results = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(results, columns=['trade_date', 'rate'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 将 Decimal 转换为 float
        df['rate'] = df['rate'].astype(float)

        logger.info(f"获取国债收益率数据：{len(df)}条，日期范围{df['trade_date'].min()} - {df['trade_date'].max()}")

        return df

    def get_stock_index_data(self, end_date, lookback_days=300):
        """
        获取上证指数和中证红利指数历史数据

        Args:
            end_date: 截止日期
            lookback_days: 向前回溯天数

        Returns:
            DataFrame with columns: trade_date, sh_index, dividend_index
        """
        import pandas as pd

        actual_date = self._find_nearest_date_with_data(
            'index_price_data',
            'trade_date',
            end_date,
            '上证指数'
        )

        if not actual_date:
            raise ValueError(f"无法获取{end_date}的股票指数数据")

        cursor = self.conn.cursor()

        query = """
            SELECT
                trade_date,
                `上证指数` as sh_index,
                `中证红利` as dividend_index
            FROM index_price_data
            WHERE trade_date <= %s
              AND `上证指数` IS NOT NULL
              AND `中证红利` IS NOT NULL
            ORDER BY trade_date DESC
            LIMIT %s
        """

        cursor.execute(query, (actual_date, lookback_days))
        results = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(results, columns=['trade_date', 'sh_index', 'dividend_index'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 将 Decimal 转换为 float
        df['sh_index'] = df['sh_index'].astype(float)
        df['dividend_index'] = df['dividend_index'].astype(float)

        logger.info(f"获取股票指数数据：{len(df)}条，日期范围{df['trade_date'].min()} - {df['trade_date'].max()}")

        return df

    def get_market_turnover_data(self, end_date, lookback_days=300):
        """
        获取全市场成交额和市值数据

        Args:
            end_date: 截止日期
            lookback_days: 向前回溯天数

        Returns:
            DataFrame with columns: trade_date, turnover_amount(亿元), total_market_value(亿元)
        """
        import pandas as pd

        actual_date = self._find_nearest_date_with_data(
            'price_data',
            'trade_date',
            end_date,
            'amount'
        )

        if not actual_date:
            raise ValueError(f"无法获取{end_date}的成交数据")

        cursor = self.conn.cursor()

        # 计算成交额（从千元转为亿元）和总市值（从元转为亿元）
        query = """
            SELECT
                p.trade_date,
                SUM(p.amount) / 100000 as turnover_amount,  -- 千元 -> 亿元
                SUM(i.total_market_value) / 100000000 as total_market_value  -- 元 -> 亿元
            FROM price_data p
            LEFT JOIN (
                SELECT trade_date, SUM(total_market_value) as total_market_value
                FROM industry_indices
                GROUP BY trade_date
            ) i ON p.trade_date = i.trade_date
            WHERE p.trade_date <= %s
            GROUP BY p.trade_date
            ORDER BY p.trade_date DESC
            LIMIT %s
        """

        cursor.execute(query, (actual_date, lookback_days))
        results = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(results, columns=['trade_date', 'turnover_amount', 'total_market_value'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 将 Decimal 转换为 float
        df['turnover_amount'] = df['turnover_amount'].astype(float)
        df['total_market_value'] = df['total_market_value'].astype(float)

        # 处理市值缺失值（向前填充）
        df['total_market_value'] = df['total_market_value'].ffill()

        logger.info(f"获取市场成交数据：{len(df)}条，日期范围{df['trade_date'].min()} - {df['trade_date'].max()}")

        return df

    def get_sector_index_data(self, sector_name, end_date, lookback_days=300):
        """
        获取特定板块指数数据

        Args:
            sector_name: 板块名称（asset_type）
            end_date: 截止日期
            lookback_days: 向前回溯天数

        Returns:
            DataFrame with columns: trade_date, index_value
        """
        import pandas as pd

        actual_date = self._find_nearest_date_with_data(
            'industry_indices',
            'trade_date',
            end_date,
            'index_value'
        )

        if not actual_date:
            raise ValueError(f"无法获取{end_date}的板块指数数据")

        cursor = self.conn.cursor()

        query = """
            SELECT
                trade_date,
                index_value
            FROM industry_indices
            WHERE trade_date <= %s
              AND asset_type = %s
              AND index_value IS NOT NULL
            ORDER BY trade_date DESC
            LIMIT %s
        """

        cursor.execute(query, (actual_date, sector_name, lookback_days))
        results = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(results, columns=['trade_date', 'index_value'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 将 Decimal 转换为 float
        df['index_value'] = df['index_value'].astype(float)

        logger.info(f"获取{sector_name}板块数据：{len(df)}条")

        return df

    def get_fund_sector(self, fund_code):
        """
        获取基金所属板块

        Args:
            fund_code: 基金代码

        Returns:
            板块名称（asset_type）
        """
        cursor = self.conn.cursor()

        query = """
            SELECT asset_type
            FROM product_info
            WHERE fund_code = %s
        """

        cursor.execute(query, (fund_code,))
        result = cursor.fetchone()
        cursor.close()

        if not result:
            logger.warning(f"未找到基金{fund_code}的板块信息")
            return None

        sector = result[0]
        logger.info(f"基金{fund_code}属于{sector}板块")
        return sector

    def get_all_sectors_ranking(self, end_date, period_days=20):
        """
        获取所有板块的涨跌幅排名

        Args:
            end_date: 截止日期
            period_days: 统计周期（天数）

        Returns:
            list of dict: [{'sector': 'xxx', 'change': xx%, 'rank': x}, ...]
        """
        import pandas as pd

        actual_date = self._find_nearest_date_with_data(
            'industry_indices',
            'trade_date',
            end_date,
            'index_value'
        )

        if not actual_date:
            return []

        start_date = actual_date - timedelta(days=period_days * 1.5)  # 多取一些防止缺失

        cursor = self.conn.cursor()

        query = """
            SELECT
                asset_type,
                trade_date,
                index_value
            FROM industry_indices
            WHERE trade_date BETWEEN %s AND %s
              AND index_value IS NOT NULL
            ORDER BY asset_type, trade_date
        """

        cursor.execute(query, (start_date, actual_date))
        results = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(results, columns=['asset_type', 'trade_date', 'index_value'])

        # 将 Decimal 转换为 float
        df['index_value'] = df['index_value'].astype(float)

        # 计算每个板块的涨跌幅
        rankings = []
        for sector in df['asset_type'].unique():
            sector_data = df[df['asset_type'] == sector].sort_values('trade_date')
            if len(sector_data) >= 2:
                latest = sector_data.iloc[-1]['index_value']
                earliest = sector_data.iloc[0]['index_value']
                change = (latest - earliest) / earliest * 100

                rankings.append({
                    'sector': sector,
                    'change': round(change, 2)
                })

        # 排序
        rankings.sort(key=lambda x: x['change'], reverse=True)
        for i, item in enumerate(rankings, 1):
            item['rank'] = i

        logger.info(f"获取{len(rankings)}个板块的排名数据")

        return rankings

    def fetch_all_data(self, fund_code, analysis_date):
        """
        获取所有需要的数据

        Args:
            fund_code: 基金代码
            analysis_date: 分析日期

        Returns:
            dict包含所有数据
        """
        logger.info(f"开始获取数据：基金{fund_code}，日期{analysis_date}")

        self.connect()

        try:
            # 转换日期格式
            if isinstance(analysis_date, str):
                analysis_date = datetime.strptime(analysis_date, '%Y-%m-%d').date()

            # 获取基金板块
            sector = self.get_fund_sector(fund_code)

            # 获取各类数据（300个交易日约1.2年）
            data = {
                'fund_code': fund_code,
                'sector': sector,
                'analysis_date': analysis_date,
                'reits_index': self.get_reits_index_data(analysis_date, 300),
                'interest_rate': self.get_interest_rate_data(analysis_date, 300),
                'stock_index': self.get_stock_index_data(analysis_date, 300),
                'market_turnover': self.get_market_turnover_data(analysis_date, 300),
                'fund_price': self.get_fund_price_data(fund_code, analysis_date, 150),  # 获取基金价格数据
            }

            # 如果有板块信息，获取板块数据
            if sector:
                data['sector_index'] = self.get_sector_index_data(sector, analysis_date, 300)
                data['sectors_ranking'] = self.get_all_sectors_ranking(analysis_date, 20)
            else:
                data['sector_index'] = None
                data['sectors_ranking'] = []

            logger.info("数据获取完成")

            return data

        except Exception as e:
            logger.error(f"数据获取失败: {e}", exc_info=True)
            raise
        finally:
            self.close()
