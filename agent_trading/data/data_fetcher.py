#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取模块
从数据库获取预测结果、价格数据、分红数据等
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from datetime import date, datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from agent_trading.data.db_manager import DBManager
from agent_trading.utils.logger import get_logger
from agent_trading.config import TABLE_DECISIONS_STRONG, MIN_LISTED_DAYS

logger = get_logger()


class DataFetcher:
    """
    数据获取类

    从数据库获取决策专家所需的各类数据
    """

    def __init__(self):
        """初始化数据获取器"""
        self.db = DBManager()

    def get_trading_dates(
        self,
        start_date: str,
        end_date: str,
        fund_codes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取指定日期范围内的交易日和基金列表

        Args:
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 终止日期 (YYYY-MM-DD)
            fund_codes: 基金代码列表，如果为空则获取所有基金

        Returns:
            List[Dict]: 包含 analysis_date 和 fund_code 的记录列表
        """
        if fund_codes:
            placeholders = ','.join(['%s'] * len(fund_codes))
            sql = f"""
                SELECT DISTINCT analysis_date, fund_code
                FROM price_predictions
                WHERE analysis_date BETWEEN %s AND %s
                  AND fund_code IN ({placeholders})
                ORDER BY fund_code, analysis_date
            """
            params = (start_date, end_date) + tuple(fund_codes)
        else:
            sql = """
                SELECT DISTINCT analysis_date, fund_code
                FROM price_predictions
                WHERE analysis_date BETWEEN %s AND %s
                ORDER BY fund_code, analysis_date
            """
            params = (start_date, end_date)

        results = self.db.execute_query(sql, params)
        return results

    def get_fund_codes_in_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[str]:
        """
        获取符合条件的基金代码列表

        只纳入在回测起始日期(start_date)时就已上市满MIN_LISTED_DAYS天的基金
        与对照组(baseline_strategies)保持一致，确保固定基金池

        Args:
            start_date: 起始日期
            end_date: 终止日期

        Returns:
            List[str]: 基金代码列表
        """
        # 验证日期格式
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"日期格式错误: {e}")
            return []

        if start_dt > end_dt:
            logger.error(f"起始日期({start_date})不能晚于截止日期({end_date})")
            return []

        # 1. 从 product_info 表获取所有基金的上市日期
        sql_list_date = """
            SELECT fund_code, list_date
            FROM product_info
            WHERE list_date IS NOT NULL
        """
        list_date_results = self.db.execute_query(sql_list_date)

        if not list_date_results:
            logger.warning("未找到任何基金的上市日期信息")
            return []

        # 2. 筛选在 start_date 时满足上市天数要求的基金
        eligible_funds = []

        for row in list_date_results:
            fund_code = row['fund_code']
            list_date = row['list_date']

            # 确保日期类型一致
            if isinstance(list_date, str):
                try:
                    list_date = datetime.strptime(list_date, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"基金 {fund_code} 的上市日期格式错误: {list_date}")
                    continue
            elif isinstance(list_date, datetime):
                list_date = list_date.date()

            # 计算在 start_date 时的上市天数
            listed_days_at_start = (start_dt.date() - list_date).days

            # 只纳入在 start_date 时就满足上市天数要求的基金
            if listed_days_at_start >= MIN_LISTED_DAYS:
                eligible_funds.append(fund_code)

        if not eligible_funds:
            logger.warning(f"在 {start_date} 时没有基金满足上市天数 >= {MIN_LISTED_DAYS} 天的要求")
            return []

        logger.info(f"在 {start_date} 时满足上市天数要求的基金数: {len(eligible_funds)}")

        # 3. 从 price_predictions 表验证这些基金在回测期间有预测数据
        placeholders = ','.join(['%s'] * len(eligible_funds))
        sql_predictions = f"""
            SELECT DISTINCT fund_code
            FROM price_predictions
            WHERE fund_code IN ({placeholders})
              AND analysis_date BETWEEN %s AND %s
            ORDER BY fund_code
        """
        params = tuple(eligible_funds) + (start_date, end_date)
        prediction_results = self.db.execute_query(sql_predictions, params)

        # 提取有预测数据的基金代码
        fund_codes_with_predictions = [r['fund_code'] for r in prediction_results]

        if not fund_codes_with_predictions:
            logger.warning(f"符合条件的基金在 [{start_date}, {end_date}] 期间没有预测数据")
            return []

        logger.info(f"最终符合条件且有预测数据的基金数: {len(fund_codes_with_predictions)}")

        return fund_codes_with_predictions

    def get_prediction_data(
        self,
        analysis_date: str,
        fund_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定日期和基金的预测数据

        包括四专家输出和预测专家输出

        Args:
            analysis_date: 分析日期
            fund_code: 基金代码

        Returns:
            Dict: 预测数据，包含四专家和预测专家的输出
        """
        sql = """
            SELECT
                analysis_date,
                fund_code,
                direction_prediction,
                volatility_threshold,
                conf_T1,
                conf_T5,
                conf_T20,
                pred_dir_T1,
                pred_dir_T5,
                pred_dir_T20,
                output_announcement,
                output_market,
                output_price,
                output_event,
                t1_prob_up,
                t1_prob_down,
                t1_prob_side,
                t5_prob_up,
                t5_prob_down,
                t5_prob_side,
                t20_prob_up,
                t20_prob_down,
                t20_prob_side
            FROM price_predictions
            WHERE analysis_date = %s AND fund_code = %s
        """
        results = self.db.execute_query(sql, (analysis_date, fund_code), fetch_one=True)

        if not results:
            return None

        data = results[0]

        # 解析JSON字段
        json_fields = [
            'direction_prediction',
            'output_announcement',
            'output_market',
            'output_price',
            'output_event'
        ]

        for field in json_fields:
            if data.get(field):
                try:
                    if isinstance(data[field], str):
                        data[field] = json.loads(data[field])
                except json.JSONDecodeError as e:
                    logger.warning(f"解析{field}的JSON失败: {e}")
                    data[field] = None

        return data

    def get_price_data(
        self,
        fund_code: str,
        end_date: str,
        days: int = 20
    ) -> List[Dict[str, Any]]:
        """
        获取指定基金截止到指定日期的历史价格数据

        Args:
            fund_code: 基金代码
            end_date: 截止日期
            days: 获取的天数

        Returns:
            List[Dict]: 价格数据列表，从旧到新排序
        """
        sql = """
            SELECT trade_date, close
            FROM price_data
            WHERE fund_code = %s AND trade_date <= %s
            ORDER BY trade_date DESC
            LIMIT %s
        """
        results = self.db.execute_query(sql, (fund_code, end_date, days))

        # 反转顺序，使其从旧到新
        return list(reversed(results))

    def get_single_day_price(
        self,
        fund_code: str,
        trade_date: str
    ) -> Optional[float]:
        """
        获取单日收盘价

        Args:
            fund_code: 基金代码
            trade_date: 交易日期

        Returns:
            float: 收盘价，如果不存在返回None
        """
        sql = """
            SELECT close
            FROM price_data
            WHERE fund_code = %s AND trade_date = %s
        """
        results = self.db.execute_query(sql, (fund_code, trade_date), fetch_one=True)

        if results:
            return float(results[0]['close'])
        return None

    def get_dividend_on_date(
        self,
        fund_code: str,
        ex_dividend_date: str
    ) -> Optional[float]:
        """
        获取指定除息日的分红金额

        Args:
            fund_code: 基金代码
            ex_dividend_date: 除息日

        Returns:
            float: 每份分红金额，如果没有分红返回None
        """
        sql = """
            SELECT dividend_per_share
            FROM dividend
            WHERE fund_code = %s AND ex_dividend_date = %s
        """
        results = self.db.execute_query(sql, (fund_code, ex_dividend_date), fetch_one=True)

        if results:
            return float(results[0]['dividend_per_share'])
        return None

    def get_previous_trading_date(
        self,
        fund_code: str,
        current_date: str
    ) -> Optional[str]:
        """
        获取前一个交易日

        Args:
            fund_code: 基金代码
            current_date: 当前日期

        Returns:
            str: 前一个交易日，如果不存在返回None
        """
        sql = """
            SELECT trade_date
            FROM price_data
            WHERE fund_code = %s AND trade_date < %s
            ORDER BY trade_date DESC
            LIMIT 1
        """
        results = self.db.execute_query(sql, (fund_code, current_date), fetch_one=True)

        if results:
            trade_date = results[0]['trade_date']
            if isinstance(trade_date, date):
                return trade_date.strftime('%Y-%m-%d')
            return str(trade_date)
        return None

    def get_fund_trading_dates(
        self,
        fund_code: str,
        start_date: str,
        end_date: str
    ) -> List[str]:
        """
        获取指定基金在日期范围内的所有交易日

        Args:
            fund_code: 基金代码
            start_date: 起始日期
            end_date: 终止日期

        Returns:
            List[str]: 交易日列表
        """
        sql = """
            SELECT DISTINCT analysis_date
            FROM price_predictions
            WHERE fund_code = %s
              AND analysis_date BETWEEN %s AND %s
            ORDER BY analysis_date
        """
        results = self.db.execute_query(sql, (fund_code, start_date, end_date))

        dates = []
        for r in results:
            d = r['analysis_date']
            if isinstance(d, date):
                dates.append(d.strftime('%Y-%m-%d'))
            else:
                dates.append(str(d))
        return dates

    def save_trading_decision(self, decision_data: Dict[str, Any]) -> int:
        """
        保存决策记录到数据库

        Args:
            decision_data: 决策数据字典

        Returns:
            int: 插入记录的ID
        """
        sql = """
            INSERT INTO trading_decisions_strong (
                analysis_date, fund_code,
                position_before, shares_before, cash_before, nav_before,
                days_since_start, is_building_phase,
                current_price, volatility_threshold, price_change_1d, ratio_t, daily_regime,
                signal_level_t1, signal_level_t5, signal_level_t20,
                delta_steps, target_position, action_type, rationale_core,
                position_after, shares_after, cash_after, nav_after,
                peak_nav, max_drawdown, total_return,
                round_entry_price, round_peak_nav, round_max_drawdown, round_return, round_holding_days,
                trade_shares, trade_amount, trade_cost, dividend_today, total_dividend_received,
                llm_prompt, llm_reasoning, llm_response,
                model_name, model_type,
                risk_override_reason, delta_steps_original
            ) VALUES (
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s
            )
        """
        params = (
            decision_data['analysis_date'],
            decision_data['fund_code'],
            decision_data['position_before'],
            decision_data['shares_before'],
            decision_data['cash_before'],
            decision_data['nav_before'],
            decision_data['days_since_start'],
            decision_data['is_building_phase'],
            decision_data['current_price'],
            decision_data['volatility_threshold'],
            decision_data.get('price_change_1d'),
            decision_data.get('ratio_t'),
            decision_data.get('daily_regime'),
            decision_data.get('signal_level_t1'),
            decision_data.get('signal_level_t5'),
            decision_data.get('signal_level_t20'),
            decision_data['delta_steps'],
            decision_data['target_position'],
            decision_data.get('action_type'),
            decision_data.get('rationale_core'),
            decision_data['position_after'],
            decision_data['shares_after'],
            decision_data['cash_after'],
            decision_data['nav_after'],
            decision_data['peak_nav'],
            decision_data['max_drawdown'],
            decision_data['total_return'],
            decision_data.get('round_entry_price'),
            decision_data.get('round_peak_nav'),
            decision_data.get('round_max_drawdown'),
            decision_data.get('round_return'),
            decision_data.get('round_holding_days', 0),
            decision_data.get('trade_shares', 0),
            decision_data.get('trade_amount', 0),
            decision_data.get('trade_cost', 0),
            decision_data.get('dividend_today', 0),
            decision_data.get('total_dividend_received', 0),
            decision_data['llm_prompt'],
            decision_data.get('llm_reasoning'),
            decision_data['llm_response'],
            decision_data['model_name'],
            decision_data.get('model_type', 'strong_llm'),
            decision_data.get('risk_override_reason'),
            decision_data.get('delta_steps_original')
        )

        return self.db.execute_insert(sql, params)

    def get_existing_decision_dates(
        self,
        fund_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> List[str]:
        """
        获取已有决策记录的日期列表

        Args:
            fund_code: 基金代码
            start_date: 起始日期（可选）
            end_date: 终止日期（可选）

        Returns:
            List[str]: 已有决策记录的日期列表（按日期排序）
        """
        sql = f"""
            SELECT DISTINCT analysis_date
            FROM {TABLE_DECISIONS_STRONG}
            WHERE fund_code = %s
        """
        params = [fund_code]

        if start_date:
            sql += " AND analysis_date >= %s"
            params.append(start_date)

        if end_date:
            sql += " AND analysis_date <= %s"
            params.append(end_date)

        sql += " ORDER BY analysis_date"

        results = self.db.execute_query(sql, tuple(params))

        # 提取日期列表
        dates = []
        for row in results:
            date_value = row['analysis_date']
            # 转换为字符串格式
            if isinstance(date_value, date):
                dates.append(date_value.strftime('%Y-%m-%d'))
            else:
                dates.append(str(date_value))

        return dates

    def get_last_decision_record(
        self,
        fund_code: str,
        end_date: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定基金的最后一条决策记录（用于恢复账户状态）

        Args:
            fund_code: 基金代码
            end_date: 截止日期（可选，获取该日期之前的最后一条记录）

        Returns:
            Dict: 决策记录，如果不存在返回None
        """
        sql = f"""
            SELECT *
            FROM {TABLE_DECISIONS_STRONG}
            WHERE fund_code = %s
        """
        params = [fund_code]

        if end_date:
            sql += " AND analysis_date <= %s"
            params.append(end_date)

        sql += " ORDER BY analysis_date DESC LIMIT 1"

        results = self.db.execute_query(sql, tuple(params))

        if results:
            return results[0]
        return None
