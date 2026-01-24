#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库管理模块
封装数据库连接和基本操作
"""

import os
import sys
import pymysql
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_trading_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_trading_dir)
sys.path.insert(0, project_root)

from config.db_config import get_db_reits_config
from agent_trading.utils.logger import get_logger

logger = get_logger()


class DBManager:
    """
    数据库管理类

    封装MySQL数据库的连接和基本操作
    """

    def __init__(self):
        """初始化数据库管理器"""
        self.config = get_db_reits_config()
        self._connection = None

    @contextmanager
    def get_connection(self):
        """
        获取数据库连接（上下文管理器）

        Yields:
            pymysql.Connection: 数据库连接
        """
        conn = None
        try:
            conn = pymysql.connect(**self.config)
            yield conn
        finally:
            if conn:
                conn.close()

    def execute_query(
        self,
        sql: str,
        params: tuple = None,
        fetch_one: bool = False
    ) -> List[Dict[str, Any]]:
        """
        执行查询语句

        Args:
            sql: SQL查询语句
            params: 查询参数
            fetch_one: 是否只获取一条记录

        Returns:
            List[Dict]: 查询结果列表
        """
        with self.get_connection() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql, params)
                if fetch_one:
                    result = cursor.fetchone()
                    return [result] if result else []
                else:
                    return cursor.fetchall()

    def execute_insert(
        self,
        sql: str,
        params: tuple = None
    ) -> int:
        """
        执行插入语句

        Args:
            sql: SQL插入语句
            params: 插入参数

        Returns:
            int: 插入记录的ID
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                conn.commit()
                return cursor.lastrowid

    def execute_many(
        self,
        sql: str,
        params_list: List[tuple]
    ) -> int:
        """
        批量执行语句

        Args:
            sql: SQL语句
            params_list: 参数列表

        Returns:
            int: 受影响的行数
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, params_list)
                conn.commit()
                return cursor.rowcount

    def execute_update(
        self,
        sql: str,
        params: tuple = None
    ) -> int:
        """
        执行更新语句

        Args:
            sql: SQL更新语句
            params: 更新参数

        Returns:
            int: 受影响的行数
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                conn.commit()
                return cursor.rowcount

    def table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在

        Args:
            table_name: 表名

        Returns:
            bool: 表是否存在
        """
        sql = """
            SELECT COUNT(*) as cnt
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """
        result = self.execute_query(
            sql,
            (self.config['database'], table_name),
            fetch_one=True
        )
        return result[0]['cnt'] > 0 if result else False

    def create_table_if_not_exists(self, create_sql: str):
        """
        如果表不存在则创建

        Args:
            create_sql: CREATE TABLE语句
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_sql)
                conn.commit()
