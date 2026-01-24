# db_config.py

def get_db_reits_config():
    """
    返回 MySQL 数据库reits连接的配置信息。
    """
    db_reits_config = {
        'host': '127.0.0.1',       # 数据库主机
        'port': 3306,               # 数据库端口
        'user': 'root',              # 数据库用户名
        'password': 'YOUR_DB_PASSWORD',        # 数据库密码
        'database': 'reits',         # 数据库名称
        'charset': 'utf8mb4',        # 字符集
        'init_command': "SET SESSION collation_connection = 'utf8mb4_unicode_ci'"  # 设置连接排序规则
    }
    return db_reits_config

