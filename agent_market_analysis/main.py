"""
市场分析Agent主程序入口
"""

import sys
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_market_analysis import MarketAnalyzer


def setup_logging(log_dir):
    """配置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # 生成日志文件名
    log_filename = f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_path}")
    return logger


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='公募REITs市场整体分析Agent')
    parser.add_argument('fund_code', type=str, help='基金代码')
    parser.add_argument('date', type=str, help='分析日期 (YYYY-MM-DD)')
    parser.add_argument('--fund-name', type=str, default=None, help='基金名称（可选）')
    parser.add_argument('--log-dir', type=str, default='log', help='日志目录（默认：log）')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（可选，不指定则只打印到终端）')

    args = parser.parse_args()

    # 设置日志
    log_dir = Path(__file__).parent.parent / args.log_dir
    logger = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("市场分析Agent启动")
    logger.info(f"基金代码: {args.fund_code}")
    logger.info(f"分析日期: {args.date}")
    logger.info(f"基金名称: {args.fund_name or '未指定'}")
    logger.info("=" * 80)

    try:
        # 创建分析器
        analyzer = MarketAnalyzer()

        # 执行分析
        logger.info("\n开始执行分析...")
        result = analyzer.analyze(
            fund_code=args.fund_code,
            analysis_date=args.date,
            fund_name=args.fund_name
        )

        # 打印结果到终端
        print("\n" + "=" * 80)
        print("分析结果")
        print("=" * 80)
        print(result['analysis_result'])
        print("=" * 80)

        # 如果指定了输出文件，保存结果
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存完整结果（包含推理过程）
            output_data = {
                'metadata': result['metadata'],
                'analysis_result': result['analysis_result'],
                'reasoning_process': result.get('reasoning_process', ''),
                'timestamp': datetime.now().isoformat()
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"\n分析结果已保存到: {output_path}")

        logger.info("\n分析完成!")
        return 0

    except Exception as e:
        logger.error(f"\n分析失败: {e}", exc_info=True)
        print(f"\n错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
