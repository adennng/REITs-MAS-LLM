#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向预测主流程
整合所有模块，完成从数据获取到预测结果写入的完整流程
"""

import sys
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

# 导入配置
from agent_direction import config  # noqa: E402

# 导入各个模块
from agent_direction.prediction.data_manager import get_fund_date_combinations  # noqa: E402
from agent_direction.prediction.experts_caller import call_experts_async, call_experts_sync  # noqa: E402
from agent_direction.prediction.price_context_calculator import calculate_price_context, get_fund_price_data  # noqa: E402
from agent_price.daily_threshold_calculator import calculate_daily_volatility_threshold  # noqa: E402
from agent_direction.prediction.direction_predictor import predict_direction  # noqa: E402
from agent_direction.prediction.db_writer import write_prediction_to_db  # noqa: E402


def setup_logger(log_mode: str = 'detailed') -> logging.Logger:
    """
    配置日志记录器

    Args:
        log_mode: 日志模式 ('detailed' 或 'simple')

    Returns:
        Logger: 配置好的日志记录器
    """
    logger = logging.getLogger('DirectionPrediction')
    logger.setLevel(logging.DEBUG if log_mode == 'detailed' else logging.INFO)

    # 清除已有的处理器
    logger.handlers.clear()

    # 确定日志目录
    log_dir = os.path.join(project_root, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'direction_prediction_{timestamp}.log'
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

    logger.info(f"日志文件: {log_path}")

    return logger


def process_single_combination(
    fund_code: str,
    date: str,
    logger: logging.Logger,
    log_mode: str = 'detailed'
) -> Tuple[bool, str]:
    """
    处理单个(基金代码, 交易日期)组合

    Args:
        fund_code: 基金代码
        date: 交易日期
        logger: 日志记录器
        log_mode: 日志模式

    Returns:
        Tuple[bool, str]: (是否成功, 错误信息)
    """
    logger.info("=" * 80)
    logger.info(f"开始处理: 基金 {fund_code}, 日期 {date}")
    logger.info("=" * 80)

    try:
        # Step 1: 调用四个专家
        logger.info("Step 1: 调用四个专家")

        if config.ASYNC_EXPERTS:
            all_success, expert_results = call_experts_async(
                fund_code=fund_code,
                date=date,
                timeout=config.EXPERT_TIMEOUT
            )
        else:
            all_success, expert_results = call_experts_sync(
                fund_code=fund_code,
                date=date
            )

        if not all_success:
            error_msg = f"四专家调用失败，失败的专家: {expert_results.get('errors', [])}"
            logger.error(error_msg)
            return False, error_msg

        # 提取四个专家的结果（排除errors字段）
        four_experts_list = [
            expert_results[expert_name]
            for expert_name in ['announcement', 'market', 'price', 'event']
            if expert_name in expert_results
        ]

        if len(four_experts_list) != 4:
            error_msg = f"四专家结果不完整，只获得了{len(four_experts_list)}个专家的结果"
            logger.error(error_msg)
            return False, error_msg

        if log_mode == 'detailed':
            logger.debug("四个专家返回的内容:")
            for i, expert_result in enumerate(four_experts_list, 1):
                expert_type = expert_result.get('expert_type', 'unknown')
                logger.debug(f"  专家{i} ({expert_type}):")
                logger.debug(f"    {json.dumps(expert_result, ensure_ascii=False, indent=6)}")

        logger.info("✓ 四个专家调用成功")

        # Step 2: 获取价格数据并计算动态阈值
        logger.info("Step 2: 获取价格数据并计算动态阈值")

        # 获取基金价格数据
        fund_price_df = get_fund_price_data(fund_code, date, 150)

        # 计算动态阈值
        volatility_threshold = None
        if len(fund_price_df) >= 2:
            price_list = fund_price_df['close'].tolist()
            volatility_threshold = calculate_daily_volatility_threshold(price_list)
            logger.info(f"✓ 计算动态阈值: {volatility_threshold*100:.2f}%" if volatility_threshold else "✗ 动态阈值计算失败")
        else:
            logger.warning("✗ 基金价格数据不足，无法计算动态阈值")

        # Step 3: 计算价格上下文（传入波动阈值）
        logger.info("Step 3: 计算价格上下文")
        price_context = calculate_price_context(
            fund_code=fund_code,
            current_date=date,
            volatility_threshold=volatility_threshold if volatility_threshold else 0.0,
            num_days=config.PRICE_CONTEXT_DAYS
        )

        if log_mode == 'detailed':
            logger.debug("价格上下文:")
            logger.debug(f"  {json.dumps(price_context, ensure_ascii=False, indent=4)}")

        logger.info("✓ 价格上下文计算完成")

        # Step 3: 调用LLM进行方向预测
        logger.info("Step 3: 调用LLM进行方向预测")

        if log_mode == 'detailed':
            logger.debug("传递给LLM的内容:")
            logger.debug(f"  四专家结果: {len(four_experts_list)}个专家")
            logger.debug(f"  价格上下文: {price_context}")

        success, result = predict_direction(
            fund_code=fund_code,
            date=date,
            four_experts_results=expert_results,
            price_context=price_context,
            volatility_threshold=volatility_threshold,
            max_retries=config.MAX_RETRIES,
            show_reasoning=config.SHOW_REASONING
        )

        if not success:
            error_msg = f"方向预测失败: {result.get('error', '未知错误')}"
            logger.error(error_msg)
            return False, error_msg

        prediction = result['prediction']
        reasoning = result.get('reasoning', '')  # 提取推理过程
        llm_input = result.get('llm_input', '')  # 提取传递给LLM的完整输入

        if log_mode == 'detailed':
            logger.debug("LLM返回的预测结果:")
            logger.debug(f"  {json.dumps(prediction, ensure_ascii=False, indent=4)}")

        logger.info("✓ 方向预测完成")

        # Step 4: 写入数据库
        logger.info("Step 4: 写入数据库")

        # 解析direction_probs（T+1, T+5, T+20）
        t1_probs = prediction.get('predictions', {}).get('T+1', {}).get('direction_probs', {})
        t5_probs = prediction.get('predictions', {}).get('T+5', {}).get('direction_probs', {})
        t20_probs = prediction.get('predictions', {}).get('T+20', {}).get('direction_probs', {})

        t1_prob_up = t1_probs.get('up')
        t1_prob_down = t1_probs.get('down')
        t1_prob_side = t1_probs.get('side')

        t5_prob_up = t5_probs.get('up')
        t5_prob_down = t5_probs.get('down')
        t5_prob_side = t5_probs.get('side')

        t20_prob_up = t20_probs.get('up')
        t20_prob_down = t20_probs.get('down')
        t20_prob_side = t20_probs.get('side')

        if log_mode == 'detailed':
            logger.debug("准备写入数据库的内容:")
            logger.debug(f"  analysis_date: {date}")
            logger.debug(f"  fund_code: {fund_code}")
            logger.debug(f"  direction_prediction: {json.dumps(prediction, ensure_ascii=False, indent=4)}")
            logger.debug(f"  direction_prediction_cot (推理过程):")
            if reasoning:
                logger.debug(f"    {reasoning}")
            else:
                logger.debug(f"    (无推理内容)")
            logger.debug(f"  direction_prediction_input长度: {len(llm_input)}字符")
            logger.debug(f"  volatility_threshold: {volatility_threshold}")
            logger.debug(f"  output_announcement: {json.dumps(expert_results.get('announcement'), ensure_ascii=False, indent=4)[:200]}...")
            logger.debug(f"  output_market: {json.dumps(expert_results.get('market'), ensure_ascii=False, indent=4)[:200]}...")
            logger.debug(f"  output_price: {json.dumps(expert_results.get('price'), ensure_ascii=False, indent=4)[:200]}...")
            logger.debug(f"  output_event: {json.dumps(expert_results.get('event'), ensure_ascii=False, indent=4)[:200]}...")
            logger.debug(f"  price_context: {json.dumps(price_context, ensure_ascii=False, indent=4)[:200]}...")
            logger.debug(f"  T+1概率: up={t1_prob_up}, down={t1_prob_down}, side={t1_prob_side}")
            logger.debug(f"  T+5概率: up={t5_prob_up}, down={t5_prob_down}, side={t5_prob_side}")
            logger.debug(f"  T+20概率: up={t20_prob_up}, down={t20_prob_down}, side={t20_prob_side}")

        write_success = write_prediction_to_db(
            fund_code=fund_code,
            analysis_date=date,
            prediction=prediction,
            reasoning_content=reasoning,
            llm_input=llm_input,
            volatility_threshold=volatility_threshold,
            output_announcement=expert_results.get('announcement'),
            output_market=expert_results.get('market'),
            output_price=expert_results.get('price'),
            output_event=expert_results.get('event'),
            price_context=price_context,
            t1_prob_up=t1_prob_up,
            t1_prob_down=t1_prob_down,
            t1_prob_side=t1_prob_side,
            t5_prob_up=t5_prob_up,
            t5_prob_down=t5_prob_down,
            t5_prob_side=t5_prob_side,
            t20_prob_up=t20_prob_up,
            t20_prob_down=t20_prob_down,
            t20_prob_side=t20_prob_side
        )

        if not write_success:
            error_msg = "写入数据库失败"
            logger.error(error_msg)
            return False, error_msg

        logger.info("✓ 写入数据库完成")

        logger.info("=" * 80)
        logger.info(f"✓ 处理完成: 基金 {fund_code}, 日期 {date}")
        logger.info("=" * 80)

        return True, ""

    except Exception as e:
        error_msg = f"处理过程中发生异常: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def process_combinations_multithread(
    combinations: List[Tuple[str, str]],
    logger: logging.Logger,
    log_mode: str = 'detailed',
    thread_pool_size: int = 5,
    timeout: int = 600
) -> Tuple[int, int, List[Dict]]:
    """
    使用多线程处理一批(基金代码, 交易日期)组合

    Args:
        combinations: (基金代码, 交易日期)组合列表
        logger: 日志记录器
        log_mode: 日志模式
        thread_pool_size: 线程池大小
        timeout: 单个任务超时时间（秒）

    Returns:
        Tuple[int, int, List]: (成功数量, 失败数量, 失败记录列表)
    """
    success_count = 0
    failure_count = 0
    failures = []

    # 用于线程安全的计数器
    lock = threading.Lock()

    def process_with_stats(fund_code: str, date: str) -> Tuple[bool, str, str, str]:
        """处理单个组合并返回结果"""
        success, error_msg = process_single_combination(
            fund_code=fund_code,
            date=date,
            logger=logger,
            log_mode=log_mode
        )
        return success, error_msg, fund_code, date

    logger.info(f"使用 {thread_pool_size} 个线程并发处理 {len(combinations)} 个组合")

    # 使用 ThreadPoolExecutor 进行并发处理
    with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
        # 提交所有任务
        future_to_combination = {
            executor.submit(process_with_stats, fund_code, date): (fund_code, date)
            for fund_code, date in combinations
        }

        # 等待任务完成
        completed = 0
        for future in as_completed(future_to_combination):
            fund_code, date = future_to_combination[future]

            try:
                success, error_msg, fc, dt = future.result(timeout=timeout)

                with lock:
                    completed += 1
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                        failures.append({
                            'fund_code': fc,
                            'date': dt,
                            'error': error_msg
                        })

                    # 显示进度
                    logger.info(f"  批次内进度: [{completed}/{len(combinations)}] "
                              f"成功: {success_count}, 失败: {failure_count}")

            except TimeoutError:
                with lock:
                    completed += 1
                    failure_count += 1
                    error_msg = f"处理超时（超过{timeout}秒）"
                    failures.append({
                        'fund_code': fund_code,
                        'date': date,
                        'error': error_msg
                    })
                    logger.error(f"✗ 组合处理超时 - 基金: {fund_code}, 日期: {date}")

            except Exception as e:
                with lock:
                    completed += 1
                    failure_count += 1
                    error_msg = f"处理异常: {str(e)}"
                    failures.append({
                        'fund_code': fund_code,
                        'date': date,
                        'error': error_msg
                    })
                    logger.error(f"✗ 组合处理异常 - 基金: {fund_code}, 日期: {date}, 错误: {e}")

    return success_count, failure_count, failures


def process_combinations_singlethread(
    combinations: List[Tuple[str, str]],
    logger: logging.Logger,
    log_mode: str = 'detailed'
) -> Tuple[int, int, List[Dict]]:
    """
    使用单线程顺序处理一批(基金代码, 交易日期)组合

    Args:
        combinations: (基金代码, 交易日期)组合列表
        logger: 日志记录器
        log_mode: 日志模式

    Returns:
        Tuple[int, int, List]: (成功数量, 失败数量, 失败记录列表)
    """
    success_count = 0
    failure_count = 0
    failures = []

    for idx, (fund_code, date) in enumerate(combinations, 1):
        # 显示进度
        logger.info(f"  批次内进度: [{idx}/{len(combinations)}]")

        # 处理单个组合
        success, error_msg = process_single_combination(
            fund_code=fund_code,
            date=date,
            logger=logger,
            log_mode=log_mode
        )

        if success:
            success_count += 1
        else:
            failure_count += 1
            failures.append({
                'fund_code': fund_code,
                'date': date,
                'error': error_msg
            })

        # 显示当前统计
        logger.info(f"  当前统计: 成功 {success_count}, 失败 {failure_count}")

    return success_count, failure_count, failures


def run_direction_prediction():
    """
    运行方向预测主流程
    """
    # 设置日志
    logger = setup_logger(config.LOG_MODE)

    logger.info("=" * 100)
    logger.info("方向预测专家 - 开始执行")
    logger.info("=" * 100)

    # 显示配置参数
    logger.info("配置参数:")
    logger.info(f"  起始日期: {config.START_DATE}")
    logger.info(f"  截止日期: {config.END_DATE}")
    logger.info(f"  特定基金代码: {config.SPECIFIC_FUND_CODES if config.SPECIFIC_FUND_CODES else '全部基金'}")
    logger.info(f"  基金上市时间要求: ≥{config.MIN_LISTED_DAYS}天")
    logger.info(f"  日志模式: {config.LOG_MODE}")
    logger.info(f"  异步调用专家: {config.ASYNC_EXPERTS}")
    logger.info(f"  显示推理过程: {config.SHOW_REASONING}")
    logger.info(f"  多线程处理: {config.ENABLE_MULTITHREAD}")
    if config.ENABLE_MULTITHREAD:
        logger.info(f"  线程池大小: {config.THREAD_POOL_SIZE}")
        logger.info(f"  单组合超时: {config.COMBINATION_TIMEOUT}秒")
    logger.info(f"  批次处理: {config.ENABLE_BATCH_PROCESSING}")
    if config.ENABLE_BATCH_PROCESSING:
        logger.info(f"  批次大小: {config.BATCH_SIZE}")
        logger.info(f"  批次间延迟: {config.BATCH_DELAY}秒")
    logger.info("=" * 100)

    # Step 1: 获取所有符合条件的(基金代码, 交易日期)组合
    logger.info("Step 1: 获取符合条件的(基金代码, 交易日期)组合")

    try:
        combinations = get_fund_date_combinations(
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            specific_fund_codes=config.SPECIFIC_FUND_CODES if config.SPECIFIC_FUND_CODES else None,
            min_listed_days=config.MIN_LISTED_DAYS
        )

        logger.info(f"✓ 共找到 {len(combinations)} 个符合条件的组合")

        if len(combinations) == 0:
            logger.warning("没有找到符合条件的组合，程序退出")
            return

        # 显示前几个组合
        if len(combinations) <= 5:
            logger.info(f"所有组合: {combinations}")
        else:
            logger.info(f"前5个组合: {combinations[:5]}")
            logger.info(f"后5个组合: {combinations[-5:]}")

    except Exception as e:
        logger.error(f"获取组合失败: {e}", exc_info=True)
        return

    # Step 2: 遍历处理每个组合
    logger.info("=" * 100)
    logger.info("Step 2: 开始处理每个组合")
    logger.info("=" * 100)

    total_count = len(combinations)
    success_count = 0
    failure_count = 0
    failures = []

    # 记录开始时间
    start_time = time.time()

    # 根据配置决定是否使用批次处理
    if config.ENABLE_BATCH_PROCESSING and total_count > config.BATCH_SIZE:
        # 批次处理模式
        batch_size = config.BATCH_SIZE
        num_batches = (total_count + batch_size - 1) // batch_size  # 向上取整

        logger.info(f"批次处理模式: 共 {total_count} 个组合，分为 {num_batches} 个批次")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_count)
            batch_combinations = combinations[batch_start:batch_end]

            logger.info("=" * 80)
            logger.info(f"批次 [{batch_idx + 1}/{num_batches}]: "
                       f"处理第 {batch_start + 1}-{batch_end} 个组合 "
                       f"(共 {len(batch_combinations)} 个)")
            logger.info("=" * 80)

            # 根据配置选择多线程或单线程处理
            if config.ENABLE_MULTITHREAD:
                batch_success, batch_failure, batch_failures = process_combinations_multithread(
                    combinations=batch_combinations,
                    logger=logger,
                    log_mode=config.LOG_MODE,
                    thread_pool_size=config.THREAD_POOL_SIZE,
                    timeout=config.COMBINATION_TIMEOUT
                )
            else:
                batch_success, batch_failure, batch_failures = process_combinations_singlethread(
                    combinations=batch_combinations,
                    logger=logger,
                    log_mode=config.LOG_MODE
                )

            # 累计统计
            success_count += batch_success
            failure_count += batch_failure
            failures.extend(batch_failures)

            # 显示批次统计
            logger.info(f"批次 [{batch_idx + 1}/{num_batches}] 完成: "
                       f"成功 {batch_success}, 失败 {batch_failure}")

            # 显示总体进度
            processed = batch_end
            progress_pct = processed / total_count * 100
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / processed
            remaining_items = total_count - processed
            eta_seconds = avg_time_per_item * remaining_items

            logger.info(f"总体进度: [{processed}/{total_count}] ({progress_pct:.1f}%)")
            logger.info(f"累计统计: 成功 {success_count}, 失败 {failure_count}")
            logger.info(f"已用时间: {elapsed_time/60:.1f}分钟, "
                       f"预计剩余: {eta_seconds/60:.1f}分钟")

            # 批次间延迟
            if config.BATCH_DELAY > 0 and batch_idx < num_batches - 1:
                logger.info(f"批次间延迟 {config.BATCH_DELAY} 秒...")
                time.sleep(config.BATCH_DELAY)

    else:
        # 不分批，直接处理所有组合
        logger.info(f"直接处理模式: 共 {total_count} 个组合")

        if config.ENABLE_MULTITHREAD:
            success_count, failure_count, failures = process_combinations_multithread(
                combinations=combinations,
                logger=logger,
                log_mode=config.LOG_MODE,
                thread_pool_size=config.THREAD_POOL_SIZE,
                timeout=config.COMBINATION_TIMEOUT
            )
        else:
            success_count, failure_count, failures = process_combinations_singlethread(
                combinations=combinations,
                logger=logger,
                log_mode=config.LOG_MODE
            )

    # 计算总用时
    total_time = time.time() - start_time

    # Step 3: 输出最终统计
    logger.info("=" * 100)
    logger.info("方向预测专家 - 执行完成")
    logger.info("=" * 100)
    logger.info(f"最终统计:")
    logger.info(f"  总数: {total_count}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failure_count}")
    logger.info(f"  成功率: {success_count/total_count*100:.1f}%")
    logger.info(f"  总用时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
    if total_count > 0:
        logger.info(f"  平均每个组合用时: {total_time/total_count:.1f}秒")

    if failure_count > 0:
        logger.info("\n失败记录:")
        for failure in failures:
            logger.info(f"  - 基金: {failure['fund_code']}, 日期: {failure['date']}")
            logger.info(f"    错误: {failure['error']}")

    logger.info("=" * 100)
