# import os
# import time
# import logging
# import threading
# import re
# import random 
# from config import (
#     OUTPUT_PATH, LSH_THRESHOLD, COLLOQUIAL_WORDS, CLASSIC_CHINESE_WORDS,
#     MODERN_PERPLEXITY_PERCENTILE, CLASSIC_PERPLEXITY_PERCENTILE  # 补充缺失的配置项导入
# )
# from utils import stats, monitor_running, monitor_thread
# from deduplication import preprocess_and_md5_deduplicate, minhash_lsh_deduplicate
# from filtering import determine_layered_thresholds, layered_perplexity_filter

# def main():
#     """主流程：整合预处理、去重、筛选全流程"""
#     global monitor_running
#     start_time = time.time()
    
#     # 初始化日志
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(os.path.join(OUTPUT_PATH, "clmmu_filter_log_final.log"), encoding="utf-8"),
#             logging.StreamHandler()
#         ]
#     )
    
#     # 输出启动信息
#     logging.info("🎉 启动CLMMU/CEVAL筛选流程（模块化版本）")
#     logging.info("📋 核心配置（按需求调整）:")
#     logging.info(f"   - 语义去重阈值: {LSH_THRESHOLD}（减少冗余）")
#     logging.info(f"   - 现代文困惑度: ≤{MODERN_PERPLEXITY_PERCENTILE}%分位数（低困惑度=高质量）")
#     logging.info(f"   - 古文困惑度: ≥{CLASSIC_PERPLEXITY_PERCENTILE}%分位数（高困惑度=真实古文）")
#     logging.info(f"   - 口语关键词: 扩充至{len(COLLOQUIAL_WORDS)}个（强化学术过滤）")
#     logging.info(f"   - 古文关键词: 扩充至{len(CLASSIC_CHINESE_WORDS)}个（精准识别）")
#     logging.info(f"   - 敏感话题过滤: 启用（覆盖色情、暴力、毒品等）")
#     logging.info(f"   - 已删除: 学科过滤、题型适配、事实准确性检测")
    
#     # 启动监控线程
#     monitor = threading.Thread(target=monitor_thread, daemon=True)
#     monitor.start()
    
#     try:
#         # 1. 预处理+MD5精确去重
#         preprocessed_file = preprocess_and_md5_deduplicate()
        
#         # 2. MinHash LSH语义去重
#         minhash_file = minhash_lsh_deduplicate(preprocessed_file)
        
#         # 3. 确定分层困惑度阈值
#         modern_threshold, classic_threshold = determine_layered_thresholds(minhash_file)
        
#         # 4. 分层困惑度筛选
#         kept_file, filtered_file, classic_file = layered_perplexity_filter(minhash_file, modern_threshold, classic_threshold)
        
#         # 停止监控线程
#         monitor_running = False
#         monitor.join(timeout=5)
        
#         # 输出最终统计报告
#         total_time = round(time.time() - start_time, 2)
#         logging.info("\n" + "="*80)
#         logging.info("📊 CLMMU/CEVAL筛选最终统计报告（模块化版本）")
#         logging.info("="*80)
#         logging.info(f"总输入数据: {stats['total_input']}条")
#         logging.info(f"抽样后数据: {stats['sampled_count']}条")
#         logging.info(f"📌 各阶段过滤统计:")
#         logging.info(f"   - 长度过滤: {stats['preprocess_filtered']}条")
#         logging.info(f"   - 敏感话题过滤: {stats['sensitive_filtered']}条")
#         logging.info(f"   - 口语化: {stats['colloquial_filtered']}条")
#         logging.info(f"   - 无学术特征: {stats['non_academic_filtered']}条")
#         logging.info(f"   - MD5完全重复: {stats['md5_duplicated']}条")
#         logging.info(f"   - 语义相似重复: {stats['minhash_duplicated']}条")
#         logging.info(f"   - 困惑度过滤: {stats['perplexity_filtered']}条")
#         logging.info(f"📌 最终保留统计:")
#         logging.info(f"   - 现代文保留: {stats['modern_chinese_kept']}条")
#         logging.info(f"   - 古文保留: {stats['classic_chinese_kept']}条")
#         logging.info(f"   - 总计保留: {stats['final_kept']}条")
#         logging.info(f"   - 保留比例: {stats['final_kept']/stats['sampled_count']*100:.1f}%")
#         logging.info(f"📌 阈值配置:")
#         logging.info(f"   - 现代文困惑度: ≤{modern_threshold:.2f} ({MODERN_PERPLEXITY_PERCENTILE}%分位数，低困惑度优质)")
#         logging.info(f"   - 古文困惑度: ≥{classic_threshold:.2f} ({CLASSIC_PERPLEXITY_PERCENTILE}%分位数，高困惑度真实)")
#         logging.info(f"   - 语义去重: {LSH_THRESHOLD}")
#         logging.info(f"📌 性能统计:")
#         logging.info(f"   - 总耗时: {total_time}秒 ({total_time/60:.1f}分钟)")
#         logging.info("\n📁 输出文件:")
#         logging.info(f"✅ 高质量数据（最终版）: {kept_file}")
#         logging.info(f"📜 古文专门数据: {classic_file}")
#         logging.info(f"❌ 过滤数据详情: {filtered_file}")
#         logging.info(f"📊 筛选日志: {os.path.join(OUTPUT_PATH, 'clmmu_filter_log_final.log')}")
#         logging.info("="*80)
        
#         # 保留比例校准提示
#         retention_ratio = stats['final_kept']/stats['sampled_count']*100 if stats['sampled_count'] > 0 else 0.0
#         if retention_ratio < 45:
#             logging.warning(f"⚠️ 保留比例过低（{retention_ratio:.1f}%），建议适度放松：")
#             logging.warning(f"   1. 语义去重阈值从{ LSH_THRESHOLD }→0.73")
#             logging.warning(f"   2. 现代文困惑度分位数从{MODERN_PERPLEXITY_PERCENTILE}%→80%")
#         elif retention_ratio > 55:
#             logging.warning(f"⚠️ 保留比例过高（{retention_ratio:.1f}%），建议适度收紧：")
#             logging.warning(f"   1. 语义去重阈值从{ LSH_THRESHOLD }→0.78")
#             logging.warning(f"   2. 现代文困惑度分位数从{MODERN_PERPLEXITY_PERCENTILE}%→70%")
#         else:
#             logging.info("✅ 保留比例达标（45%-55%），数据质量与数量平衡良好！")
        
#     except Exception as e:
#         logging.error(f"❌ 流程执行失败: {str(e)}", exc_info=True)
#     finally:
#         # 确保监控线程停止
#         monitor_running = False
#         monitor.join(timeout=5)
#         logging.info("🔚 CLMMU/CEVAL筛选流程结束（模块化版本）")

# if __name__ == "__main__":
#     main()

import time
import logging
import threading
from config import (
    INPUT_DIR, OUTPUT_PATH, LSH_THRESHOLD, COLLOQUIAL_WORDS, CLASSIC_CHINESE_WORDS,
    PERCENTILES, MODERN_PERPLEXITY_PERCENTILE, CLASSIC_PERPLEXITY_PERCENTILE
)
from utils import load_jsonl_files_with_sampling, monitor_thread
from filtering import preprocess_and_filter, analyze_perplexity_distribution, layered_perplexity_filter
from deduplication import md5_deduplication, minhash_lsh_deduplication

# 全局统计变量
stats = {
    "total_input": 0,
    "sampled_count": 0,
    "preprocess_filtered": 0,
    "colloquial_filtered": 0,
    "non_academic_filtered": 0,
    "md5_duplicated": 0,
    "minhash_duplicated": 0,
    "sensitive_filtered": 0,
    "perplexity_filtered": 0,
    "final_kept": 0,
    "classic_chinese_kept": 0,
    "modern_chinese_kept": 0,
    "stage_time": {}
}

def main():
    global monitor_running
    start_time = time.time()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{OUTPUT_PATH}/clmmu_filter_log_final.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("🎉 启动CLMMU/CEVAL筛选流程（模块化版本）")
    logging.info("📋 核心配置:")
    logging.info(f"   - 语义去重阈值: {LSH_THRESHOLD}（减少冗余）")
    logging.info(f"   - 现代文困惑度: ≤{PERCENTILES[MODERN_PERPLEXITY_PERCENTILE]}%分位数（低困惑度=高质量）")
    logging.info(f"   - 古文困惑度: ≥{PERCENTILES[CLASSIC_PERPLEXITY_PERCENTILE]}%分位数（高困惑度=真实古文）")
    logging.info(f"   - 口语关键词: {len(COLLOQUIAL_WORDS)}个（强化学术过滤）")
    logging.info(f"   - 古文关键词: {len(CLASSIC_CHINESE_WORDS)}个（精准识别）")
    logging.info(f"   - 敏感话题过滤: 启用（覆盖色情、暴力、毒品等）")
    logging.info(f"   - 输出文件: 高质量数据文件、日志文件、困惑度分布图")
    
    # 启动监控线程
    monitor_running = True
    monitor = threading.Thread(target=monitor_thread, args=(stats,), daemon=True)
    monitor.start()
    
    try:
        # 1. 加载数据（支持抽样）
        data_generator = load_jsonl_files_with_sampling(INPUT_DIR, stats)
        
        # 2. 预处理+基础筛选（长度、敏感词、口语化、学术特征）
        filtered_data = preprocess_and_filter(data_generator, stats)
        
        # 3. MD5精确去重
        md5_dedup_data = md5_deduplication(iter(filtered_data), stats)
        
        # 4. Minhash LSH语义去重
        minhash_data = minhash_lsh_deduplication(md5_dedup_data, stats)
        
        # 5. 分析困惑度分布，确定阈值
        modern_threshold, classic_threshold = analyze_perplexity_distribution(minhash_data)
        
        # 6. 分层困惑度筛选
        kept_file = layered_perplexity_filter(minhash_data, modern_threshold, classic_threshold, stats)
        
        # 停止监控
        monitor_running = False
        monitor.join()
        
        # 输出最终统计
        total_time = round(time.time() - start_time, 2)
        logging.info("\n" + "="*80)
        logging.info("📊 CLMMU/CEVAL筛选最终统计报告")
        logging.info("="*80)
        logging.info(f"总输入数据: {stats['total_input']}条")
        logging.info(f"抽样后数据: {stats['sampled_count']}条")
        logging.info(f"📌 各阶段过滤统计:")
        logging.info(f"   - 长度过滤: {stats['preprocess_filtered']}条")
        logging.info(f"   - 敏感话题过滤: {stats['sensitive_filtered']}条")
        logging.info(f"   - 口语化: {stats['colloquial_filtered']}条")
        logging.info(f"   - 无学术特征: {stats['non_academic_filtered']}条")
        logging.info(f"   - MD5完全重复: {stats['md5_duplicated']}条")
        logging.info(f"   - 语义相似重复: {stats['minhash_duplicated']}条")
        logging.info(f"   - 困惑度过滤: {stats['perplexity_filtered']}条")
        logging.info(f"📌 最终保留统计:")
        logging.info(f"   - 现代文保留: {stats['modern_chinese_kept']}条")
        logging.info(f"   - 古文保留: {stats['classic_chinese_kept']}条")
        logging.info(f"   - 总计保留: {stats['final_kept']}条")
        logging.info(f"   - 保留比例: {stats['final_kept']/stats['sampled_count']*100:.1f}%")
        logging.info(f"📌 阈值配置:")
        logging.info(f"   - 现代文困惑度: ≤{modern_threshold:.2f} ({PERCENTILES[MODERN_PERPLEXITY_PERCENTILE]}%分位数)")
        logging.info(f"   - 古文困惑度: ≥{classic_threshold:.2f} ({PERCENTILES[CLASSIC_PERPLEXITY_PERCENTILE]}%分位数)")
        logging.info(f"   - 语义去重: {LSH_THRESHOLD}")
        logging.info(f"📌 性能统计:")
        logging.info(f"   - 总耗时: {total_time}秒 ({total_time/60:.1f}分钟)")
        logging.info("\n📁 输出文件:")
        logging.info(f"✅ 高质量数据：{kept_file}")
        logging.info(f"📊 困惑度分布图：{OUTPUT_PATH}/layered_perplexity_distribution_final.png")
        logging.info(f"📋 筛选日志：{OUTPUT_PATH}/clmmu_filter_log_final.log")
        logging.info("="*80)
        
        # 保留比例校准提示（修复未定义变量错误）
        retention_ratio = stats['final_kept']/stats['sampled_count']*100
        if retention_ratio < 45:
            logging.warning(f"⚠️ 保留比例过低（{retention_ratio:.1f}%），建议适度放松：")
            logging.warning(f"   1. 语义去重阈值从{LSH_THRESHOLD}→0.73")
            logging.warning(f"   2. 现代文困惑度分位数从{PERCENTILES[MODERN_PERPLEXITY_PERCENTILE]}%→80%")
        elif retention_ratio > 55:
            logging.warning(f"⚠️ 保留比例过高（{retention_ratio:.1f}%），建议适度收紧：")
            logging.warning(f"   1. 语义去重阈值从{LSH_THRESHOLD}→0.78")
            logging.warning(f"   2. 现代文困惑度分位数从{PERCENTILES[MODERN_PERPLEXITY_PERCENTILE]}%→70%")
        else:
            logging.info("✅ 保留比例达标（45%-55%），数据质量与数量平衡良好！")
        
    except Exception as e:
        logging.error(f"❌ 流程执行失败: {str(e)}", exc_info=True)
    finally:
        monitor_running = False
        monitor.join(timeout=5)
        logging.info("🔚 CLMMU/CEVAL筛选流程结束")

if __name__ == "__main__":
    main()