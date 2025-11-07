# # import os
# # import json
# # import torch
# # import logging
# # import mmap
# # import time
# # import hashlib
# # import psutil
# # import threading
# # import re
# # import numpy as np
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from tqdm import tqdm
# # from datasketch import MinHash, MinHashLSH

# # # 忽略无关警告
# # import warnings
# # warnings.filterwarnings("ignore", category=UserWarning)

# # # ========== 核心配置 ==========
# # INPUT_DIR = "data"  # 输入JSONL文件目录
# # INTERMEDIATE_PATH = "data/intermediate"  # 中间文件目录
# # OUTPUT_PATH = "data/output"  # 输出文件目录
# # os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
# # os.makedirs(OUTPUT_PATH, exist_ok=True)

# # # 抽样配置
# # SAMPLING_ENABLE = True
# # SAMPLE_RATIO = 0.01
# # MAX_SAMPLE_COUNT = 10000

# # # 批量配置
# # BATCH_SIZE_PREPROCESS = 1024
# # BATCH_SIZE_PERPLEXITY = 16
# # BATCH_SIZE_MINHASH = 5000

# # # 数据过滤基础配置
# # MIN_CHAR_LEN = 10
# # MAX_CHAR_LEN = 10000
# # PERPLEXITY_THRESHOLD = None
# # MAX_SEQ_LENGTH = 512

# # # 去重配置
# # MINHASH_NUM_PERM = 128
# # LSH_THRESHOLD = 0.8

# # # 模型配置（GPT2 中文自回归）
# # MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"
# # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # # 全局统计变量
# # stats = {
# #     "total_input": 0,
# #     "sampled_count": 0,
# #     "preprocess_filtered": 0,
# #     "md5_duplicated": 0,
# #     "minhash_duplicated": 0,
# #     "perplexity_filtered": 0,
# #     "final_kept": 0,
# #     "stage_time": {}
# # }

# # # 监控线程变量
# # monitor_running = True
# # gpu_util = 0
# # cpu_mem = 0

# # # ========== 工具函数 ==========
# # def get_gpu_utilization():
# #     try:
# #         result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
# #         return int(result.strip().split("\n")[0]) if result else 0
# #     except:
# #         return 0

# # def get_cpu_memory():
# #     process = psutil.Process(os.getpid())
# #     return round(process.memory_info().rss / (1024 * 1024), 2)

# # def monitor_thread():
# #     global monitor_running, gpu_util, cpu_mem
# #     logging.info("🔍 监控线程启动（每30秒更新GPU/内存状态）")
# #     while monitor_running:
# #         gpu_util = get_gpu_utilization()
# #         cpu_mem = get_cpu_memory()
# #         total_processed = stats["sampled_count"] - stats["preprocess_filtered"] - stats["md5_duplicated"] - stats["minhash_duplicated"]
# #         progress = (total_processed / stats["sampled_count"] * 100) if stats["sampled_count"] > 0 else 0.0
# #         logging.info(
# #             f"📊 监控状态 - GPU利用率：{gpu_util}% | CPU内存：{cpu_mem}MB | "
# #             f"总输入：{stats['total_input']} | 抽样后：{stats['sampled_count']} | 已处理：{total_processed} | 进度：{progress:.1f}%"
# #         )
# #         time.sleep(30)
# #     logging.info("🔍 监控线程停止")

# # def load_jsonl_files_with_sampling(input_dir):
# #     jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl")]
# #     if not jsonl_files:
# #         raise ValueError(f"❌ 输入目录 {input_dir} 下无JSONL文件，请检查路径！")
# #     logging.info(f"📂 发现 {len(jsonl_files)} 个JSONL文件，开始加载{'（抽样模式）' if SAMPLING_ENABLE else '（全量模式）'}")
    
# #     sampled_count = 0
# #     for file_idx, file in enumerate(jsonl_files):
# #         file_name = os.path.basename(file)
# #         logging.info(f"📄 正在读取文件 {file_idx+1}/{len(jsonl_files)}：{file_name}")
        
# #         with open(file, "r", encoding="utf-8") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
# #             for line_bytes in iter(mm.readline, b""):
# #                 line = line_bytes.decode("utf-8").strip()
# #                 if not line:
# #                     continue
# #                 try:
# #                     data = json.loads(line)
# #                     text = data.get("text", "").strip()
# #                     stats["total_input"] += 1
                    
# #                     if SAMPLING_ENABLE:
# #                         if sampled_count >= MAX_SAMPLE_COUNT:
# #                             logging.info(f"✅ 抽样完成：已抽取 {sampled_count} 条样本（达到最大限制）")
# #                             return
# #                         if np.random.random() > SAMPLE_RATIO:
# #                             continue
# #                         sampled_count += 1
# #                         stats["sampled_count"] = sampled_count
                    
# #                     yield {"text": text, "original_data": data, "source_file": file_name}
# #                 except:
# #                     stats["preprocess_filtered"] += 1
# #                     continue
# #     logging.info(f"✅ 所有文件加载完成 {'（抽样模式）' if SAMPLING_ENABLE else '（全量模式）'}")
# #     logging.info(f"📊 加载统计：总输入 {stats['total_input']} 条 | 抽样后 {stats['sampled_count']} 条")

# # # ========== 1. 预处理 + MD5去重 ==========
# # def preprocess_and_md5_deduplicate():
# #     start_time = time.time()
# #     logging.info("🚀 开始预处理 + MD5精确去重")
    
# #     md5_set = set()
# #     output_file = os.path.join(INTERMEDIATE_PATH, "preprocessed_md5_dedup.jsonl")
# #     batch_buffer = []
    
# #     with open(output_file, "w", encoding="utf-8") as f_out:
# #         for item in tqdm(load_jsonl_files_with_sampling(INPUT_DIR), desc="预处理+MD5去重", total=stats["sampled_count"] if SAMPLING_ENABLE else None):
# #             text = item["text"]
# #             original_data = item["original_data"]
# #             source_file = item["source_file"]
            
# #             if len(text) < MIN_CHAR_LEN or len(text) > MAX_CHAR_LEN:
# #                 stats["preprocess_filtered"] += 1
# #                 continue
# #             text = re.sub(r"[\u200b\s]+", " ", text).strip()
            
# #             md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
# #             if md5_hash in md5_set:
# #                 stats["md5_duplicated"] += 1
# #                 continue
# #             md5_set.add(md5_hash)
            
# #             batch_buffer.append({"text": text, "original_data": original_data, "source_file": source_file, "md5": md5_hash})
# #             if len(batch_buffer) >= BATCH_SIZE_PREPROCESS:
# #                 for data in batch_buffer:
# #                     f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
# #                 batch_buffer = []
        
# #         if batch_buffer:
# #             for data in batch_buffer:
# #                 f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    
# #     stats["stage_time"]["preprocess_md5"] = round(time.time() - start_time, 2)
# #     remaining = stats["sampled_count"] - stats["preprocess_filtered"] - stats["md5_duplicated"]
# #     logging.info(
# #         f"✅ 预处理+MD5去重完成 | 耗时：{stats['stage_time']['preprocess_md5']}秒 | "
# #         f"抽样后：{stats['sampled_count']} | 长度过滤：{stats['preprocess_filtered']} | "
# #         f"完全重复（MD5）：{stats['md5_duplicated']} | 剩余：{remaining}"
# #     )
# #     return output_file

# # # ========== 2. Minhash LSH语义去重 ==========
# # def create_minhash_signature(text, num_perm=MINHASH_NUM_PERM):
# #     minhash = MinHash(num_perm=num_perm)
# #     for token in list(text):
# #         token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
# #         minhash.update(token_hash.encode('utf-8'))
# #     return minhash

# # def minhash_lsh_deduplicate(input_file):
# #     start_time = time.time()
# #     logging.info("🚀 开始Minhash LSH语义去重")
    
# #     texts = []
# #     data_list = []
# #     with open(input_file, "r", encoding="utf-8") as f:
# #         for line in tqdm(f, desc="读取预处理数据"):
# #             data = json.loads(line)
# #             texts.append(data["text"])
# #             data_list.append(data)
    
# #     if not texts:
# #         raise ValueError("❌ MD5去重后无有效数据")
    
# #     lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
# #     keep_indices = []
# #     duplicate_count = 0
    
# #     for i in tqdm(range(len(texts)), desc="MinHash去重"):
# #         minhash = create_minhash_signature(texts[i])
# #         similar_docs = lsh.query(minhash)
# #         if not similar_docs:
# #             lsh.insert(str(i), minhash)
# #             keep_indices.append(i)
# #         else:
# #             should_keep = True
# #             for doc_id in similar_docs:
# #                 if int(doc_id) < i:
# #                     should_keep = False
# #                     break
# #             if should_keep:
# #                 lsh.insert(str(i), minhash)
# #                 keep_indices.append(i)
# #             else:
# #                 duplicate_count += 1
    
# #     stats["minhash_duplicated"] = duplicate_count
# #     output_file = os.path.join(INTERMEDIATE_PATH, "minhash_dedup.jsonl")
# #     with open(output_file, "w", encoding="utf-8") as f_out:
# #         for idx in keep_indices:
# #             f_out.write(json.dumps(data_list[idx], ensure_ascii=False) + "\n")
    
# #     stats["stage_time"]["minhash_lsh"] = round(time.time() - start_time, 2)
# #     remaining = len(keep_indices)
# #     logging.info(f"✅ Minhash LSH语义去重完成 | 耗时：{stats['stage_time']['minhash_lsh']}秒 | "
# #                  f"语义相似重复：{stats['minhash_duplicated']} | 剩余：{remaining}")
# #     return output_file

# # # ========== 3. 困惑度分析 ==========
# # def load_perplexity_model():
# #     logging.info("📥 加载模型计算困惑度")
# #     start_time = time.time()
# #     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #     model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
# #     model.eval()
# #     logging.info(f"✅ 模型加载完成 | 耗时：{round(time.time() - start_time, 2)}秒")
# #     return tokenizer, model

# # def calculate_perplexity(text, tokenizer, model):
# #     try:
# #         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
# #         with torch.no_grad():
# #             outputs = model(**inputs)
# #             logits = outputs.logits
# #             shift_logits = logits[..., :-1, :].contiguous()
# #             shift_labels = inputs["input_ids"][..., 1:].contiguous()
# #             loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
# #             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
# #             avg_loss = loss.mean()
# #             return torch.exp(avg_loss).item()
# #     except:
# #         return float('inf')

# # def calculate_perplexity_batch(texts, tokenizer, model):
# #     return [calculate_perplexity(text, tokenizer, model) for text in texts]

# # def setup_chinese_font():
# #     """设置中文字体，解决matplotlib中文显示问题"""
# #     try:
# #         import matplotlib.pyplot as plt
# #         import matplotlib.font_manager as fm
        
# #         # 尝试多种中文字体
# #         chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'STHeiti']
        
# #         for font_name in chinese_fonts:
# #             if font_name in [f.name for f in fm.fontManager.ttflist]:
# #                 plt.rcParams['font.family'] = font_name
# #                 break
# #         else:
# #             # 如果没有找到中文字体，使用默认字体并警告
# #             logging.warning("⚠️ 未找到中文字体，图表中的中文可能显示为方块")
        
# #         # 解决负号显示问题
# #         plt.rcParams['axes.unicode_minus'] = False
        
# #         return True
# #     except Exception as e:
# #         logging.warning(f"⚠️ 设置中文字体失败: {str(e)}")
# #         return False
    
# # def analyze_perplexity_distribution(input_file, sample_size=1000):
# #     logging.info("🔍 开始困惑度分布分析")
# #     tokenizer, model = load_perplexity_model()
    
# #     # 读取数据
# #     texts = []
# #     with open(input_file, "r", encoding="utf-8") as f:
# #         lines = f.readlines()
# #         if len(lines) > sample_size:
# #             import random
# #             lines = random.sample(lines, sample_size)
# #         for line in lines:
# #             try:
# #                 data = json.loads(line)
# #                 text = data.get("text", "").strip()
# #                 if text and len(text) >= MIN_CHAR_LEN:  # 确保文本有效
# #                     texts.append(text)
# #             except:
# #                 continue
    
# #     if not texts:
# #         logging.error("❌ 没有有效的文本数据用于困惑度分析")
# #         return np.array([50.0] * 9)  # 返回默认值
    
# #     logging.info(f"📊 将分析 {len(texts)} 条文本的困惑度分布")
    
# #     # 计算困惑度
# #     perplexities = []
# #     for i in tqdm(range(0, len(texts), BATCH_SIZE_PERPLEXITY), desc="分析困惑度分布"):
# #         batch = texts[i:i+BATCH_SIZE_PERPLEXITY]
# #         batch_perplexities = calculate_perplexity_batch(batch, tokenizer, model)
# #         perplexities.extend(batch_perplexities)
    
# #     # 处理结果
# #     perplexities = np.array(perplexities)
# #     valid_perplexities = perplexities[perplexities < 10000]  # 过滤异常值
    
# #     if len(valid_perplexities) == 0:
# #         logging.error("❌ 无法计算困惑度分布：所有样本困惑度为 inf 或 NaN")
# #         return np.array([50.0] * 9)
    
# #     # 计算百分位数
# #     percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
# #     percentile_values = np.percentile(valid_perplexities, percentiles)
    
# #     logging.info("📈 困惑度分布分析结果：")
# #     for p, val in zip(percentiles, percentile_values):
# #         logging.info(f"    {p}% 分位数: {val:.2f}")
    
# #     # 绘制分布图（修复字体问题）
# #     try:
# #         import matplotlib.pyplot as plt
        
# #         # 设置中文字体
# #         setup_chinese_font()
        
# #         plt.figure(figsize=(12, 8))
        
# #         # 绘制直方图
# #         n, bins, patches = plt.hist(valid_perplexities, bins=50, alpha=0.7, 
# #                                    edgecolor='black', color='skyblue')
        
# #         # 添加中位数和90%分位线
# #         median_line = plt.axvline(percentile_values[3], color='red', linestyle='--', 
# #                                  linewidth=2, label=f'Median: {percentile_values[3]:.2f}')
# #         p90_line = plt.axvline(percentile_values[5], color='orange', linestyle='--', 
# #                               linewidth=2, label=f'90%: {percentile_values[5]:.2f}')
        
# #         # 设置标签和标题（使用英文避免字体问题）
# #         plt.xlabel('Perplexity', fontsize=12)
# #         plt.ylabel('Frequency', fontsize=12)
# #         plt.title('Perplexity Distribution Histogram', fontsize=14, fontweight='bold')
        
# #         # 添加图例
# #         plt.legend(fontsize=10)
        
# #         # 添加网格
# #         plt.grid(True, alpha=0.3)
        
# #         # 调整布局
# #         plt.tight_layout()
        
# #         # 保存图片
# #         output_path = os.path.join(OUTPUT_PATH, 'perplexity_distribution.png')
# #         plt.savefig(output_path, dpi=300, bbox_inches='tight')
# #         plt.close()
        
# #         logging.info(f"📊 分布图已保存: {output_path}")
        
# #         # 额外保存一个文本版本的分析结果
# #         analysis_file = os.path.join(OUTPUT_PATH, 'perplexity_analysis.txt')
# #         with open(analysis_file, 'w', encoding='utf-8') as f:
# #             f.write("困惑度分布分析结果\n")
# #             f.write("=" * 50 + "\n")
# #             for p, val in zip(percentiles, percentile_values):
# #                 f.write(f"{p}% 分位数: {val:.2f}\n")
# #             f.write(f"\n样本数量: {len(valid_perplexities)}\n")
# #             f.write(f"推荐阈值 (90%分位数): {percentile_values[5]:.2f}\n")
        
# #         logging.info(f"📄 分析结果已保存: {analysis_file}")
        
# #     except Exception as e:
# #         logging.warning(f"⚠️ 生成分布图失败: {str(e)}")
    
# #     return percentile_values

# # # ========== 4. 确定阈值 & 筛选 ==========
# # def determine_perplexity_threshold(input_file):
# #     percentiles = analyze_perplexity_distribution(input_file, sample_size=500)
# #     recommended_threshold = percentiles[5]
# #     logging.info(f"🤖 自动推荐阈值: {recommended_threshold:.2f}")
# #     return int(recommended_threshold)

# # def perplexity_filter(input_file, threshold):
# #     start_time = time.time()
# #     logging.info(f"🚀 开始困惑度筛选（阈值≤{threshold}）")
    
# #     tokenizer, model = load_perplexity_model()
    
# #     kept_file = os.path.join(OUTPUT_PATH, "kept_data.jsonl")
# #     filtered_file = os.path.join(OUTPUT_PATH, "filtered_data.jsonl")
    
# #     batch_texts = []
# #     batch_data = []
    
# #     with open(input_file, "r", encoding="utf-8") as f_in, \
# #          open(kept_file, "w", encoding="utf-8") as f_kept, \
# #          open(filtered_file, "w", encoding="utf-8") as f_filtered:
        
# #         for line in tqdm(f_in, desc="计算困惑度并筛选"):
# #             data = json.loads(line)
# #             batch_texts.append(data["text"])
# #             batch_data.append(data)
            
# #             if len(batch_texts) >= BATCH_SIZE_PERPLEXITY:
# #                 perplexities = calculate_perplexity_batch(batch_texts, tokenizer, model)
# #                 for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
# #                     data_item["perplexity"] = round(perp,2)
# #                     if perp <= threshold:
# #                         final_data = data_item["original_data"].copy()
# #                         final_data["cleaned_text"] = text
# #                         final_data["md5"] = data_item["md5"]
# #                         final_data["perplexity"] = data_item["perplexity"]
# #                         final_data["source_file"] = data_item["source_file"]
# #                         f_kept.write(json.dumps(final_data, ensure_ascii=False)+"\n")
# #                         stats["final_kept"] += 1
# #                     else:
# #                         filtered_data = data_item.copy()
# #                         filtered_data["filter_reason"] = f"困惑度超出阈值({perp:.2f}>{threshold})"
# #                         f_filtered.write(json.dumps(filtered_data, ensure_ascii=False)+"\n")
# #                         stats["perplexity_filtered"] += 1
# #                 batch_texts=[]
# #                 batch_data=[]
        
# #         if batch_texts:
# #             perplexities = calculate_perplexity_batch(batch_texts, tokenizer, model)
# #             for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
# #                 data_item["perplexity"] = round(perp,2)
# #                 if perp <= threshold:
# #                     final_data = data_item["original_data"].copy()
# #                     final_data["cleaned_text"] = text
# #                     final_data["md5"] = data_item["md5"]
# #                     final_data["perplexity"] = data_item["perplexity"]
# #                     final_data["source_file"] = data_item["source_file"]
# #                     f_kept.write(json.dumps(final_data, ensure_ascii=False)+"\n")
# #                     stats["final_kept"] += 1
# #                 else:
# #                     filtered_data = data_item.copy()
# #                     filtered_data["filter_reason"] = f"困惑度超出阈值({perp:.2f}>{threshold})"
# #                     f_filtered.write(json.dumps(filtered_data, ensure_ascii=False)+"\n")
# #                     stats["perplexity_filtered"] += 1
    
# #     stats["stage_time"]["perplexity"] = round(time.time() - start_time,2)
# #     logging.info(f"✅ 困惑度筛选完成 | 耗时：{stats['stage_time']['perplexity']}秒 | 低质量过滤：{stats['perplexity_filtered']} | 最终保留：{stats['final_kept']}")
# #     return kept_file, filtered_file

# # # ========== 主函数 ==========
# # def main():
# #     global monitor_running
# #     start_time = time.time()
    
# #     logging.basicConfig(
# #         level=logging.INFO,
# #         format="%(asctime)s - %(levelname)s - %(message)s",
# #         handlers=[
# #             logging.FileHandler(os.path.join(OUTPUT_PATH, "filter_log.log"), encoding="utf-8"),
# #             logging.StreamHandler()
# #         ]
# #     )
    
# #     # 启动监控线程
# #     monitor = threading.Thread(target=monitor_thread, daemon=True)
# #     monitor.start()
    
# #     # 1. 预处理+MD5去重
# #     preprocessed_file = preprocess_and_md5_deduplicate()
    
# #     # 2. Minhash LSH去重
# #     minhash_file = minhash_lsh_deduplicate(preprocessed_file)
    
# #     # 3. 确定困惑度阈值
# #     threshold = determine_perplexity_threshold(minhash_file)
    
# #     # 4. 困惑度筛选
# #     kept_file, filtered_file = perplexity_filter(minhash_file, threshold)
    
# #     # 停止监控
# #     monitor_running = False
# #     monitor.join()
    
# #     total_time = round(time.time() - start_time,2)
# #     logging.info(f"🎉 全流程完成 | 总耗时：{total_time}秒")
# #     logging.info(f"📊 最终统计：{stats}")

# # if __name__ == "__main__":
# #     main()


# import os
# import json
# import torch
# import logging
# import mmap
# import time
# import hashlib
# import psutil
# import threading
# import re
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# from datasketch import MinHash, MinHashLSH

# # 忽略无关警告
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# # ========== 核心配置 ==========
# INPUT_DIR = "data"
# INTERMEDIATE_PATH = "data/intermediate"
# OUTPUT_PATH = "data/output"
# os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
# os.makedirs(OUTPUT_PATH, exist_ok=True)

# # 抽样配置
# SAMPLING_ENABLE = True
# SAMPLE_RATIO = 0.01
# MAX_SAMPLE_COUNT = 1000

# # 批量配置
# BATCH_SIZE_PREPROCESS = 1024
# BATCH_SIZE_PERPLEXITY = 32  # 增加批量提高GPU利用率
# BATCH_SIZE_MINHASH = 5000

# # 数据过滤基础配置
# MIN_CHAR_LEN = 10
# MAX_CHAR_LEN = 10000
# MAX_SEQ_LENGTH = 512

# # 分层困惑度阈值配置
# CLASSIC_CHINESE_THRESHOLD = 40.55  # 古文阈值，高于此值保留
# MODERN_CHINESE_THRESHOLD = None    # 现代文阈值，自动确定

# # 去重配置
# MINHASH_NUM_PERM = 128
# LSH_THRESHOLD = 0.8

# # 模型配置
# MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 全局统计变量
# stats = {
#     "total_input": 0,
#     "sampled_count": 0,
#     "preprocess_filtered": 0,
#     "md5_duplicated": 0,
#     "minhash_duplicated": 0,
#     "perplexity_filtered": 0,
#     "final_kept": 0,
#     "classic_chinese_kept": 0,  # 新增：古文保留数量
#     "modern_chinese_kept": 0,   # 新增：现代文保留数量
#     "stage_time": {}
# }

# # 监控线程变量
# monitor_running = True
# gpu_util = 0
# cpu_mem = 0

# # ========== 工具函数 ==========
# def get_gpu_utilization():
#     try:
#         result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
#         return int(result.strip().split("\n")[0]) if result else 0
#     except:
#         return 0

# def get_cpu_memory():
#     process = psutil.Process(os.getpid())
#     return round(process.memory_info().rss / (1024 * 1024), 2)

# def monitor_thread():
#     global monitor_running, gpu_util, cpu_mem
#     logging.info("🔍 监控线程启动（每30秒更新GPU/内存状态）")
#     while monitor_running:
#         gpu_util = get_gpu_utilization()
#         cpu_mem = get_cpu_memory()
#         total_processed = stats["sampled_count"] - stats["preprocess_filtered"] - stats["md5_duplicated"] - stats["minhash_duplicated"]
#         progress = (total_processed / stats["sampled_count"] * 100) if stats["sampled_count"] > 0 else 0.0
#         logging.info(
#             f"📊 监控状态 - GPU利用率：{gpu_util}% | CPU内存：{cpu_mem}MB | "
#             f"总输入：{stats['total_input']} | 抽样后：{stats['sampled_count']} | 已处理：{total_processed} | 进度：{progress:.1f}%"
#         )
#         time.sleep(30)
#     logging.info("🔍 监控线程停止")

# def load_jsonl_files_with_sampling(input_dir):
#     jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl")]
#     if not jsonl_files:
#         raise ValueError(f"❌ 输入目录 {input_dir} 下无JSONL文件，请检查路径！")
#     logging.info(f"📂 发现 {len(jsonl_files)} 个JSONL文件，开始加载{'（抽样模式）' if SAMPLING_ENABLE else '（全量模式）'}")
    
#     sampled_count = 0
#     for file_idx, file in enumerate(jsonl_files):
#         file_name = os.path.basename(file)
#         logging.info(f"📄 正在读取文件 {file_idx+1}/{len(jsonl_files)}：{file_name}")
        
#         with open(file, "r", encoding="utf-8") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
#             for line_bytes in iter(mm.readline, b""):
#                 line = line_bytes.decode("utf-8").strip()
#                 if not line:
#                     continue
#                 try:
#                     data = json.loads(line)
#                     text = data.get("text", "").strip()
#                     stats["total_input"] += 1
                    
#                     if SAMPLING_ENABLE:
#                         if sampled_count >= MAX_SAMPLE_COUNT:
#                             logging.info(f"✅ 抽样完成：已抽取 {sampled_count} 条样本（达到最大限制）")
#                             return
#                         if np.random.random() > SAMPLE_RATIO:
#                             continue
#                         sampled_count += 1
#                         stats["sampled_count"] = sampled_count
                    
#                     yield {"text": text, "original_data": data, "source_file": file_name}
#                 except:
#                     stats["preprocess_filtered"] += 1
#                     continue
#     logging.info(f"✅ 所有文件加载完成 {'（抽样模式）' if SAMPLING_ENABLE else '（全量模式）'}")
#     logging.info(f"📊 加载统计：总输入 {stats['total_input']} 条 | 抽样后 {stats['sampled_count']} 条")

# # ========== 1. 预处理 + MD5去重 ==========
# def preprocess_and_md5_deduplicate():
#     start_time = time.time()
#     logging.info("🚀 开始预处理 + MD5精确去重")
    
#     md5_set = set()
#     output_file = os.path.join(INTERMEDIATE_PATH, "preprocessed_md5_dedup.jsonl")
#     batch_buffer = []
    
#     with open(output_file, "w", encoding="utf-8") as f_out:
#         for item in tqdm(load_jsonl_files_with_sampling(INPUT_DIR), desc="预处理+MD5去重", total=stats["sampled_count"] if SAMPLING_ENABLE else None):
#             text = item["text"]
#             original_data = item["original_data"]
#             source_file = item["source_file"]
            
#             if len(text) < MIN_CHAR_LEN or len(text) > MAX_CHAR_LEN:
#                 stats["preprocess_filtered"] += 1
#                 continue
#             text = re.sub(r"[\u200b\s]+", " ", text).strip()
            
#             md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
#             if md5_hash in md5_set:
#                 stats["md5_duplicated"] += 1
#                 continue
#             md5_set.add(md5_hash)
            
#             batch_buffer.append({"text": text, "original_data": original_data, "source_file": source_file, "md5": md5_hash})
#             if len(batch_buffer) >= BATCH_SIZE_PREPROCESS:
#                 for data in batch_buffer:
#                     f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
#                 batch_buffer = []
        
#         if batch_buffer:
#             for data in batch_buffer:
#                 f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    
#     stats["stage_time"]["preprocess_md5"] = round(time.time() - start_time, 2)
#     remaining = stats["sampled_count"] - stats["preprocess_filtered"] - stats["md5_duplicated"]
#     logging.info(
#         f"✅ 预处理+MD5去重完成 | 耗时：{stats['stage_time']['preprocess_md5']}秒 | "
#         f"抽样后：{stats['sampled_count']} | 长度过滤：{stats['preprocess_filtered']} | "
#         f"完全重复（MD5）：{stats['md5_duplicated']} | 剩余：{remaining}"
#     )
#     return output_file

# # ========== 2. Minhash LSH语义去重 ==========
# def create_minhash_signature(text, num_perm=MINHASH_NUM_PERM):
#     minhash = MinHash(num_perm=num_perm)
#     for token in list(text):
#         token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
#         minhash.update(token_hash.encode('utf-8'))
#     return minhash

# def minhash_lsh_deduplicate(input_file):
#     start_time = time.time()
#     logging.info("🚀 开始Minhash LSH语义去重")
    
#     texts = []
#     data_list = []
#     with open(input_file, "r", encoding="utf-8") as f:
#         for line in tqdm(f, desc="读取预处理数据"):
#             data = json.loads(line)
#             texts.append(data["text"])
#             data_list.append(data)
    
#     if not texts:
#         logging.warning("⚠️ MD5去重后无有效数据，跳过语义去重")
#         return input_file
    
#     lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
#     keep_indices = []
#     duplicate_count = 0
    
#     for i in tqdm(range(len(texts)), desc="MinHash去重"):
#         minhash = create_minhash_signature(texts[i])
#         similar_docs = lsh.query(minhash)
#         if not similar_docs:
#             lsh.insert(str(i), minhash)
#             keep_indices.append(i)
#         else:
#             should_keep = True
#             for doc_id in similar_docs:
#                 if int(doc_id) < i:
#                     should_keep = False
#                     break
#             if should_keep:
#                 lsh.insert(str(i), minhash)
#                 keep_indices.append(i)
#             else:
#                 duplicate_count += 1
    
#     stats["minhash_duplicated"] = duplicate_count
#     output_file = os.path.join(INTERMEDIATE_PATH, "minhash_dedup.jsonl")
#     with open(output_file, "w", encoding="utf-8") as f_out:
#         for idx in keep_indices:
#             f_out.write(json.dumps(data_list[idx], ensure_ascii=False) + "\n")
    
#     stats["stage_time"]["minhash_lsh"] = round(time.time() - start_time, 2)
#     remaining = len(keep_indices)
#     logging.info(f"✅ Minhash LSH语义去重完成 | 耗时：{stats['stage_time']['minhash_lsh']}秒 | "
#                  f"语义相似重复：{stats['minhash_duplicated']} | 剩余：{remaining}")
#     return output_file

# # ========== 3. 古文检测函数 ==========
# def is_classic_chinese(text):
#     """
#     检测文本是否为古文
#     基于古文特征词和字符比例判断
#     """
#     # 古文常见特征词
#     classic_words = [
#         '之', '乎', '者', '也', '曰', '吾', '汝', '尔', '乃', '兮', '孔子','老子','庄子','道家','墨家','法家'
#         '矣', '哉', '耶', '欤', '夫', '盖', '故', '然', '则', '而','秦','汉','明代','唐','明朝','宋朝','宋代'
#         '以', '于', '为', '其', '所', '诸', '焉', '耳', '已', '云','三国','西汉','东汉','南北朝','北宋','南宋','咸丰','郑和'
#     ]
    
#     # 古文常见句式开头
#     classic_patterns = [
#         r'^[\u4e00-\u9fff]{1,5}曰',  # "某某曰" 格式
#         r'^昔者', r'^初', r'^当是时', r'^于是', r'^既而'
#     ]
    
#     # 计算古文特征词密度
#     total_chars = len(text)
#     if total_chars == 0:
#         return False
    
#     classic_char_count = 0
#     for word in classic_words:
#         classic_char_count += text.count(word)
    
#     # 特征词密度阈值
#     density_threshold = 0.03  # 3%的字符是古文特征词
    
#     # 检查古文句式
#     pattern_match = any(re.match(pattern, text) for pattern in classic_patterns)
    
#     # 判断条件：特征词密度高或匹配古文句式
#     is_classic = (classic_char_count / total_chars > density_threshold) or pattern_match
    
#     return is_classic

# # ========== 4. 困惑度分析（分层版本） ==========
# def load_perplexity_model():
#     logging.info("📥 加载模型计算困惑度")
#     start_time = time.time()
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
#     model.eval()
#     logging.info(f"✅ 模型加载完成 | 耗时：{round(time.time() - start_time, 2)}秒")
#     return tokenizer, model

# def calculate_perplexity_batch_optimized(texts, tokenizer, model):
#     """优化版的批量困惑度计算，提高GPU利用率"""
#     try:
#         # 批量编码所有文本
#         inputs = tokenizer(
#             texts, 
#             padding=True, 
#             truncation=True, 
#             max_length=MAX_SEQ_LENGTH, 
#             return_tensors="pt"
#         ).to(DEVICE)
        
#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]
#         batch_size = input_ids.shape[0]
        
#         with torch.no_grad():
#             # 单次前向传播计算整个批次
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
            
#             perplexities = []
#             for i in range(batch_size):
#                 # 提取有效token（排除padding）
#                 valid_mask = (input_ids[i] != tokenizer.pad_token_id)
#                 valid_token_ids = input_ids[i][valid_mask]
                
#                 if len(valid_token_ids) <= 1:  # 需要至少2个token计算困惑度
#                     perplexities.append(float('inf'))
#                     continue
                
#                 # 计算损失
#                 shift_logits = logits[i, :-1, :].contiguous()
#                 shift_labels = valid_token_ids[1:].contiguous()
                
#                 loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
#                 loss = loss_fct(
#                     shift_logits.view(-1, shift_logits.size(-1)), 
#                     shift_labels.view(-1)
#                 )
                
#                 perplexity = torch.exp(loss).item()
#                 perplexities.append(perplexity)
            
#             return perplexities
            
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             logging.warning("⚠️ GPU内存不足，减小批量大小")
#             # 递归减半批量大小
#             half_size = len(texts) // 2
#             if half_size >= 1:
#                 perplexities1 = calculate_perplexity_batch_optimized(texts[:half_size], tokenizer, model)
#                 perplexities2 = calculate_perplexity_batch_optimized(texts[half_size:], tokenizer, model)
#                 return perplexities1 + perplexities2
#             else:
#                 return [calculate_perplexity(text, tokenizer, model) for text in texts]
#         else:
#             raise e
#     except Exception as e:
#         logging.warning(f"批量计算失败，回退到逐条计算: {str(e)}")
#         return [calculate_perplexity(text, tokenizer, model) for text in texts]

# def calculate_perplexity(text, tokenizer, model):
#     """单文本困惑度计算（备用）"""
#     try:
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = inputs["input_ids"][..., 1:].contiguous()
#             loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             avg_loss = loss.mean()
#             return torch.exp(avg_loss).item()
#     except:
#         return float('inf')

# def analyze_perplexity_distribution_layered(input_file, sample_size=1000):
#     """分层困惑度分布分析"""
#     logging.info("🔍 开始分层困惑度分布分析")
#     tokenizer, model = load_perplexity_model()
    
#     # 读取数据
#     texts = []
#     with open(input_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         if len(lines) > sample_size:
#             import random
#             lines = random.sample(lines, sample_size)
#         for line in lines:
#             try:
#                 data = json.loads(line)
#                 text = data.get("text", "").strip()
#                 if text and len(text) >= MIN_CHAR_LEN:
#                     texts.append(text)
#             except:
#                 continue
    
#     if not texts:
#         logging.error("❌ 没有有效的文本数据用于困惑度分析")
#         return np.array([50.0] * 9), np.array([50.0] * 9)
    
#     logging.info(f"📊 将分析 {len(texts)} 条文本的困惑度分布")
    
#     # 计算困惑度
#     perplexities = []
#     classic_texts = []  # 古文文本
#     modern_texts = []   # 现代文文本
    
#     # 先分类再批量计算
#     for i in tqdm(range(0, len(texts), BATCH_SIZE_PERPLEXITY), desc="文本分类"):
#         batch = texts[i:i+BATCH_SIZE_PERPLEXITY]
#         for text in batch:
#             if is_classic_chinese(text):
#                 classic_texts.append(text)
#             else:
#                 modern_texts.append(text)
    
#     logging.info(f"📊 文本分类结果: 古文 {len(classic_texts)} 条, 现代文 {len(modern_texts)} 条")
    
#     # 批量计算现代文困惑度
#     modern_perplexities = []
#     for i in tqdm(range(0, len(modern_texts), BATCH_SIZE_PERPLEXITY), desc="计算现代文困惑度"):
#         batch = modern_texts[i:i+BATCH_SIZE_PERPLEXITY]
#         batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
#         modern_perplexities.extend(batch_perplexities)
    
#     # 批量计算古文困惑度
#     classic_perplexities = []
#     for i in tqdm(range(0, len(classic_texts), BATCH_SIZE_PERPLEXITY), desc="计算古文困惑度"):
#         batch = classic_texts[i:i+BATCH_SIZE_PERPLEXITY]
#         batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
#         classic_perplexities.extend(batch_perplexities)
    
#     # 处理结果
#     modern_perplexities = np.array(modern_perplexities)
#     classic_perplexities = np.array(classic_perplexities)
    
#     modern_valid = modern_perplexities[modern_perplexities < 10000]
#     classic_valid = classic_perplexities[classic_perplexities < 10000]
    
#     if len(modern_valid) == 0:
#         logging.warning("⚠️ 现代文困惑度计算失败，使用默认值")
#         modern_valid = np.array([50.0])
    
#     if len(classic_valid) == 0:
#         logging.warning("⚠️ 古文困惑度计算失败，使用默认值")
#         classic_valid = np.array([50.0])
    
#     # 计算百分位数
#     percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
#     modern_percentiles = np.percentile(modern_valid, percentiles)
#     classic_percentiles = np.percentile(classic_valid, percentiles)
    
#     logging.info("📈 现代文困惑度分布:")
#     for p, val in zip(percentiles, modern_percentiles):
#         logging.info(f"    {p}% 分位数: {val:.2f}")
    
#     logging.info("📈 古文困惑度分布:")
#     for p, val in zip(percentiles, classic_percentiles):
#         logging.info(f"    {p}% 分位数: {val:.2f}")
    
#     # 绘制分层分布图
#     try:
#         import matplotlib.pyplot as plt
        
#         plt.figure(figsize=(12, 8))
        
#         # 绘制双直方图
#         plt.hist(modern_valid, bins=50, alpha=0.7, label='Modern Chinese', color='skyblue', edgecolor='black')
#         plt.hist(classic_valid, bins=50, alpha=0.7, label='Classic Chinese', color='salmon', edgecolor='black')
        
#         # 添加阈值线
#         plt.axvline(CLASSIC_CHINESE_THRESHOLD, color='red', linestyle='--', 
#                    linewidth=2, label=f'Classic Threshold: {CLASSIC_CHINESE_THRESHOLD}')
        
#         if len(modern_valid) > 0:
#             modern_threshold = modern_percentiles[5]  # 90%分位数
#             plt.axvline(modern_threshold, color='blue', linestyle='--', 
#                        linewidth=2, label=f'Modern Threshold: {modern_threshold:.2f}')
        
#         plt.xlabel('Perplexity', fontsize=12)
#         plt.ylabel('Frequency', fontsize=12)
#         plt.title('Layered Perplexity Distribution', fontsize=14, fontweight='bold')
#         plt.legend(fontsize=10)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
        
#         output_path = os.path.join(OUTPUT_PATH, 'layered_perplexity_distribution.png')
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         logging.info(f"📊 分层分布图已保存: {output_path}")
        
#     except Exception as e:
#         logging.warning(f"⚠️ 生成分布图失败: {str(e)}")
    
#     return modern_percentiles, classic_percentiles

# def determine_layered_thresholds(input_file):
#     """确定分层阈值"""
#     logging.info("🎯 开始确定分层困惑度阈值")
    
#     try:
#         modern_percentiles, classic_percentiles = analyze_perplexity_distribution_layered(input_file, sample_size=500)
        
#         # 现代文使用90%分位数作为阈值
#         modern_threshold = modern_percentiles[5]  # 90%分位数
#         logging.info(f"🤖 现代文推荐阈值: {modern_threshold:.2f} (90%分位数)")
        
#         # 古文使用固定阈值40.55，但也会显示分布情况
#         classic_threshold = CLASSIC_CHINESE_THRESHOLD
#         logging.info(f"📜 古文固定阈值: {classic_threshold}")
        
#         # 提供统计信息
#         if len(classic_percentiles) > 5:
#             classic_p90 = classic_percentiles[5]
#             logging.info(f"📊 古文90%分位数: {classic_p90:.2f}")
#             if classic_p90 > classic_threshold:
#                 logging.info("💡 古文阈值设置合理，将保留大部分古文")
#             else:
#                 logging.info("⚠️ 古文阈值可能过高，部分高质量古文可能被过滤")
        
#         return modern_threshold, classic_threshold
        
#     except Exception as e:
#         logging.error(f"❌ 确定分层阈值失败: {str(e)}")
#         logging.info("🔄 使用默认阈值: 现代文=50, 古文=40.55")
#         return 50, CLASSIC_CHINESE_THRESHOLD

# # ========== 5. 分层困惑度筛选 ==========
# def layered_perplexity_filter(input_file, modern_threshold, classic_threshold):
#     """分层困惑度筛选"""
#     start_time = time.time()
#     logging.info(f"🚀 开始分层困惑度筛选")
#     logging.info(f"   - 现代文阈值: ≤{modern_threshold:.2f}")
#     logging.info(f"   - 古文阈值: >{classic_threshold} (保留高困惑度古文)")
    
#     tokenizer, model = load_perplexity_model()
    
#     kept_file = os.path.join(OUTPUT_PATH, "kept_data_layered.jsonl")
#     filtered_file = os.path.join(OUTPUT_PATH, "filtered_data_layered.jsonl")
#     classic_file = os.path.join(OUTPUT_PATH, "classic_chinese_data.jsonl")  # 专门保存古文
    
#     batch_texts = []
#     batch_data = []
    
#     with open(input_file, "r", encoding="utf-8") as f_in, \
#          open(kept_file, "w", encoding="utf-8") as f_kept, \
#          open(filtered_file, "w", encoding="utf-8") as f_filtered, \
#          open(classic_file, "w", encoding="utf-8") as f_classic:
        
#         for line in tqdm(f_in, desc="分层困惑度筛选"):
#             try:
#                 data = json.loads(line)
#                 batch_texts.append(data["text"])
#                 batch_data.append(data)
#             except:
#                 continue
            
#             if len(batch_texts) >= BATCH_SIZE_PERPLEXITY:
#                 # 批量计算困惑度
#                 perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
                
#                 for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
#                     data_item["perplexity"] = round(perp, 2)
#                     data_item["is_classic"] = is_classic_chinese(text)
                    
#                     # 分层筛选逻辑
#                     if data_item["is_classic"]:
#                         # 古文：困惑度 > classic_threshold 时保留
#                         if perp > classic_threshold:
#                             final_data = data_item["original_data"].copy()
#                             final_data["cleaned_text"] = text
#                             final_data["md5"] = data_item["md5"]
#                             final_data["perplexity"] = data_item["perplexity"]
#                             final_data["source_file"] = data_item["source_file"]
#                             final_data["text_type"] = "classic_chinese"
#                             f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                             f_classic.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                             stats["final_kept"] += 1
#                             stats["classic_chinese_kept"] += 1
#                         else:
#                             # 古文但困惑度低，可能是质量不高的古文
#                             filtered_data = data_item.copy()
#                             filtered_data["filter_reason"] = f"古文困惑度过低({perp:.2f}≤{classic_threshold})"
#                             f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                             stats["perplexity_filtered"] += 1
#                     else:
#                         # 现代文：困惑度 ≤ modern_threshold 时保留
#                         if perp <= modern_threshold:
#                             final_data = data_item["original_data"].copy()
#                             final_data["cleaned_text"] = text
#                             final_data["md5"] = data_item["md5"]
#                             final_data["perplexity"] = data_item["perplexity"]
#                             final_data["source_file"] = data_item["source_file"]
#                             final_data["text_type"] = "modern_chinese"
#                             f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                             stats["final_kept"] += 1
#                             stats["modern_chinese_kept"] += 1
#                         else:
#                             filtered_data = data_item.copy()
#                             filtered_data["filter_reason"] = f"现代文困惑度过高({perp:.2f}>{modern_threshold})"
#                             f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                             stats["perplexity_filtered"] += 1
                
#                 batch_texts = []
#                 batch_data = []
        
#         # 处理剩余数据
#         if batch_texts:
#             perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
#             for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
#                 data_item["perplexity"] = round(perp, 2)
#                 data_item["is_classic"] = is_classic_chinese(text)
                
#                 if data_item["is_classic"]:
#                     if perp > classic_threshold:
#                         final_data = data_item["original_data"].copy()
#                         final_data["cleaned_text"] = text
#                         final_data["md5"] = data_item["md5"]
#                         final_data["perplexity"] = data_item["perplexity"]
#                         final_data["source_file"] = data_item["source_file"]
#                         final_data["text_type"] = "classic_chinese"
#                         f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                         f_classic.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                         stats["final_kept"] += 1
#                         stats["classic_chinese_kept"] += 1
#                     else:
#                         filtered_data = data_item.copy()
#                         filtered_data["filter_reason"] = f"古文困惑度过低({perp:.2f}≤{classic_threshold})"
#                         f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                         stats["perplexity_filtered"] += 1
#                 else:
#                     if perp <= modern_threshold:
#                         final_data = data_item["original_data"].copy()
#                         final_data["cleaned_text"] = text
#                         final_data["md5"] = data_item["md5"]
#                         final_data["perplexity"] = data_item["perplexity"]
#                         final_data["source_file"] = data_item["source_file"]
#                         final_data["text_type"] = "modern_chinese"
#                         f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                         stats["final_kept"] += 1
#                         stats["modern_chinese_kept"] += 1
#                     else:
#                         filtered_data = data_item.copy()
#                         filtered_data["filter_reason"] = f"现代文困惑度过高({perp:.2f}>{modern_threshold})"
#                         f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                         stats["perplexity_filtered"] += 1
    
#     stats["stage_time"]["perplexity"] = round(time.time() - start_time, 2)
#     logging.info(f"✅ 分层困惑度筛选完成 | 耗时：{stats['stage_time']['perplexity']}秒")
#     logging.info(f"📊 分层统计 - 现代文保留: {stats['modern_chinese_kept']} | 古文保留: {stats['classic_chinese_kept']}")
#     logging.info(f"📊 总计保留: {stats['final_kept']} | 过滤: {stats['perplexity_filtered']}")
    
#     return kept_file, filtered_file, classic_file

# # ========== 主函数 ==========
# def main():
#     global monitor_running
#     start_time = time.time()
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(os.path.join(OUTPUT_PATH, "filter_log_layered.log"), encoding="utf-8"),
#             logging.StreamHandler()
#         ]
#     )
    
#     logging.info("🎉 启动分层数据筛选流程")
#     logging.info(f"📋 分层配置:")
#     logging.info(f"   - 现代文: 自动确定阈值，保留低困惑度文本")
#     logging.info(f"   - 古文: 固定阈值 {CLASSIC_CHINESE_THRESHOLD}，保留高困惑度文本")
    
#     # 启动监控线程
#     monitor = threading.Thread(target=monitor_thread, daemon=True)
#     monitor.start()
    
#     try:
#         # 1. 预处理+MD5去重
#         preprocessed_file = preprocess_and_md5_deduplicate()
        
#         # 2. Minhash LSH去重
#         minhash_file = minhash_lsh_deduplicate(preprocessed_file)
        
#         # 3. 确定分层阈值
#         modern_threshold, classic_threshold = determine_layered_thresholds(minhash_file)
        
#         # 4. 分层困惑度筛选
#         kept_file, filtered_file, classic_file = layered_perplexity_filter(minhash_file, modern_threshold, classic_threshold)
        
#         # 停止监控
#         monitor_running = False
#         monitor.join()
        
#         # 输出最终统计
#         total_time = round(time.time() - start_time, 2)
#         logging.info("\n" + "="*60)
#         logging.info("📊 分层筛选最终统计报告")
#         logging.info("="*60)
#         logging.info(f"总输入数据: {stats['total_input']}条")
#         logging.info(f"抽样后数据: {stats['sampled_count']}条")
#         logging.info(f"现代文保留: {stats['modern_chinese_kept']}条")
#         logging.info(f"古文保留: {stats['classic_chinese_kept']}条")
#         logging.info(f"最终总计保留: {stats['final_kept']}条")
#         logging.info(f"保留比例: {stats['final_kept']/stats['sampled_count']*100:.1f}%")
#         logging.info(f"现代文阈值: {modern_threshold:.2f}")
#         logging.info(f"古文阈值: >{classic_threshold}")
#         logging.info(f"总耗时: {total_time}秒 ({total_time/60:.1f}分钟)")
#         logging.info("\n📁 输出文件:")
#         logging.info(f"✅ 全部保留数据: {kept_file}")
#         logging.info(f"📜 古文专门文件: {classic_file}")
#         logging.info(f"❌ 过滤数据: {filtered_file}")
#         logging.info("="*60)
        
#     except Exception as e:
#         logging.error(f"❌ 流程执行失败: {str(e)}", exc_info=True)
#     finally:
#         monitor_running = False
#         monitor.join(timeout=5)
#         logging.info("🔚 分层筛选流程结束")

# if __name__ == "__main__":
#     main()

import os
import json
import torch
import logging
import mmap
import time
import hashlib
import psutil
import threading
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

# 忽略无关警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ========== 核心配置（满足需求+50%保留率） ==========
INPUT_DIR = "data"
INTERMEDIATE_PATH = "data/intermediate"
OUTPUT_PATH = "data/output"
os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 抽样配置
SAMPLING_ENABLE = True
SAMPLE_RATIO = 0.01
MAX_SAMPLE_COUNT = 10000

# 批量配置
BATCH_SIZE_PREPROCESS = 1024
BATCH_SIZE_PERPLEXITY = 32
BATCH_SIZE_MINHASH = 5000

# 数据过滤基础配置
MIN_CHAR_LEN = 8
MAX_CHAR_LEN = 12000
MAX_SEQ_LENGTH = 512

# 去重配置（适度收紧，弥补删除的筛选环节）
MINHASH_NUM_PERM = 128
LSH_THRESHOLD = 0.76  # 从0.75→0.76，减少冗余

# 模型配置
MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== 核心筛选配置（满足需求调整） ==========
# 1. 口语关键词扩充（新增40+，强化过滤）
COLLOQUIAL_WORDS = [
    # 原有关键词
    "卧槽", "牛逼", "哈哈哈", "嘻嘻", "嘿嘿", "老铁", "懂吧", "给力", "666", "yyds",
    "绝绝子", "家人们", "谁懂啊", "救命", "哭死", "笑不活", "栓Q", "拿捏", "破防", "躺平",
    # 新增口语/网络流行语
    "绝了", "无语", "服了", "凑活", "咋整", "唠嗑", "侃大山", "扯犊子", "瞎逼逼", "逼逼赖赖",
    "磨磨唧唧", "叽叽歪歪", "碎碎念", "吐槽", "怼人", "杠精", "内卷", "躺平", "摆烂", "摸鱼",
    "划水", "打工人", "干饭人", "尾款人", "工具人", "冤种", "显眼包", "社恐", "社牛", "社死",
    "emo", "佛系", "卷王", "摆烂式", "摸鱼式", "划水式", "躺平式", "敷衍式", "糊弄学", "PUA",
    "CPU", "KTV", "yyds", "awsl", "绝绝子", "YYDS", "绝绝子", "栓Q", "拿捏了", "破防了"
]

# 2. 敏感话题过滤（新增！覆盖色情、暴力、毒品、政治敏感等）
SENSITIVE_KEYWORDS = {
    "色情相关": [
        "色情", "黄色", "裸聊", "性交易", "嫖娼", "卖淫", "淫荡", "色情视频", "色情图片", "AV",
        "三级片", "春宫", "艳照", "露骨", "性行为", "性器官", "手淫", "嫖娼", "包养", "小三",
        "二奶", "情夫", "情妇", "不正当关系", "一夜情", "约炮", "性服务", "色情直播", "色情小说"
    ],
    "暴力相关": [
        "杀人", "抢劫", "强奸", "绑架", "斗殴", "斗殴", "故意伤害", "杀人放火", "爆炸", "投毒",
        "凶器", "枪支", "弹药", "管制刀具", "暴力", "血腥", "恐怖", "虐杀", "虐待", "施暴",
        "殴打", "群殴", "互殴", "寻衅滋事", "聚众斗殴", "故意伤害", "故意杀人", "抢劫财物"
    ],
    "毒品相关": [
        "毒品", "大麻", "海洛因", "冰毒", "可卡因", "摇头丸", "K粉", "鸦片", "吗啡", "杜冷丁",
        "吸毒", "贩毒", "制毒", "吸毒者", "毒贩", "毒品交易", "毒品运输", "毒品走私"
    ],
    "政治敏感": [
        "敏感政治人物", "政治敏感事件", "颠覆", "分裂", "叛国", "暴动", "骚乱", "非法集会",
        "反动", "反政府", "反社会", "极端主义", "恐怖主义", "邪教", "法轮功", "台独", "港独", "疆独"
    ],
    "其他敏感": [
        "赌博", "诈骗", "传销", "非法集资", "洗钱", "偷税漏税", "贪污腐败", "行贿受贿",
        "假币", "非法交易", "黑客", "入侵", "病毒", "盗号", "诈骗短信", "诈骗电话"
    ]
}

# 3. 学术特征（保留，提升数据质量）
ACADEMIC_PATTERNS = [
    r"[A-Za-z0-9]=.*[A-Za-z0-9]",  # 公式
    r"定义[:：]", r"定理", r"公理", r"命题", r"推论", r"原理", r"方法", r"实验", r"分析", r"结论",
]
ACADEMIC_REQUIRE = False

# 全局统计变量（删除学科过滤相关统计项）
stats = {
    "total_input": 0,
    "sampled_count": 0,
    "preprocess_filtered": 0,
    "colloquial_filtered": 0,
    "non_academic_filtered": 0,
    "md5_duplicated": 0,
    "minhash_duplicated": 0,
    "sensitive_filtered": 0,  # 新增敏感话题统计
    "perplexity_filtered": 0,
    "final_kept": 0,
    "classic_chinese_kept": 0,
    "modern_chinese_kept": 0,
    "stage_time": {}
}

# 监控线程变量
monitor_running = True
gpu_util = 0
cpu_mem = 0

# ========== 工具函数 ==========
def get_gpu_utilization():
    try:
        result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
        return int(result.strip().split("\n")[0]) if result else 0
    except:
        return 0

def get_cpu_memory():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)

def monitor_thread():
    global monitor_running, gpu_util, cpu_mem
    logging.info("🔍 监控线程启动（每30秒更新GPU/内存状态）")
    while monitor_running:
        gpu_util = get_gpu_utilization()
        cpu_mem = get_cpu_memory()
        total_processed = stats["sampled_count"] - stats["preprocess_filtered"] - \
                          stats["colloquial_filtered"] - stats["non_academic_filtered"] - \
                          stats["md5_duplicated"] - stats["minhash_duplicated"] - stats["sensitive_filtered"]
        progress = (total_processed / stats["sampled_count"] * 100) if stats["sampled_count"] > 0 else 0.0
        logging.info(
            f"📊 监控状态 - GPU利用率：{gpu_util}% | CPU内存：{cpu_mem}MB | "
            f"总输入：{stats['total_input']} | 抽样后：{stats['sampled_count']} | 已处理：{total_processed} | 进度：{progress:.1f}%"
        )
        time.sleep(30)
    logging.info("🔍 监控线程停止")

def load_jsonl_files_with_sampling(input_dir):
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        raise ValueError(f"❌ 输入目录 {input_dir} 下无JSONL文件，请检查路径！")
    logging.info(f"📂 发现 {len(jsonl_files)} 个JSONL文件，开始加载{'（抽样模式）' if SAMPLING_ENABLE else '（全量模式）'}")
    
    sampled_count = 0
    for file_idx, file in enumerate(jsonl_files):
        file_name = os.path.basename(file)
        logging.info(f"📄 正在读取文件 {file_idx+1}/{len(jsonl_files)}：{file_name}")
        
        with open(file, "r", encoding="utf-8") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for line_bytes in iter(mm.readline, b""):
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "").strip()
                    stats["total_input"] += 1
                    
                    if SAMPLING_ENABLE:
                        if sampled_count >= MAX_SAMPLE_COUNT:
                            logging.info(f"✅ 抽样完成：已抽取 {sampled_count} 条样本（达到最大限制）")
                            return
                        if np.random.random() > SAMPLE_RATIO:
                            continue
                        sampled_count += 1
                        stats["sampled_count"] = sampled_count
                    
                    yield {"text": text, "original_data": data, "source_file": file_name}
                except:
                    stats["preprocess_filtered"] += 1
                    continue
    logging.info(f"✅ 所有文件加载完成 {'（抽样模式）' if SAMPLING_ENABLE else '（全量模式）'}")
    logging.info(f"📊 加载统计：总输入 {stats['total_input']} 条 | 抽样后 {stats['sampled_count']} 条")

# ========== 核心筛选工具函数（按需求调整） ==========
def is_colloquial(text):
    """扩充口语检测，过滤更多无学术价值文本"""
    # 关键词匹配
    for word in COLLOQUIAL_WORDS:
        if word in text:
            return True
    # 连续标点匹配（3个以上）
    if re.search(r"[！？。,，；;：:]{3,}", text):
        return True
    # 口语化句式匹配
    colloquial_patterns = [
        r"[我你他她它]（们）?[也都还就才又再]?[不没没什么没什么大不了]",
        r"[这那哪]（个些）?[也都还就才又再]?[不没没什么没什么大不了]",
        r"^[哈哈嘿嘿嘻嘻呵呵]+"
    ]
    if any(re.search(pattern, text) for pattern in colloquial_patterns):
        return True
    return False

def is_sensitive(text):
    """新增敏感话题检测，过滤违规数据"""
    # 敏感关键词匹配
    for category, words in SENSITIVE_KEYWORDS.items():
        for word in words:
            if word in text:
                return True
    # 敏感句式匹配
    sensitive_patterns = [
        r"出售.*(色情|AV|三级片)",
        r"(嫖娼|卖淫|性交易).*(价格|联系方式|地点)",
        r"(杀人|抢劫|绑架).*(方法|教程|工具)",
        r"(毒品|大麻|冰毒).*(购买|出售|运输)",
        r"(台独|港独|疆独).*(支持|宣传|分裂)"
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in sensitive_patterns):
        return True
    return False

def has_academic_features(text):
    """检测学术特征（CLMMU/CEVAL偏好）"""
    return any(re.search(pattern, text) for pattern in ACADEMIC_PATTERNS)

# ========== 1. 预处理 + MD5去重 + 基础筛选（新增敏感过滤） ==========
def preprocess_and_md5_deduplicate():
    start_time = time.time()
    logging.info("🚀 开始预处理 + MD5精确去重 + 基础筛选（含敏感话题过滤）")
    
    md5_set = set()
    output_file = os.path.join(INTERMEDIATE_PATH, "preprocessed_md5_dedup_with_sensitive_filter.jsonl")
    batch_buffer = []
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in tqdm(load_jsonl_files_with_sampling(INPUT_DIR), desc="预处理+筛选", total=stats["sampled_count"] if SAMPLING_ENABLE else None):
            text = item["text"]
            original_data = item["original_data"]
            source_file = item["source_file"]
            
            # 1. 基础长度过滤
            if len(text) < MIN_CHAR_LEN or len(text) > MAX_CHAR_LEN:
                stats["preprocess_filtered"] += 1
                continue
            
            # 2. 文本清洗
            text = re.sub(r"[\u200b\s]+", " ", text).strip()
            if not text:
                stats["preprocess_filtered"] += 1
                continue
            
            # 3. 敏感话题过滤（新增！优先过滤违规数据）
            if is_sensitive(text):
                stats["sensitive_filtered"] += 1
                continue
            
            # 4. 口语化筛选（扩充关键词）
            if is_colloquial(text):
                stats["colloquial_filtered"] += 1
                continue
            
            # 5. 学术特征筛选（可选）
            if ACADEMIC_REQUIRE and not has_academic_features(text):
                stats["non_academic_filtered"] += 1
                continue
            
            # 6. MD5精确去重
            md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if md5_hash in md5_set:
                stats["md5_duplicated"] += 1
                continue
            md5_set.add(md5_hash)
            
            # 记录特征
            batch_buffer.append({
                "text": text,
                "original_data": original_data,
                "source_file": source_file,
                "md5": md5_hash,
                "has_academic_features": has_academic_features(text)
            })
            
            # 批量写入
            if len(batch_buffer) >= BATCH_SIZE_PREPROCESS:
                for data in batch_buffer:
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                batch_buffer = []
        
        if batch_buffer:
            for data in batch_buffer:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    stats["stage_time"]["preprocess_md5"] = round(time.time() - start_time, 2)
    remaining = stats["sampled_count"] - stats["preprocess_filtered"] - \
                stats["colloquial_filtered"] - stats["non_academic_filtered"] - stats["md5_duplicated"] - stats["sensitive_filtered"]
    logging.info(
        f"✅ 预处理+筛选完成 | 耗时：{stats['stage_time']['preprocess_md5']}秒 | "
        f"抽样后：{stats['sampled_count']} | 长度过滤：{stats['preprocess_filtered']} | "
        f"敏感话题过滤：{stats['sensitive_filtered']} | 口语化：{stats['colloquial_filtered']} | "
        f"无学术特征：{stats['non_academic_filtered']} | 完全重复（MD5）：{stats['md5_duplicated']} | 剩余：{remaining}"
    )
    return output_file

# ========== 2. Minhash LSH语义去重 ==========
def create_minhash_signature(text, num_perm=MINHASH_NUM_PERM):
    """2-gram Token捕捉中文语义"""
    minhash = MinHash(num_perm=num_perm)
    if len(text) < 2:
        grams = [text] if text else []
    else:
        grams = [text[i:i+2] for i in range(len(text)-1)]
    for gram in grams:
        token_hash = hashlib.sha256(gram.encode('utf-8')).hexdigest()
        minhash.update(token_hash.encode('utf-8'))
    return minhash

def minhash_lsh_deduplicate(input_file):
    start_time = time.time()
    logging.info(f"🚀 开始Minhash LSH语义去重（阈值{ LSH_THRESHOLD }）")
    
    texts = []
    data_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="读取预处理数据"):
            data = json.loads(line)
            texts.append(data["text"])
            data_list.append(data)
    
    if not texts:
        logging.warning("⚠️ MD5去重后无有效数据，跳过语义去重")
        return input_file
    
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    keep_indices = []
    duplicate_count = 0
    
    for i in tqdm(range(len(texts)), desc="MinHash去重"):
        minhash = create_minhash_signature(texts[i])
        similar_docs = lsh.query(minhash)
        if not similar_docs:
            lsh.insert(str(i), minhash)
            keep_indices.append(i)
        else:
            duplicate_count += 1
    
    stats["minhash_duplicated"] = duplicate_count
    
    output_file = os.path.join(INTERMEDIATE_PATH, "minhash_dedup_with_sensitive_filter.jsonl")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx in keep_indices:
            f_out.write(json.dumps(data_list[idx], ensure_ascii=False) + "\n")
    
    stats["stage_time"]["minhash_lsh"] = round(time.time() - start_time, 2)
    remaining = len(keep_indices)
    logging.info(f"✅ Minhash LSH语义去重完成 | 耗时：{stats['stage_time']['minhash_lsh']}秒 | "
                 f"语义相似重复：{stats['minhash_duplicated']} | 剩余：{remaining}")
    return output_file

# ========== 3. 古文检测函数（扩充关键词，提升识别准确率） ==========
def is_classic_chinese(text):
    """扩充古文关键词，精准判定真实古文（避免误判）"""
    # 扩充古文核心特征词（新增30+，覆盖虚词、实词、历史人物、典籍）
    classic_words = [
        # 虚词（核心）
        '之', '乎', '者', '也', '曰', '吾', '汝', '尔', '乃', '兮', '矣', '哉', '耶', '欤', '焉', '乎', '其', '而', '以', '于',
        # # 代词/名词
        '夫', '盖', '则', '且', '若', '何', '孰', '安', '孰与', '所以', '所', '可', '能', '必', '当', '应',
        # 历史朝代/人物
        '夏', '商', '周', '秦', '汉', '魏', '蜀', '吴', '晋', '隋', '唐', '宋', '元', '明', '清',
        '黄帝', '炎帝', '尧', '舜', '禹', '汤', '文王', '武王', '周公', '孔子', '孟子', '老子', '庄子', '墨子', '荀子', '韩非子',
        # 经典典籍
        '《论语》', '《孟子》', '《大学》', '《中庸》', '《诗经》', '《尚书》', '《礼记》', '《周易》', '《春秋》',
        '《道德经》', '《庄子》', '《墨子》', '《荀子》', '《韩非子》', '《史记》', '《汉书》', '《后汉书》', '《三国志》',
        # 古文句式特征词
        '呜呼', '嗟夫', '盖闻', '窃以为', '臣闻', '圣王', '贤君', '忠臣', '义士', '孝子', '烈女'
    ]
    # 现代文强特征词（快速排除）
    modern_words = ["手机", "互联网", "电脑", "微信", "支付宝", "快递", "高铁", "空调", "电视", "网络", "APP",
                   "微博", "抖音", "快手", "直播", "电商", "网购", "外卖", "打车", "共享单车", "5G", "WiFi"]
    
    # 含现代词直接判定为现代文
    for word in modern_words:
        if word in text:
            return False
    
    total_chars = len(text)
    if total_chars == 0:
        return False
    
    # 古文特征词密度阈值（保持2%，确保精准度）
    classic_char_count = 0
    for word in classic_words:
        classic_char_count += text.count(word)
    density_threshold = 0.02
    
    # 扩充古文句式匹配
    classic_patterns = [
        r'^[\u4e00-\u9fff]{1,5}曰', r'^昔者', r'^初', r'^当是时', r'^于是', r'^呜呼', r'^嗟夫',
        r'^盖闻', r'^窃以为', r'^臣闻', r'^圣王', r'^贤君', r'^忠臣', r'^义士'
    ]
    pattern_match = any(re.match(pattern, text) for pattern in classic_patterns)
    
    # 判定逻辑：密度达标 或 句式匹配
    is_classic = (classic_char_count / total_chars > density_threshold) or pattern_match
    return is_classic

# ========== 4. 困惑度分析与筛选（保持核心逻辑） ==========
def load_perplexity_model():
    logging.info("📥 加载模型计算困惑度（用于评估文本质量）")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    logging.info(f"✅ 模型加载完成 | 耗时：{round(time.time() - start_time, 2)}秒")
    return tokenizer, model

def calculate_perplexity_batch_optimized(texts, tokenizer, model):
    """批量计算困惑度（优化维度匹配）"""
    try:
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            return_tensors="pt"
        ).to(DEVICE)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            perplexities = []
            for i in range(batch_size):
                valid_mask = (input_ids[i] != tokenizer.pad_token_id)
                valid_token_ids = input_ids[i][valid_mask]
                
                if len(valid_token_ids) <= 1:
                    perplexities.append(float('inf'))
                    continue
                
                # 修正维度匹配逻辑
                shift_logits = logits[i, :-1, :].contiguous()
                shift_labels = valid_token_ids[1:].contiguous()
                valid_len = len(shift_labels)
                shift_logits = shift_logits[:valid_len, :]
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
            
            return perplexities
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.warning("⚠️ GPU内存不足，减小批量大小")
            half_size = len(texts) // 2
            if half_size >= 1:
                perplexities1 = calculate_perplexity_batch_optimized(texts[:half_size], tokenizer, model)
                perplexities2 = calculate_perplexity_batch_optimized(texts[half_size:], tokenizer, model)
                return perplexities1 + perplexities2
            else:
                return [calculate_perplexity(text, tokenizer, model) for text in texts]
        else:
            raise e
    except Exception as e:
        logging.warning(f"批量计算失败，回退到逐条计算: {str(e)}")
        return [calculate_perplexity(text, tokenizer, model) for text in texts]

def calculate_perplexity(text, tokenizer, model):
    """单文本困惑度计算（备用）"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            avg_loss = loss.mean()
            return torch.exp(avg_loss).item()
    except:
        return float('inf')

def analyze_perplexity_distribution_layered(input_file, sample_size=1000):
    """分层困惑度分布分析（精准分位数）"""
    logging.info("🔍 开始分层困惑度分布分析（现代文=低困惑度优质，古文=高困惑度真实）")
    tokenizer, model = load_perplexity_model()
    
    texts = []
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if len(lines) > sample_size:
            import random
            lines = random.sample(lines, sample_size)
        for line in lines:
            try:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if text and len(text) >= MIN_CHAR_LEN:
                    texts.append(text)
            except:
                continue
    
    if not texts:
        logging.error("❌ 没有有效的文本数据用于困惑度分析")
        return 100.0, 30.0, None, None
    
    logging.info(f"📊 将分析 {len(texts)} 条文本的困惑度分布")
    
    # 分类古文/现代文（扩充关键词后更精准）
    classic_texts = []
    modern_texts = []
    for text in texts:
        if is_classic_chinese(text):
            classic_texts.append(text)
        else:
            modern_texts.append(text)
    
    logging.info(f"📊 文本分类结果: 古文 {len(classic_texts)} 条, 现代文 {len(modern_texts)} 条")
    
    # 批量计算困惑度
    modern_perplexities = []
    for i in tqdm(range(0, len(modern_texts), BATCH_SIZE_PERPLEXITY), desc="计算现代文困惑度"):
        batch = modern_texts[i:i+BATCH_SIZE_PERPLEXITY]
        batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
        modern_perplexities.extend(batch_perplexities)
    
    classic_perplexities = []
    for i in tqdm(range(0, len(classic_texts), BATCH_SIZE_PERPLEXITY), desc="计算古文困惑度"):
        batch = classic_texts[i:i+BATCH_SIZE_PERPLEXITY]
        batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
        classic_perplexities.extend(batch_perplexities)
    
    # 处理有效数据（过滤异常值）
    modern_perplexities = np.array(modern_perplexities)
    classic_perplexities = np.array(classic_perplexities)
    
    modern_valid = modern_perplexities[modern_perplexities < 15000]
    classic_valid = classic_perplexities[classic_perplexities < 15000]
    
    if len(modern_valid) == 0:
        logging.warning("⚠️ 现代文困惑度计算失败，使用默认值")
        modern_valid = np.array([100.0])
    
    if len(classic_valid) == 0:
        logging.warning("⚠️ 古文困惑度计算失败，使用默认值")
        classic_valid = np.array([30.0])
    
    # 定义分位数（明确索引对应关系）
    percentiles = [0, 10, 20, 25, 30, 50, 75, 90, 95, 100]
    modern_percentiles = np.percentile(modern_valid, percentiles)
    classic_percentiles = np.percentile(classic_valid, percentiles)
    
    # 核心阈值配置（确保保留率50%）
    modern_threshold_idx = 6  # 75%分位数（过滤高困惑度现代文）
    classic_threshold_idx = 1  # 10%分位数（保留高困惑度古文）
    
    modern_threshold = modern_percentiles[modern_threshold_idx]
    classic_threshold = classic_percentiles[classic_threshold_idx]
    
    # 日志输出（清晰易懂）
    logging.info("📈 现代文困惑度分布（低困惑度=质量高、模型易理解）:")
    for p, val in zip(percentiles, modern_percentiles):
        logging.info(f"    {p}% 分位数: {val:.2f}")
    logging.info(f"🎯 现代文阈值: ≤{modern_threshold:.2f} ({percentiles[modern_threshold_idx]}%分位数)")
    
    logging.info("📈 古文困惑度分布（高困惑度=更真实、非现代改写）:")
    for p, val in zip(percentiles, classic_percentiles):
        logging.info(f"    {p}% 分位数: {val:.2f}")
    logging.info(f"🎯 古文阈值: ≥{classic_threshold:.2f} ({percentiles[classic_threshold_idx]}%分位数)")
    
    # 绘制分布图
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.hist(modern_valid, bins=50, alpha=0.7, label='Modern Chinese (Low Perplexity = High Quality)', color='skyblue', edgecolor='black')
        plt.hist(classic_valid, bins=50, alpha=0.7, label='Classic Chinese (High Perplexity = Authentic)', color='salmon', edgecolor='black')
        plt.axvline(modern_threshold, color='blue', linestyle='--', linewidth=2, label=f'Modern ≤ {modern_threshold:.2f} ({percentiles[modern_threshold_idx]}%tile)')
        plt.axvline(classic_threshold, color='red', linestyle='--', linewidth=2, label=f'Classic ≥ {classic_threshold:.2f} ({percentiles[classic_threshold_idx]}%tile)')
        plt.xlabel('Perplexity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Layered Perplexity Distribution (CLMMU/CEVAL Metric Priority)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_PATH, 'layered_perplexity_distribution_final.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📊 分层分布图已保存: {output_path}")
        
    except Exception as e:
        logging.warning(f"⚠️ 生成分布图失败: {str(e)}")
    
    return modern_threshold, classic_threshold, modern_percentiles, classic_percentiles

def determine_layered_thresholds(input_file):
    """确定分层阈值"""
    logging.info("🎯 开始确定分层困惑度阈值")
    
    try:
        modern_threshold, classic_threshold, modern_percentiles, classic_percentiles = analyze_perplexity_distribution_layered(input_file, sample_size=500)
        
        # 明确输出阈值和对应分位数
        percentiles = [0, 10, 20, 25, 30, 50, 75, 90, 95, 100]
        modern_p = percentiles[6]  # 75%分位数
        classic_p = percentiles[1]  # 10%分位数
        
        logging.info(f"🤖 现代文最终阈值: ≤{modern_threshold:.2f} ({modern_p}%分位数)")
        logging.info(f"📜 古文最终阈值: ≥{classic_threshold:.2f} ({classic_p}%分位数)")
        
        return modern_threshold, classic_threshold
        
    except Exception as e:
        logging.error(f"❌ 确定分层阈值失败: {str(e)}")
        logging.info("🔄 使用默认阈值: 现代文=80 (75%分位数), 古文=30 (10%分位数)")
        return 80.0, 30.0

def layered_perplexity_filter(input_file, modern_threshold, classic_threshold):
    """分层困惑度筛选"""
    start_time = time.time()
    
    # 明确阈值和解释
    percentiles = [0, 10, 20, 25, 30, 50, 75, 90, 95, 100]
    modern_p = percentiles[6]
    classic_p = percentiles[1]
    
    logging.info(f"🚀 开始分层困惑度筛选")
    logging.info(f"   - 现代文：≤{modern_threshold:.2f} ({modern_p}%分位数)，保留低困惑度优质数据")
    logging.info(f"   - 古文：≥{classic_threshold:.2f} ({classic_p}%分位数)，保留高困惑度真实古文")
    
    tokenizer, model = load_perplexity_model()
    
    kept_file = os.path.join(OUTPUT_PATH, "clmmu_kept_data_final.jsonl")
    filtered_file = os.path.join(OUTPUT_PATH, "clmmu_filtered_data_final.jsonl")
    classic_file = os.path.join(OUTPUT_PATH, "clmmu_classic_chinese_data_final.jsonl")
    
    batch_texts = []
    batch_data = []
    
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(kept_file, "w", encoding="utf-8") as f_kept, \
         open(filtered_file, "w", encoding="utf-8") as f_filtered, \
         open(classic_file, "w", encoding="utf-8") as f_classic:
        
        total_input = 0
        for line in f_in:
            total_input += 1
        f_in.seek(0)  # 重置文件指针
        
        for line in tqdm(f_in, desc="分层困惑度筛选", total=total_input):
            try:
                data = json.loads(line)
                batch_texts.append(data["text"])
                batch_data.append(data)
            except:
                continue
            
            if len(batch_texts) >= BATCH_SIZE_PERPLEXITY:
                perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
                
                for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
                    data_item["perplexity"] = round(perp, 2)
                    data_item["is_classic"] = is_classic_chinese(text)
                    
                    # 筛选逻辑
                    if data_item["is_classic"]:
                        # 古文：保留≥10%分位数的高困惑度数据，排除异常值
                        if perp >= classic_threshold and perp < 10000:
                            final_data = data_item["original_data"].copy()
                            final_data["cleaned_text"] = text
                            final_data["md5"] = data_item["md5"]
                            final_data["perplexity"] = data_item["perplexity"]
                            final_data["source_file"] = data_item["source_file"]
                            final_data["text_type"] = "classic_chinese"
                            final_data["has_academic_features"] = data_item.get("has_academic_features", False)
                            f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                            f_classic.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                            stats["final_kept"] += 1
                            stats["classic_chinese_kept"] += 1
                        else:
                            filtered_data = data_item.copy()
                            filtered_data["filter_reason"] = f"古文困惑度不达标（需≥{classic_threshold:.2f}，当前{perp:.2f}）"
                            f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
                            stats["perplexity_filtered"] += 1
                    else:
                        # 现代文：保留≤75%分位数的低困惑度数据，排除异常值
                        if perp <= modern_threshold and perp < 5000:
                            final_data = data_item["original_data"].copy()
                            final_data["cleaned_text"] = text
                            final_data["md5"] = data_item["md5"]
                            final_data["perplexity"] = data_item["perplexity"]
                            final_data["source_file"] = data_item["source_file"]
                            final_data["text_type"] = "modern_chinese"
                            final_data["has_academic_features"] = data_item.get("has_academic_features", False)
                            f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                            stats["final_kept"] += 1
                            stats["modern_chinese_kept"] += 1
                        else:
                            filtered_data = data_item.copy()
                            filtered_data["filter_reason"] = f"现代文困惑度不达标（需≤{modern_threshold:.2f}，当前{perp:.2f}）"
                            f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
                            stats["perplexity_filtered"] += 1
                
                batch_texts = []
                batch_data = []
        
        # 处理剩余数据
        if batch_texts:
            perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
            for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
                data_item["perplexity"] = round(perp, 2)
                data_item["is_classic"] = is_classic_chinese(text)
                
                if data_item["is_classic"]:
                    if perp >= classic_threshold and perp < 10000:
                        final_data = data_item["original_data"].copy()
                        final_data["cleaned_text"] = text
                        final_data["md5"] = data_item["md5"]
                        final_data["perplexity"] = data_item["perplexity"]
                        final_data["source_file"] = data_item["source_file"]
                        final_data["text_type"] = "classic_chinese"
                        f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                        f_classic.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                        stats["final_kept"] += 1
                        stats["classic_chinese_kept"] += 1
                    else:
                        filtered_data = data_item.copy()
                        filtered_data["filter_reason"] = f"古文困惑度不达标（需≥{classic_threshold:.2f}，当前{perp:.2f}）"
                        f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
                        stats["perplexity_filtered"] += 1
                else:
                    if perp <= modern_threshold and perp < 5000:
                        final_data = data_item["original_data"].copy()
                        final_data["cleaned_text"] = text
                        final_data["md5"] = data_item["md5"]
                        final_data["perplexity"] = data_item["perplexity"]
                        final_data["source_file"] = data_item["source_file"]
                        final_data["text_type"] = "modern_chinese"
                        f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                        stats["final_kept"] += 1
                        stats["modern_chinese_kept"] += 1
                    else:
                        filtered_data = data_item.copy()
                        filtered_data["filter_reason"] = f"现代文困惑度不达标（需≤{modern_threshold:.2f}，当前{perp:.2f}）"
                        f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
                        stats["perplexity_filtered"] += 1
    
    stats["stage_time"]["perplexity"] = round(time.time() - start_time, 2)
    logging.info(f"✅ 分层困惑度筛选完成 | 耗时：{stats['stage_time']['perplexity']}秒")
    logging.info(f"📊 分层统计 - 现代文保留: {stats['modern_chinese_kept']} | 古文保留: {stats['classic_chinese_kept']}")
    logging.info(f"📊 总计保留: {stats['final_kept']} | 过滤: {stats['perplexity_filtered']}")
    
    return kept_file, filtered_file, classic_file

# ========== 主函数 ==========
def main():
    global monitor_running
    start_time = time.time()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_PATH, "clmmu_filter_log_final.log"), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("🎉 启动CLMMU/CEVAL筛选流程（最终版）")
    logging.info("📋 核心配置（按需求调整）:")
    logging.info(f"   - 语义去重阈值: {LSH_THRESHOLD}（减少冗余）")
    logging.info(f"   - 现代文困惑度: ≤75%分位数（低困惑度=高质量）")
    logging.info(f"   - 古文困惑度: ≥10%分位数（高困惑度=真实古文）")
    logging.info(f"   - 口语关键词: 扩充至{len(COLLOQUIAL_WORDS)}个（强化学术过滤）")
    logging.info(f"   - 古文关键词: 扩充至{len([w for w in is_classic_chinese.__code__.co_consts if isinstance(w, str) and len(w) <= 10])}个（精准识别）")
    logging.info(f"   - 敏感话题过滤: 启用（覆盖色情、暴力、毒品等）")
    logging.info(f"   - 已删除: 学科过滤、题型适配、事实准确性检测")
    
    # 启动监控线程
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    
    try:
        # 1. 预处理+MD5去重+基础筛选（含敏感过滤）
        preprocessed_file = preprocess_and_md5_deduplicate()
        
        # 2. Minhash LSH语义去重
        minhash_file = minhash_lsh_deduplicate(preprocessed_file)
        
        # 3. 确定分层阈值
        modern_threshold, classic_threshold = determine_layered_thresholds(minhash_file)
        
        # 4. 分层困惑度筛选
        kept_file, filtered_file, classic_file = layered_perplexity_filter(minhash_file, modern_threshold, classic_threshold)
        
        # 停止监控
        monitor_running = False
        monitor.join()
        
        # 输出最终统计
        total_time = round(time.time() - start_time, 2)
        logging.info("\n" + "="*80)
        logging.info("📊 CLMMU/CEVAL筛选最终统计报告（最终版）")
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
        logging.info(f"   - 现代文困惑度: ≤{modern_threshold:.2f} (75%分位数，低困惑度优质)")
        logging.info(f"   - 古文困惑度: ≥{classic_threshold:.2f} (10%分位数，高困惑度真实)")
        logging.info(f"   - 语义去重: {LSH_THRESHOLD}")
        logging.info(f"📌 性能统计:")
        logging.info(f"   - 总耗时: {total_time}秒 ({total_time/60:.1f}分钟)")
        logging.info("\n📁 输出文件:")
        logging.info(f"✅ 高质量数据（最终版）: {kept_file}")
        logging.info(f"📜 古文专门数据: {classic_file}")
        logging.info(f"❌ 过滤数据详情: {filtered_file}")
        logging.info(f"📊 筛选日志: {os.path.join(OUTPUT_PATH, 'clmmu_filter_log_final.log')}")
        logging.info("="*80)
        
        # 保留比例校准提示
        retention_ratio = stats['final_kept']/stats['sampled_count']*100
        if retention_ratio < 45:
            logging.warning(f"⚠️ 保留比例过低（{retention_ratio:.1f}%），建议适度放松：")
            logging.warning(f"   1. 语义去重阈值从0.76→0.73")
            logging.warning(f"   2. 现代文困惑度分位数从75%→80%")
        elif retention_ratio > 55:
            logging.warning(f"⚠️ 保留比例过高（{retention_ratio:.1f}%），建议适度收紧：")
            logging.warning(f"   1. 语义去重阈值从0.76→0.78")
            logging.warning(f"   2. 现代文困惑度分位数从75%→70%")
        else:
            logging.info("✅ 保留比例达标（45%-55%），数据质量与数量平衡良好！")
        
    except Exception as e:
        logging.error(f"❌ 流程执行失败: {str(e)}", exc_info=True)
    finally:
        monitor_running = False
        monitor.join(timeout=5)
        logging.info("🔚 CLMMU/CEVAL筛选流程结束（最终版）")

if __name__ == "__main__":
    main()