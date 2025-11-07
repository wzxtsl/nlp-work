# import os
# import json
# import logging
# import mmap
# import numpy as np
# import psutil
# import threading
# import re
# from tqdm import tqdm
# from config import (
#     INPUT_DIR, SAMPLING_ENABLE, SAMPLE_RATIO, MAX_SAMPLE_COUNT,
#     COLLOQUIAL_WORDS, SENSITIVE_KEYWORDS, ACADEMIC_PATTERNS,
#     CLASSIC_CHINESE_WORDS, MODERN_CHINESE_WORDS, CLASSIC_DENSITY_THRESHOLD
# )

# # 全局统计变量（跨文件共享）
# stats = {
#     "total_input": 0,
#     "sampled_count": 0,
#     "preprocess_filtered": 0,
#     "colloquial_filtered": 0,
#     "non_academic_filtered": 0,
#     "md5_duplicated": 0,
#     "minhash_duplicated": 0,
#     "sensitive_filtered": 0,
#     "perplexity_filtered": 0,
#     "final_kept": 0,
#     "classic_chinese_kept": 0,
#     "modern_chinese_kept": 0,
#     "stage_time": {}
# }

# # 监控线程全局变量
# monitor_running = True
# gpu_util = 0
# cpu_mem = 0

# def get_gpu_utilization():
#     """获取GPU利用率（仅支持NVIDIA显卡）"""
#     try:
#         result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
#         return int(result.strip().split("\n")[0]) if result else 0
#     except:
#         return 0

# def get_cpu_memory():
#     """获取当前进程CPU内存占用（MB）"""
#     process = psutil.Process(os.getpid())
#     return round(process.memory_info().rss / (1024 * 1024), 2)

# def monitor_thread():
#     """监控线程：每30秒输出GPU/内存状态"""
#     logging.info("🔍 监控线程启动（每30秒更新GPU/内存状态）")
#     while monitor_running:
#         global gpu_util, cpu_mem
#         gpu_util = get_gpu_utilization()
#         cpu_mem = get_cpu_memory()
        
#         # 计算已处理进度
#         total_processed = stats["sampled_count"] - stats["preprocess_filtered"] - \
#                           stats["colloquial_filtered"] - stats["non_academic_filtered"] - \
#                           stats["md5_duplicated"] - stats["minhash_duplicated"] - stats["sensitive_filtered"]
#         progress = (total_processed / stats["sampled_count"] * 100) if stats["sampled_count"] > 0 else 0.0
        
#         logging.info(
#             f"📊 监控状态 - GPU利用率：{gpu_util}% | CPU内存：{cpu_mem}MB | "
#             f"总输入：{stats['total_input']} | 抽样后：{stats['sampled_count']} | 已处理：{total_processed} | 进度：{progress:.1f}%"
#         )
#         threading.Event().wait(30)  # 更稳定的休眠
#     logging.info("🔍 监控线程停止")

# def load_jsonl_files_with_sampling():
#     """加载JSONL文件（支持抽样模式）"""
#     jsonl_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
#     if not jsonl_files:
#         raise ValueError(f"❌ 输入目录 {INPUT_DIR} 下无JSONL文件，请检查路径！")
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
                    
#                     # 抽样逻辑
#                     if SAMPLING_ENABLE:
#                         if sampled_count >= MAX_SAMPLE_COUNT:
#                             logging.info(f"✅ 抽样完成：已抽取 {sampled_count} 条样本（达到最大限制）")
#                             return
#                         if np.random.random() > SAMPLE_RATIO:
#                             continue
#                         sampled_count += 1
#                         stats["sampled_count"] = sampled_count
                    
#                     yield {"text": text, "original_data": data, "source_file": file_name}
#                 except Exception as e:
#                     stats["preprocess_filtered"] += 1
#                     continue
    
#     logging.info(f"✅ 所有文件加载完成 {'（抽样模式）' if SAMPLING_ENABLE else '（全量模式）'}")
#     logging.info(f"📊 加载统计：总输入 {stats['total_input']} 条 | 抽样后 {stats['sampled_count']} 条")

# def is_colloquial(text):
#     """检测文本是否为口语化（关键词+句式匹配）"""
#     # 1. 口语关键词匹配
#     for word in COLLOQUIAL_WORDS:
#         if word in text:
#             return True
#     # 2. 连续标点匹配（3个以上）
#     if re.search(r"[！？。,，；;：:]{3,}", text):
#         return True
#     # 3. 口语化句式匹配
#     colloquial_patterns = [
#         r"[我你他她它]（们）?[也都还就才又再]?[不没没什么没什么大不了]",
#         r"[这那哪]（个些）?[也都还就才又再]?[不没没什么没什么大不了]",
#         r"^[哈哈嘿嘿嘻嘻呵呵]+"
#     ]
#     if any(re.search(pattern, text) for pattern in colloquial_patterns):
#         return True
#     return False

# def is_sensitive(text):
#     """检测文本是否包含敏感话题"""
#     # 1. 敏感关键词匹配
#     for category, words in SENSITIVE_KEYWORDS.items():
#         for word in words:
#             if word in text:
#                 return True
#     # 2. 敏感句式匹配
#     sensitive_patterns = [
#         r"出售.*(色情|AV|三级片)",
#         r"(嫖娼|卖淫|性交易).*(价格|联系方式|地点)",
#         r"(杀人|抢劫|绑架).*(方法|教程|工具)",
#         r"(毒品|大麻|冰毒).*(购买|出售|运输)",
#         r"(台独|港独|疆独).*(支持|宣传|分裂)"
#     ]
#     if any(re.search(pattern, text, re.IGNORECASE) for pattern in sensitive_patterns):
#         return True
#     return False

# def has_academic_features(text):
#     """检测文本是否包含学术特征"""
#     return any(re.search(pattern, text) for pattern in ACADEMIC_PATTERNS)

# def is_classic_chinese(text):
#     """检测文本是否为古文（关键词密度+句式匹配）"""
#     # 1. 含现代词直接排除
#     for word in MODERN_CHINESE_WORDS:
#         if word in text:
#             return False
    
#     total_chars = len(text)
#     if total_chars == 0:
#         return False
    
#     # 2. 计算古文特征词密度
#     classic_char_count = 0
#     for word in CLASSIC_CHINESE_WORDS:
#         classic_char_count += text.count(word)
#     density = classic_char_count / total_chars
    
#     # 3. 古文句式匹配
#     classic_patterns = [
#         r'^[\u4e00-\u9fff]{1,5}曰', r'^昔者', r'^初', r'^当是时', r'^于是', r'^呜呼', r'^嗟夫',
#         r'^盖闻', r'^窃以为', r'^臣闻', r'^圣王', r'^贤君', r'^忠臣', r'^义士'
#     ]
#     pattern_match = any(re.match(pattern, text) for pattern in classic_patterns)
    
#     # 4. 判定逻辑：密度达标 或 句式匹配
#     return (density > CLASSIC_DENSITY_THRESHOLD) or pattern_match

import os
import json
import mmap
import time
import psutil
import threading
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    MODEL_ID, DEVICE, MAX_SEQ_LENGTH, SAMPLING_ENABLE,
    SAMPLE_RATIO, MAX_SAMPLE_COUNT, MIN_CHAR_LEN
)

# 全局监控变量
monitor_running = True
gpu_util = 0
cpu_mem = 0

def get_gpu_utilization():
    """获取GPU利用率"""
    try:
        result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
        return int(result.strip().split("\n")[0]) if result else 0
    except:
        return 0

def get_cpu_memory():
    """获取CPU内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)

def monitor_thread(stats):
    """监控线程：每30秒输出GPU/内存状态"""
    global monitor_running, gpu_util, cpu_mem
    import logging
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

def load_jsonl_files_with_sampling(input_dir, stats):
    """加载JSONL文件并支持抽样"""
    import logging
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

def load_perplexity_model():
    """加载困惑度计算模型"""
    import logging
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
        import logging
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
        import logging
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

def plot_perplexity_distribution(modern_valid, classic_valid, modern_threshold, classic_threshold, percentiles, modern_idx, classic_idx, output_path):
    """绘制困惑度分布图"""
    try:
        plt.figure(figsize=(12, 8))
        plt.hist(modern_valid, bins=50, alpha=0.7, label='Modern Chinese (Low Perplexity = High Quality)', color='skyblue', edgecolor='black')
        plt.hist(classic_valid, bins=50, alpha=0.7, label='Classic Chinese (High Perplexity = Authentic)', color='salmon', edgecolor='black')
        plt.axvline(modern_threshold, color='blue', linestyle='--', linewidth=2, label=f'Modern ≤ {modern_threshold:.2f} ({percentiles[modern_idx]}%tile)')
        plt.axvline(classic_threshold, color='red', linestyle='--', linewidth=2, label=f'Classic ≥ {classic_threshold:.2f} ({percentiles[classic_idx]}%tile)')
        plt.xlabel('Perplexity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Layered Perplexity Distribution (CLMMU/CEVAL Metric Priority)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        import logging
        logging.warning(f"⚠️ 生成分布图失败: {str(e)}")
        return False