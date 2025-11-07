# import os
# import json
# import time
# import numpy as np
# import torch
# import logging
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from config import (
#     OUTPUT_PATH, MODEL_ID, DEVICE, MAX_SEQ_LENGTH, BATCH_SIZE_PERPLEXITY,MIN_CHAR_LEN,
#     MODERN_PERPLEXITY_PERCENTILE, CLASSIC_PERPLEXITY_PERCENTILE, PERPLEXITY_MAX_LIMIT
# )
# from utils import stats, is_classic_chinese

# def load_perplexity_model():
#     """加载困惑度计算模型（GPT2-Chinese）"""
#     logging.info("📥 加载模型计算困惑度（用于评估文本质量）")
#     start_time = time.time()
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token  # 补全pad_token
#     model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
#     model.eval()  # 推理模式
#     logging.info(f"✅ 模型加载完成 | 耗时：{round(time.time() - start_time, 2)}秒")
#     return tokenizer, model

# def calculate_perplexity_batch_optimized(texts, tokenizer, model):
#     """批量计算困惑度（优化维度匹配和内存占用）"""
#     try:
#         # 文本编码
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
        
#         with torch.no_grad():  # 禁用梯度计算
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
            
#             perplexities = []
#             for i in range(batch_size):
#                 # 过滤pad_token
#                 valid_mask = (input_ids[i] != tokenizer.pad_token_id)
#                 valid_token_ids = input_ids[i][valid_mask]
                
#                 if len(valid_token_ids) <= 1:
#                     perplexities.append(float('inf'))
#                     continue
                
#                 # 修正维度匹配（shift logits和labels）
#                 shift_logits = logits[i, :-1, :].contiguous()
#                 shift_labels = valid_token_ids[1:].contiguous()
#                 valid_len = len(shift_labels)
#                 shift_logits = shift_logits[:valid_len, :]  # 截断到有效长度
                
#                 # 计算交叉熵损失
#                 loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
#                 loss = loss_fct(
#                     shift_logits.view(-1, shift_logits.size(-1)), 
#                     shift_labels.view(-1)
#                 )
                
#                 perplexity = torch.exp(loss).item()
#                 perplexities.append(perplexity)
            
#             return perplexities
            
#     except RuntimeError as e:
#         # GPU内存不足时自动减小批量
#         if "out of memory" in str(e):
#             logging.warning("⚠️ GPU内存不足，减小批量大小")
#             half_size = len(texts) // 2
#             if half_size >= 1:
#                 perplexities1 = calculate_perplexity_batch_optimized(texts[:half_size], tokenizer, model)
#                 perplexities2 = calculate_perplexity_batch_optimized(texts[half_size:], tokenizer, model)
#                 return perplexities1 + perplexities2
#             else:
#                 return [calculate_perplexity_single(text, tokenizer, model) for text in texts]
#         else:
#             raise e
#     except Exception as e:
#         logging.warning(f"批量计算失败，回退到逐条计算: {str(e)}")
#         return [calculate_perplexity_single(text, tokenizer, model) for text in texts]

# def calculate_perplexity_single(text, tokenizer, model):
#     """单文本困惑度计算（备用方案）"""
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
#     """分层困惑度分布分析（分别计算古文/现代文分位数）"""
#     logging.info("🔍 开始分层困惑度分布分析（现代文=低困惑度优质，古文=高困惑度真实）")
#     tokenizer, model = load_perplexity_model()
    
#     # 读取样本数据
#     texts = []
#     with open(input_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         if len(lines) > sample_size:
#             import random
#             lines = random.sample(lines, sample_size)  # 随机抽样
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
#         return 100.0, 30.0  # 默认阈值
    
#     logging.info(f"📊 将分析 {len(texts)} 条文本的困惑度分布")
    
#     # 分类古文/现代文
#     classic_texts = []
#     modern_texts = []
#     for text in texts:
#         if is_classic_chinese(text):
#             classic_texts.append(text)
#         else:
#             modern_texts.append(text)
    
#     logging.info(f"📊 文本分类结果: 古文 {len(classic_texts)} 条, 现代文 {len(modern_texts)} 条")
    
#     # 批量计算困惑度
#     modern_perplexities = []
#     for i in tqdm(range(0, len(modern_texts), BATCH_SIZE_PERPLEXITY), desc="计算现代文困惑度"):
#         batch = modern_texts[i:i+BATCH_SIZE_PERPLEXITY]
#         batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
#         modern_perplexities.extend(batch_perplexities)
    
#     classic_perplexities = []
#     for i in tqdm(range(0, len(classic_texts), BATCH_SIZE_PERPLEXITY), desc="计算古文困惑度"):
#         batch = classic_texts[i:i+BATCH_SIZE_PERPLEXITY]
#         batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
#         classic_perplexities.extend(batch_perplexities)
    
#     # 过滤异常值（超出上限的视为无效）
#     modern_valid = np.array([p for p in modern_perplexities if p < PERPLEXITY_MAX_LIMIT])
#     classic_valid = np.array([p for p in classic_perplexities if p < PERPLEXITY_MAX_LIMIT])
    
#     # 处理空数据
#     if len(modern_valid) == 0:
#         logging.warning("⚠️ 现代文困惑度计算失败，使用默认值")
#         modern_valid = np.array([100.0])
#     if len(classic_valid) == 0:
#         logging.warning("⚠️ 古文困惑度计算失败，使用默认值")
#         classic_valid = np.array([30.0])
    
#     # 计算目标分位数
#     modern_threshold = np.percentile(modern_valid, MODERN_PERPLEXITY_PERCENTILE)
#     classic_threshold = np.percentile(classic_valid, CLASSIC_PERPLEXITY_PERCENTILE)
    
#     # 输出分布日志
#     logging.info("📈 现代文困惑度分布（低困惑度=质量高、模型易理解）:")
#     percentiles = [0, 10, 25, 50, 75, 90, 95, 100]
#     for p in percentiles:
#         logging.info(f"    {p}% 分位数: {np.percentile(modern_valid, p):.2f}")
#     logging.info(f"🎯 现代文阈值: ≤{modern_threshold:.2f} ({MODERN_PERPLEXITY_PERCENTILE}%分位数)")
    
#     logging.info("📈 古文困惑度分布（高困惑度=更真实、非现代改写）:")
#     for p in percentiles:
#         logging.info(f"    {p}% 分位数: {np.percentile(classic_valid, p):.2f}")
#     logging.info(f"🎯 古文阈值: ≥{classic_threshold:.2f} ({CLASSIC_PERPLEXITY_PERCENTILE}%分位数)")
    
#     # 绘制分布图（可选）
#     try:
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(12, 8))
#         plt.hist(modern_valid, bins=50, alpha=0.7, label='Modern Chinese (Low Perplexity = High Quality)', color='skyblue', edgecolor='black')
#         plt.hist(classic_valid, bins=50, alpha=0.7, label='Classic Chinese (High Perplexity = Authentic)', color='salmon', edgecolor='black')
#         plt.axvline(modern_threshold, color='blue', linestyle='--', linewidth=2, label=f'Modern ≤ {modern_threshold:.2f} ({MODERN_PERPLEXITY_PERCENTILE}%tile)')
#         plt.axvline(classic_threshold, color='red', linestyle='--', linewidth=2, label=f'Classic ≥ {classic_threshold:.2f} ({CLASSIC_PERPLEXITY_PERCENTILE}%tile)')
#         plt.xlabel('Perplexity', fontsize=12)
#         plt.ylabel('Frequency', fontsize=12)
#         plt.title('Layered Perplexity Distribution (CLMMU/CEVAL Metric Priority)', fontsize=14, fontweight='bold')
#         plt.legend(fontsize=10)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
        
#         output_path = os.path.join(OUTPUT_PATH, 'layered_perplexity_distribution_final.png')
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         logging.info(f"📊 分层分布图已保存: {output_path}")
#     except Exception as e:
#         logging.warning(f"⚠️ 生成分布图失败: {str(e)}")
    
#     return modern_threshold, classic_threshold

# def determine_layered_thresholds(input_file):
#     """确定古文/现代文的困惑度筛选阈值"""
#     logging.info("🎯 开始确定分层困惑度阈值")
#     try:
#         modern_threshold, classic_threshold = analyze_perplexity_distribution_layered(input_file, sample_size=500)
#         logging.info(f"🤖 现代文最终阈值: ≤{modern_threshold:.2f} ({MODERN_PERPLEXITY_PERCENTILE}%分位数)")
#         logging.info(f"📜 古文最终阈值: ≥{classic_threshold:.2f} ({CLASSIC_PERPLEXITY_PERCENTILE}%分位数)")
#         return modern_threshold, classic_threshold
#     except Exception as e:
#         logging.error(f"❌ 确定分层阈值失败: {str(e)}")
#         logging.info(f"🔄 使用默认阈值: 现代文=80 ({MODERN_PERPLEXITY_PERCENTILE}%分位数), 古文=30 ({CLASSIC_PERPLEXITY_PERCENTILE}%分位数)")
#         return 80.0, 30.0

# def layered_perplexity_filter(input_file, modern_threshold, classic_threshold):
#     """分层困惑度筛选（现代文低困惑度 + 古文高困惑度）"""
#     start_time = time.time()
#     logging.info(f"🚀 开始分层困惑度筛选")
#     logging.info(f"   - 现代文：≤{modern_threshold:.2f} ({MODERN_PERPLEXITY_PERCENTILE}%分位数)，保留低困惑度优质数据")
#     logging.info(f"   - 古文：≥{classic_threshold:.2f} ({CLASSIC_PERPLEXITY_PERCENTILE}%分位数)，保留高困惑度真实古文")
    
#     tokenizer, model = load_perplexity_model()
    
#     # 输出文件路径
#     kept_file = os.path.join(OUTPUT_PATH, "clmmu_kept_data_final.jsonl")
#     filtered_file = os.path.join(OUTPUT_PATH, "clmmu_filtered_data_final.jsonl")
#     classic_file = os.path.join(OUTPUT_PATH, "clmmu_classic_chinese_data_final.jsonl")
    
#     batch_texts = []
#     batch_data = []
    
#     # 统计总输入数
#     with open(input_file, "r", encoding="utf-8") as f_in:
#         total_input = sum(1 for _ in f_in)
    
#     # 筛选流程
#     with open(input_file, "r", encoding="utf-8") as f_in, \
#          open(kept_file, "w", encoding="utf-8") as f_kept, \
#          open(filtered_file, "w", encoding="utf-8") as f_filtered, \
#          open(classic_file, "w", encoding="utf-8") as f_classic:
        
#         f_in.seek(0)  # 重置文件指针
#         for line in tqdm(f_in, desc="分层困惑度筛选", total=total_input):
#             try:
#                 data = json.loads(line)
#                 batch_texts.append(data["text"])
#                 batch_data.append(data)
#             except:
#                 continue
            
#             # 批量处理
#             if len(batch_texts) >= BATCH_SIZE_PERPLEXITY:
#                 perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
                
#                 for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
#                     data_item["perplexity"] = round(perp, 2)
#                     data_item["is_classic"] = is_classic_chinese(text)
                    
#                     # 筛选逻辑
#                     if data_item["is_classic"]:
#                         # 古文：保留≥阈值且无异常值
#                         if perp >= classic_threshold and perp < PERPLEXITY_MAX_LIMIT:
#                             final_data = data_item["original_data"].copy()
#                             final_data.update({
#                                 "cleaned_text": text,
#                                 "md5": data_item["md5"],
#                                 "perplexity": data_item["perplexity"],
#                                 "source_file": data_item["source_file"],
#                                 "text_type": "classic_chinese",
#                                 "has_academic_features": data_item.get("has_academic_features", False)
#                             })
#                             f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                             f_classic.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                             stats["final_kept"] += 1
#                             stats["classic_chinese_kept"] += 1
#                         else:
#                             filtered_data = data_item.copy()
#                             filtered_data["filter_reason"] = f"古文困惑度不达标（需≥{classic_threshold:.2f}，当前{perp:.2f}）"
#                             f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                             stats["perplexity_filtered"] += 1
#                     else:
#                         # 现代文：保留≤阈值且无异常值
#                         if perp <= modern_threshold and perp < PERPLEXITY_MAX_LIMIT / 3:  # 现代文阈值更严格
#                             final_data = data_item["original_data"].copy()
#                             final_data.update({
#                                 "cleaned_text": text,
#                                 "md5": data_item["md5"],
#                                 "perplexity": data_item["perplexity"],
#                                 "source_file": data_item["source_file"],
#                                 "text_type": "modern_chinese",
#                                 "has_academic_features": data_item.get("has_academic_features", False)
#                             })
#                             f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                             stats["final_kept"] += 1
#                             stats["modern_chinese_kept"] += 1
#                         else:
#                             filtered_data = data_item.copy()
#                             filtered_data["filter_reason"] = f"现代文困惑度不达标（需≤{modern_threshold:.2f}，当前{perp:.2f}）"
#                             f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                             stats["perplexity_filtered"] += 1
                
#                 # 重置批量缓存
#                 batch_texts = []
#                 batch_data = []
        
#         # 处理剩余数据
#         if batch_texts:
#             perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
#             for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
#                 # 重复上述筛选逻辑（略，与批量处理一致）
#                 data_item["perplexity"] = round(perp, 2)
#                 data_item["is_classic"] = is_classic_chinese(text)
#                 if data_item["is_classic"]:
#                     if perp >= classic_threshold and perp < PERPLEXITY_MAX_LIMIT:
#                         final_data = data_item["original_data"].copy()
#                         final_data.update({
#                             "cleaned_text": text, "md5": data_item["md5"], "perplexity": data_item["perplexity"],
#                             "source_file": data_item["source_file"], "text_type": "classic_chinese",
#                             "has_academic_features": data_item.get("has_academic_features", False)
#                         })
#                         f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                         f_classic.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                         stats["final_kept"] += 1
#                         stats["classic_chinese_kept"] += 1
#                     else:
#                         filtered_data = data_item.copy()
#                         filtered_data["filter_reason"] = f"古文困惑度不达标（需≥{classic_threshold:.2f}，当前{perp:.2f}）"
#                         f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                         stats["perplexity_filtered"] += 1
#                 else:
#                     if perp <= modern_threshold and perp < PERPLEXITY_MAX_LIMIT / 3:
#                         final_data = data_item["original_data"].copy()
#                         final_data.update({
#                             "cleaned_text": text, "md5": data_item["md5"], "perplexity": data_item["perplexity"],
#                             "source_file": data_item["source_file"], "text_type": "modern_chinese",
#                             "has_academic_features": data_item.get("has_academic_features", False)
#                         })
#                         f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
#                         stats["final_kept"] += 1
#                         stats["modern_chinese_kept"] += 1
#                     else:
#                         filtered_data = data_item.copy()
#                         filtered_data["filter_reason"] = f"现代文困惑度不达标（需≤{modern_threshold:.2f}，当前{perp:.2f}）"
#                         f_filtered.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")
#                         stats["perplexity_filtered"] += 1
    
#     # 统计结果
#     stats["stage_time"]["perplexity"] = round(time.time() - start_time, 2)
#     logging.info(f"✅ 分层困惑度筛选完成 | 耗时：{stats['stage_time']['perplexity']}秒")
#     logging.info(f"📊 分层统计 - 现代文保留: {stats['modern_chinese_kept']} | 古文保留: {stats['classic_chinese_kept']}")
#     logging.info(f"📊 总计保留: {stats['final_kept']} | 过滤: {stats['perplexity_filtered']}")
    
#     return kept_file, filtered_file, classic_file

import re
import numpy as np
from tqdm import tqdm
import time

from config import (
    COLLOQUIAL_WORDS, SENSITIVE_KEYWORDS, ACADEMIC_PATTERNS, ACADEMIC_REQUIRE,
    MIN_CHAR_LEN, MAX_CHAR_LEN, PERCENTILES, MODERN_PERPLEXITY_PERCENTILE,
    CLASSIC_PERPLEXITY_PERCENTILE, MAX_MODERN_PERPLEXITY, MAX_CLASSIC_PERPLEXITY,
    CLASSIC_CHINESE_WORDS, MODERN_CHINESE_WORDS, CLASSIC_CHINESE_DENSITY_THRESHOLD,
    OUTPUT_PATH, BATCH_SIZE_PERPLEXITY
)
from utils import (
    load_perplexity_model, calculate_perplexity_batch_optimized,
    plot_perplexity_distribution
)

def is_colloquial(text):
    """检测口语化文本（扩充关键词和句式匹配）"""
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
    """检测敏感话题文本"""
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
    """检测学术特征"""
    return any(re.search(pattern, text) for pattern in ACADEMIC_PATTERNS)

def preprocess_and_filter(data_generator, stats):
    """预处理+基础筛选（长度、敏感词、口语化、学术特征）"""
    import logging
    start_time = time.time()
    logging.info("🚀 开始预处理 + 基础筛选（含敏感话题过滤）")
    
    filtered_data = []
    
    for item in tqdm(data_generator, desc="预处理+筛选", total=stats["sampled_count"] if stats["sampled_count"] > 0 else None):
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
        
        # 3. 敏感话题过滤
        if is_sensitive(text):
            stats["sensitive_filtered"] += 1
            continue
        
        # 4. 口语化筛选
        if is_colloquial(text):
            stats["colloquial_filtered"] += 1
            continue
        
        # 5. 学术特征筛选（可选）
        if ACADEMIC_REQUIRE and not has_academic_features(text):
            stats["non_academic_filtered"] += 1
            continue
        
        # 记录特征
        filtered_data.append({
            "text": text,
            "original_data": original_data,
            "source_file": source_file,
            "has_academic_features": has_academic_features(text)
        })
    
    remaining = len(filtered_data)
    logging.info(
        f"✅ 预处理+筛选完成 | 长度过滤：{stats['preprocess_filtered']}条 | "
        f"敏感话题过滤：{stats['sensitive_filtered']}条 | 口语化：{stats['colloquial_filtered']}条 | "
        f"无学术特征：{stats['non_academic_filtered']}条 | 剩余：{remaining}条"
    )
    return filtered_data

def is_classic_chinese(text):
    """检测古文（基于关键词密度和句式）"""
    # 含现代词直接判定为现代文
    for word in MODERN_CHINESE_WORDS:
        if word in text:
            return False
    
    total_chars = len(text)
    if total_chars == 0:
        return False
    
    # 计算古文特征词密度
    classic_char_count = 0
    for word in CLASSIC_CHINESE_WORDS:
        classic_char_count += text.count(word)
    density = classic_char_count / total_chars
    
    # 古文句式匹配
    classic_patterns = [
        r'^[\u4e00-\u9fff]{1,5}曰', r'^昔者', r'^初', r'^当是时', r'^于是', r'^呜呼', r'^嗟夫',
        r'^盖闻', r'^窃以为', r'^臣闻', r'^圣王', r'^贤君', r'^忠臣', r'^义士'
    ]
    pattern_match = any(re.match(pattern, text) for pattern in classic_patterns)
    
    # 判定逻辑：密度达标 或 句式匹配
    return (density > CLASSIC_CHINESE_DENSITY_THRESHOLD) or pattern_match

def analyze_perplexity_distribution(minhash_data, sample_size=1000):
    """分析困惑度分布，返回分层阈值"""
    import logging
    logging.info("🔍 开始分层困惑度分布分析（现代文=低困惑度优质，古文=高困惑度真实）")
    
    # 提取有效文本
    texts = [item["text"] for item in minhash_data if len(item["text"]) >= MIN_CHAR_LEN]
    if len(texts) > sample_size:
        import random
        texts = random.sample(texts, sample_size)
    
    if not texts:
        logging.error("❌ 没有有效的文本数据用于困惑度分析")
        return 80.0, 30.0  # 默认阈值
    
    logging.info(f"📊 将分析 {len(texts)} 条文本的困惑度分布")
    
    # 分类古文/现代文
    classic_texts = []
    modern_texts = []
    for text in texts:
        if is_classic_chinese(text):
            classic_texts.append(text)
        else:
            modern_texts.append(text)
    
    logging.info(f"📊 文本分类结果: 古文 {len(classic_texts)} 条, 现代文 {len(modern_texts)} 条")
    
    # 加载模型
    tokenizer, model = load_perplexity_model()
    
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
    
    # 计算分位数
    modern_percentiles = np.percentile(modern_valid, PERCENTILES)
    classic_percentiles = np.percentile(classic_valid, PERCENTILES)
    
    # 获取目标阈值
    modern_threshold = modern_percentiles[MODERN_PERPLEXITY_PERCENTILE]
    classic_threshold = classic_percentiles[CLASSIC_PERPLEXITY_PERCENTILE]
    
    # 日志输出
    logging.info("📈 现代文困惑度分布（低困惑度=质量高、模型易理解）:")
    for p, val in zip(PERCENTILES, modern_percentiles):
        logging.info(f"    {p}% 分位数: {val:.2f}")
    logging.info(f"🎯 现代文阈值: ≤{modern_threshold:.2f} ({PERCENTILES[MODERN_PERPLEXITY_PERCENTILE]}%分位数)")
    
    logging.info("📈 古文困惑度分布（高困惑度=更真实、非现代改写）:")
    for p, val in zip(PERCENTILES, classic_percentiles):
        logging.info(f"    {p}% 分位数: {val:.2f}")
    logging.info(f"🎯 古文阈值: ≥{classic_threshold:.2f} ({PERCENTILES[CLASSIC_PERPLEXITY_PERCENTILE]}%分位数)")
    
    # 绘制分布图
    plot_path = f"{OUTPUT_PATH}/layered_perplexity_distribution_final.png"
    if plot_perplexity_distribution(modern_valid, classic_valid, modern_threshold, classic_threshold,
                                   PERCENTILES, MODERN_PERPLEXITY_PERCENTILE, CLASSIC_PERPLEXITY_PERCENTILE, plot_path):
        logging.info(f"📊 分层分布图已保存: {plot_path}")
    
    return modern_threshold, classic_threshold

def layered_perplexity_filter(minhash_data, modern_threshold, classic_threshold, stats):
    """分层困惑度筛选（只生成高质量数据文件）"""
    import logging
    import json
    start_time = time.time()
    
    # 日志说明
    modern_p = PERCENTILES[MODERN_PERPLEXITY_PERCENTILE]
    classic_p = PERCENTILES[CLASSIC_PERPLEXITY_PERCENTILE]
    logging.info(f"🚀 开始分层困惑度筛选")
    logging.info(f"   - 现代文：≤{modern_threshold:.2f} ({modern_p}%分位数)，保留低困惑度优质数据")
    logging.info(f"   - 古文：≥{classic_threshold:.2f} ({classic_p}%分位数)，保留高困惑度真实古文")
    
    # 加载模型
    tokenizer, model = load_perplexity_model()
    
    # 输出文件（只保留高质量数据）
    kept_file = f"{OUTPUT_PATH}/clmmu_kept_data_final.jsonl"
    
    batch_texts = []
    batch_data = []
    total_input = len(minhash_data)
    
    with open(kept_file, "w", encoding="utf-8") as f_kept:
        for item in tqdm(minhash_data, desc="分层困惑度筛选", total=total_input):
            try:
                batch_texts.append(item["text"])
                batch_data.append(item)
            except:
                continue
            
            if len(batch_texts) >= BATCH_SIZE_PERPLEXITY:
                perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
                
                for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
                    data_item["perplexity"] = round(perp, 2)
                    data_item["is_classic"] = is_classic_chinese(text)
                    
                    # 筛选逻辑
                    if data_item["is_classic"]:
                        # 古文：保留≥阈值且无异常值
                        if perp >= classic_threshold and perp < MAX_CLASSIC_PERPLEXITY:
                            final_data = data_item["original_data"].copy()
                            final_data["cleaned_text"] = text
                            final_data["md5"] = data_item["md5"]
                            final_data["perplexity"] = data_item["perplexity"]
                            final_data["source_file"] = data_item["source_file"]
                            final_data["text_type"] = "classic_chinese"
                            final_data["has_academic_features"] = data_item.get("has_academic_features", False)
                            f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                            stats["final_kept"] += 1
                            stats["classic_chinese_kept"] += 1
                        else:
                            stats["perplexity_filtered"] += 1
                    else:
                        # 现代文：保留≤阈值且无异常值
                        if perp <= modern_threshold and perp < MAX_MODERN_PERPLEXITY:
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
                    if perp >= classic_threshold and perp < MAX_CLASSIC_PERPLEXITY:
                        final_data = data_item["original_data"].copy()
                        final_data["cleaned_text"] = text
                        final_data["md5"] = data_item["md5"]
                        final_data["perplexity"] = data_item["perplexity"]
                        final_data["source_file"] = data_item["source_file"]
                        final_data["text_type"] = "classic_chinese"
                        f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                        stats["final_kept"] += 1
                        stats["classic_chinese_kept"] += 1
                    else:
                        stats["perplexity_filtered"] += 1
                else:
                    if perp <= modern_threshold and perp < MAX_MODERN_PERPLEXITY:
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
                        stats["perplexity_filtered"] += 1
    
    logging.info(f"✅ 分层困惑度筛选完成 | 耗时：{round(time.time() - start_time, 2)}秒")
    logging.info(f"📊 分层统计 - 现代文保留: {stats['modern_chinese_kept']} | 古文保留: {stats['classic_chinese_kept']}")
    logging.info(f"📊 总计保留: {stats['final_kept']} | 过滤: {stats['perplexity_filtered']}")
    
    return kept_file