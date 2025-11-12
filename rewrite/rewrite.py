# rewrite.py (最终修复版)

import json
import logging
import os
import random
from shutil import disk_usage
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rewrite_config import *
# 导入修改后的函数
from model_utils import (
    load_rewrite_model, generate_rewrites_batch,
    calculate_semantic_similarity, calculate_redundancy_ratio
)
from prompt_templates import PROMPTS

# ================================================================
# ========== 【核心修改1】: 统一加载“工具模型” ==========
# ================================================================

# 在全局只加载一次“工具模型”，用于困惑度和语义相似度计算
try:
    print(f"正在加载统一的工具模型（{MODEL_ID}）...")
    tool_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tool_tokenizer.pad_token is None:
        tool_tokenizer.pad_token = tool_tokenizer.eos_token
    
    tool_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True, trust_remote_code=True
    ).to(DEVICE)
    tool_model.eval()
    print(f"✅ 工具模型加载成功")
except Exception as e:
    print(f"❌ 加载工具模型失败：{str(e)}")
    exit(1)

# ================================================================
# ========== 核心辅助函数 (适配模型传递) ==========
# ================================================================

def calculate_perplexity(text):
    """【已修改】使用全局加载的工具模型计算困惑度"""
    if not text.strip(): return float('inf')
    inputs = tool_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
    with torch.no_grad():
        outputs = tool_model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def calculate_modern_perplexity_threshold(input_data_path):
    # 此函数不变
    logging.info(f"🔍 正在计算现代文{MODERN_PERPLEXITY_PERCENTILE}%分位数")
    # ... (代码不变)
    modern_perplexities = []
    with open(input_data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="提取现代文困惑度"):
            try:
                item = json.loads(line)
                if item.get("text_type") == "modern_chinese":
                    perplexity = item.get("perplexity", 0)
                    if perplexity > 0: modern_perplexities.append(perplexity)
            except: continue
    if not modern_perplexities:
        logging.warning("⚠️ 未找到现代文数据，使用默认阈值800")
        return 800.0
    threshold = np.percentile(modern_perplexities, MODERN_PERPLEXITY_PERCENTILE)
    logging.info(f"🎯 现代文改写阈值：{threshold:.2f}")
    return threshold


def check_disk_space(path, required_gb=10):
    # 此函数不变
    # ...
    try:
        free_gb = disk_usage(path).free / (1024**3)
        if free_gb < required_gb:
            logging.error(f"磁盘空间不足！需要至少{required_gb}GB，当前剩余{free_gb:.2f}GB")
            return False
        return True
    except:
        return True

def should_rewrite(item, high_perplexity_threshold):
    # 此函数不变
    # ...
    text = item.get("text", "").strip()
    if not text: return False, None
    if SKIP_CLASSIC_CHINESE and item.get("text_type") == "classic_chinese": return False, "古文"
    perplexity = item.get("perplexity", 0)
    if perplexity > high_perplexity_threshold: return True, "high_perplexity"
    redundancy_ratio = calculate_redundancy_ratio(text)
    if redundancy_ratio > REDUNDANCY_RATIO_THRESHOLD: return True, "redundant"
    if ENABLE_LOGIC_INJECTION and item.get("text_type") == "modern_chinese":
        is_short = len(text) < LOGIC_CANDIDATE_MAX_LEN
        has_no_logic = not any(word in text for word in LOGIC_KEYWORDS)
        if is_short and has_no_logic and random.random() < LOGIC_INJECTION_SAMPLING_RATE:
            return True, "add_logic_chain"
    return False, "无需改写"


def check_quality(original_text, rewritten_text, rewrite_reason, original_perplexity):
    """【已修改】将全局工具模型作为参数传递给相似度计算函数"""
    if not rewritten_text or "改写失败" in rewritten_text: return False, "生成空结果或失败", None

    if rewrite_reason == "add_logic_chain":
        if not any(word in rewritten_text for word in LOGIC_KEYWORDS): return False, "未注入逻辑链", None
        sim_score = calculate_semantic_similarity(original_text, rewritten_text, tool_model, tool_tokenizer) # 传递模型
        if sim_score < LOGIC_SIMILARITY_THRESHOLD: return False, f"相似度过低({sim_score:.2f})", None
        rew_perplexity = calculate_perplexity(rewritten_text)
        return True, "logic_injected", rew_perplexity

    sim_score = calculate_semantic_similarity(original_text, rewritten_text, tool_model, tool_tokenizer) # 传递模型
    if sim_score < SEMANTIC_SIMILARITY_THRESHOLD: return False, f"相似度低({sim_score:.2f})", None
    
    rew_perplexity = None
    if rewrite_reason == "high_perplexity":
        rew_perplexity = calculate_perplexity(rewritten_text)
        if rew_perplexity >= original_perplexity * PERPLEXITY_REDUCTION_RATIO: return False, f"困惑度未降({rew_perplexity:.0f})", rew_perplexity
    elif rewrite_reason == "redundant":
        orig_red = calculate_redundancy_ratio(original_text)
        rew_red = calculate_redundancy_ratio(rewritten_text)
        if rew_red >= orig_red: return False, f"冗余未降({rew_red:.2f})", None
        
    return True, "合格", rew_perplexity

# ================================================================
# ========== 主流程 (保持vLLM的“先分类，后批量”模式) ==========
# ================================================================

def main():
    if not check_disk_space(os.path.dirname(REWRITTEN_OUTPUT_PATH)): return
    if not os.path.exists(INPUT_DATA_PATH):
        logging.error(f"输入文件不存在：{INPUT_DATA_PATH}")
        return

    high_perplexity_threshold = calculate_modern_perplexity_threshold(INPUT_DATA_PATH)
    
    try:
        vllm_model, vllm_tokenizer = load_rewrite_model()
    except Exception as e:
        logging.error(f"模型加载失败：{str(e)}")
        return
        
    print("步骤 1/2: 正在分类数据...")
    items_to_rewrite = []
    skipped_items_output = []
    
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        
    for line in tqdm(all_lines, desc="分类进度"):
        try:
            item = json.loads(line)
            # --- 【【【新增代码：截断超长文本】】】 ---
            original_text = item.get("text", "").strip()
            # 我们需要给Prompt模板的文本留出一些空间，所以截断长度要比max_model_len小
            # 假设Prompt模板本身大约占200个token
            max_model_len=8192
            max_text_len = max_model_len - 200 
            if len(original_text) > max_text_len:
                original_text = original_text[:max_text_len]
                item['text'] = original_text # 将截断后的文本写回item，供后续使用
            need_rewrite, rewrite_reason = should_rewrite(item, high_perplexity_threshold)
            
            if need_rewrite and rewrite_reason in PROMPTS:
                item['rewrite_reason'] = rewrite_reason
                items_to_rewrite.append(item)
            else:
                original_id = item.get("id", str(hash(item.get("text", ""))))
                skipped_items_output.append(json.dumps({
                    "id": original_id, "original_text": item.get("text", ""), "rewritten_text": None,
                    "status": "skipped", "reason": rewrite_reason
                }, ensure_ascii=False))
        except Exception:
            continue
            
    print(f"步骤 2/2: 找到 {len(items_to_rewrite)} 条数据需要改写，开始批量处理...")
    
    os.makedirs(os.path.dirname(REWRITTEN_OUTPUT_PATH), exist_ok=True)

    with open(REWRITTEN_OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        if skipped_items_output:
            f_out.write("\n".join(skipped_items_output) + "\n")
            
        for i in tqdm(range(0, len(items_to_rewrite), BATCH_SIZE), desc="vLLM改写进度"):
            batch_chunk = items_to_rewrite[i : i + BATCH_SIZE]
            prompts = [PROMPTS[item['rewrite_reason']].format(text=item.get("text", "").strip()) for item in batch_chunk]
            rewritten_texts = generate_rewrites_batch(vllm_model, prompts)
            
            output_buffer = []
            for j, rewritten_text in enumerate(rewritten_texts):
                original_item = batch_chunk[j]
                original_text = original_item.get("text", "").strip()
                original_id = original_item.get("id", str(hash(original_text)))
                
                quality_ok, quality_reason, rew_perplexity = check_quality(
                    original_text, rewritten_text, original_item['rewrite_reason'], original_item.get("perplexity", 0)
                )
                
                result = {
                    "id": original_id, "original_text": original_text, "rewritten_text": rewritten_text,
                    "status": "success" if quality_ok else "failed", "reason": quality_reason,
                    "orig_perplexity": round(original_item.get("perplexity", 0), 2) if original_item['rewrite_reason'] == "high_perplexity" else None,
                    "rew_perplexity": round(rew_perplexity, 2) if rew_perplexity else None
                }
                output_buffer.append(json.dumps(result, ensure_ascii=False))
            
            f_out.write("\n".join(output_buffer) + "\n")

    print(f"\n处理完成！结果文件：{REWRITTEN_OUTPUT_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()