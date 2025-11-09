import json
import logging
import os
from shutil import disk_usage
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewrite_config import *
from model_utils import (
    load_rewrite_model, generate_rewrite,
    calculate_semantic_similarity, calculate_redundancy_ratio
)
from prompt_templates import PROMPTS

# ========== 困惑度计算（保持不变） ==========
try:
    perplexity_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token
    
    perplexity_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    ).to(DEVICE)
    perplexity_model.eval()
    print(f"✅ 困惑度计算模型加载成功（复用{MODEL_ID}）")
except Exception as e:
    print(f"❌ 加载困惑度计算模型失败：{str(e)}")
    exit(1)

def calculate_perplexity(text):
    if not text.strip():
        return float('inf')
    
    inputs = perplexity_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = perplexity_model(** inputs, labels=inputs["input_ids"])
    
    return torch.exp(outputs.loss).item()

# ========== 动态计算阈值（全量数据） ==========
def calculate_modern_perplexity_threshold(input_data_path):
    logging.info(f"🔍 正在计算现代文{MODERN_PERPLEXITY_PERCENTILE}%分位数")
    modern_perplexities = []
    
    with open(input_data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="提取现代文困惑度"):
            try:
                item = json.loads(line)
                if item.get("text_type") == "modern_chinese":
                    perplexity = item.get("perplexity", 0)
                    if perplexity > 0:
                        modern_perplexities.append(perplexity)
            except Exception as e:
                continue
    
    if not modern_perplexities:
        logging.warning("⚠️ 未找到现代文数据，使用默认阈值800")
        return 800.0
    
    threshold = np.percentile(modern_perplexities, MODERN_PERPLEXITY_PERCENTILE)
    logging.info(f"🎯 现代文改写阈值：{threshold:.2f}")
    return threshold

# ========== 磁盘空间检查（适配13.9GB） ==========
def check_disk_space(path, required_gb=10):  # 降低要求至10GB
    try:
        disk = disk_usage(path)
        free_gb = disk.free / (1024 **3)
        if free_gb < required_gb:
            logging.error(f"磁盘空间不足！需要至少{required_gb}GB，当前剩余{free_gb:.2f}GB")
            return False
        return True
    except Exception as e:
        logging.warning(f"磁盘空间检查失败：{str(e)}，继续执行但可能有风险")
        return True

# ========== 核心辅助函数（精简逻辑） ==========
def should_rewrite(item, high_perplexity_threshold):
    text = item["text"]
    if SKIP_CLASSIC_CHINESE and item.get("text_type") == "classic_chinese":
        return False, "古文"
    
    perplexity = item.get("perplexity", 0)
    redundancy_ratio = calculate_redundancy_ratio(text)
    
    if perplexity > high_perplexity_threshold:
        return True, "高困惑度"
    if redundancy_ratio > REDUNDANCY_RATIO_THRESHOLD:
        return True, "冗余"
    
    return False, "无需改写"

def check_quality(original_text, rewritten_text, rewrite_reason, original_perplexity):
    if not rewritten_text:
        return False, "生成空结果", None
    
    sim_score = calculate_semantic_similarity(original_text, rewritten_text)
    if sim_score < SEMANTIC_SIMILARITY_THRESHOLD:
        return False, f"相似度低({sim_score:.2f})", None
    
    rew_perplexity = None
    if rewrite_reason == "高困惑度":
        rew_perplexity = calculate_perplexity(rewritten_text)
        if rew_perplexity >= original_perplexity * PERPLEXITY_REDUCTION_RATIO:
            return False, f"困惑度未降({rew_perplexity:.0f})", rew_perplexity
    
    elif rewrite_reason == "冗余":
        orig_red = calculate_redundancy_ratio(original_text)
        rew_red = calculate_redundancy_ratio(rewritten_text)
        if rew_red >= orig_red:
            return False, f"冗余未降({rew_red:.2f})", None
    
    return True, "合格", rew_perplexity

# ========== 主流程（空间优化版） ==========
def main():
    # 检查磁盘空间（最低10GB）
    if not check_disk_space(os.path.dirname(REWRITTEN_OUTPUT_PATH)):
        print("错误：磁盘空间不足，建议清理至少10GB空间后重试")
        return
    
    # 检查输入文件
    if not os.path.exists(INPUT_DATA_PATH):
        logging.error(f"输入文件不存在：{INPUT_DATA_PATH}")
        print(f"错误：找不到输入文件！{INPUT_DATA_PATH}")
        return
    
    # 动态计算阈值
    high_perplexity_threshold = calculate_modern_perplexity_threshold(INPUT_DATA_PATH)
    
    # 批量大小保持6（平衡速度和内存）
    adjusted_batch_size = 6
    logging.info(f"批量大小：{adjusted_batch_size}")
    
    # 加载模型
    try:
        model, tokenizer = load_rewrite_model()
    except Exception as e:
        logging.error(f"模型加载失败：{str(e)}")
        print(f"错误：模型加载失败！{str(e)}")
        return
    
    # 统计总数据量
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"总数据量：{total_lines}条，可用磁盘空间：{disk_usage(os.path.dirname(REWRITTEN_OUTPUT_PATH)).free/(1024**3):.2f}GB")
    
    # 全量处理（优化空间占用）
    print("开始全量处理（空间优化模式）...")
    batch_buffer = []  # 批量缓存输出，减少IO
    log_buffer = []    # 内存缓存日志，最后写入
    
    # 确保输出目录存在
    try:
        os.makedirs(os.path.dirname(REWRITTEN_OUTPUT_PATH), exist_ok=True)
    except Exception:
        pass

    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f_in, \
         open(REWRITTEN_OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        
        for batch_start in tqdm(range(0, total_lines, adjusted_batch_size), desc="改写进度"):
            # 读取当前批次
            batch = []
            for _ in range(adjusted_batch_size):
                line = f_in.readline()
                if not line:
                    break
                batch.append(line)
            if not batch:
                break
            
            # 处理批次数据
            for line in batch:
                try:
                    item = json.loads(line)
                    original_id = item.get("id", str(hash(item.get("text", ""))))
                    original_text = item.get("text", "").strip()
                    
                    # 空文本处理
                    if not original_text:
                        batch_buffer.append(json.dumps({
                            "id": original_id,
                            "text": original_text,
                            "rewritten": None,
                            "status": "skipped",
                            "reason": "空文本"
                        }, ensure_ascii=False))
                        continue
                    
                    # 判断是否改写
                    need_rewrite, rewrite_reason = should_rewrite(item, high_perplexity_threshold)
                    if not need_rewrite:
                        batch_buffer.append(json.dumps({
                            "id": original_id,
                            "text": original_text,
                            "rewritten": None,
                            "status": "skipped",
                            "reason": rewrite_reason
                        }, ensure_ascii=False))
                        continue
                    
                    # 生成改写
                    prompt = PROMPTS[
                        "high_perplexity" if rewrite_reason == "高困惑度" else "redundant"
                    ].format(text=original_text)
                    
                    rewritten_text = None
                    for retry in range(3):
                        try:
                            rewritten_text = generate_rewrite(model, tokenizer, prompt)
                            if rewritten_text:
                                break
                        except Exception as e:
                            log_buffer.append(f"ID={original_id} 重试{retry+1}次失败：{str(e)}")
                    
                    # 质量检查
                    original_perplexity = item.get("perplexity", 0)
                    quality_ok, quality_reason, rew_perplexity = check_quality(
                        original_text, rewritten_text, rewrite_reason, original_perplexity
                    )
                    
                    # 精简输出字段
                    result = {
                        "id": original_id,
                        "original_text": original_text,
                        "rewritten_text": rewritten_text,
                        "status": "success" if quality_ok else "failed",
                        "reason": quality_reason,
                        "orig_perplexity": round(original_perplexity, 2) if rewrite_reason == "高困惑度" else None,
                        "rew_perplexity": round(rew_perplexity, 2) if rew_perplexity else None
                    }
                    batch_buffer.append(json.dumps(result, ensure_ascii=False))
                    
                    # 每1000条打印进度
                    if int(hash(original_id)) % 1000 == 0:
                        print(f"已处理 {batch_start + len(batch)}/{total_lines} 条")
                
                except Exception as e:
                    batch_buffer.append(json.dumps({
                        "id": original_id if 'original_id' in locals() else "unknown",
                        "text": original_text if 'original_text' in locals() else "解析失败",
                        "rewritten": None,
                        "status": "error",
                        "reason": str(e)
                    }, ensure_ascii=False))
                    log_buffer.append(f"处理ID={original_id}失败：{str(e)}")
                    continue
            
            # 每处理100个批次写入一次磁盘（减少IO）
            if len(batch_buffer) >= 100 * adjusted_batch_size:
                f_out.write("\n".join(batch_buffer) + "\n")
                batch_buffer = []
        
        # 写入剩余缓存数据
        if batch_buffer:
            f_out.write("\n".join(batch_buffer) + "\n")
    
    # 最后写入日志（避免实时占用空间）
    # 确保日志目录存在
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    except Exception:
        pass

    with open(LOG_PATH, "w", encoding="utf-8") as f_log:
        f_log.write("\n".join(log_buffer))
    
    print(f"\n处理完成！结果文件：{REWRITTEN_OUTPUT_PATH}")
    print(f"剩余磁盘空间：{disk_usage(os.path.dirname(REWRITTEN_OUTPUT_PATH)).free/(1024**3):.2f}GB")

if __name__ == "__main__":
    main()