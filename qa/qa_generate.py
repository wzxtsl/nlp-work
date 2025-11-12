# qa/qa_generate.py (vLLM 高性能版本)

import os
import json
import logging
from tqdm import tqdm
import random # 确保导入 random

# 导入 vLLM 核心库
from vllm import LLM, SamplingParams

# 从配置文件导入所有参数
from qa.qa_config import (
    REWRITTEN_INPUT_PATH, QA_OUTPUT_PATH, QA_FAILED_PATH, QA_LOG_PATH,
    QA_MODEL_ID, MAX_NEW_TOKENS_QA, TEMPERATURE_QA, TOP_P_QA, BATCH_SIZE_QA,
    MAX_SOURCE_CHARS, MIN_QUESTION_LEN, MAX_QUESTION_LEN, MIN_ANSWER_LEN,
    MAX_ANSWER_LEN, REQUIRED_CHINESE_PUNCT, SEMANTIC_SIMILARITY_MIN,
    TYPE_KEYWORDS, TYPE_PROMPT_MAPPING
)
# 从模板文件导入随机选择函数
from qa.prompt_templates import get_random_prompt

# 动态导入语义相似度计算函数，保持松耦合
try:
    from rewrite.model_utils import text_to_embedding
    _HAS_EMB = True
    logging.info("✅ 成功导入语义相似度计算函数 (text_to_embedding)")
except ImportError:
    _HAS_EMB = False
    logging.warning("⚠️ 未找到语义相似度计算函数，相关质检将跳过。")

# ================================================================
# ========== 核心功能函数 (大部分保持不变) ==========
# ================================================================

def load_qa_model():
    """使用 vLLM 加载模型"""
    logging.info(f"📥 正在加载QA生成模型 (vLLM Engine): {QA_MODEL_ID}")
    try:
        # vLLM 会自动优化模型加载，对于1.8B模型，4090加载FP16性能最佳
        model = LLM(model=QA_MODEL_ID, trust_remote_code=True)
        tokenizer = model.get_tokenizer() # Tokenizer 从 vLLM 实例中获取
        logging.info("✅ QA模型 (vLLM) 加载完成")
        return model, tokenizer
    except Exception as e:
        logging.error(f"❌ vLLM 模型加载失败: {e}", exc_info=True)
        logging.error("可能原因：模型ID错误、网络问题、vLLM与CUDA驱动不兼容或显存不足。")
        exit(1)

def detect_question_type(source_text: str) -> str:
    """根据关键词猜测问题类型"""
    lower_text = source_text.lower()
    for q_type, kws in TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in source_text or kw in lower_text:
                return q_type
    if any(ch.isdigit() for ch in source_text) and any(sym in source_text for sym in ["=", "+", "-", "→", "%"]):
        return "calculate"
    return "fallback"

def semantic_similarity(a: str, b: str) -> float:
    """计算语义相似度"""
    if not _HAS_EMB:
        return 1.0  # 没有嵌入函数时直接放行
    try:
        emb1 = text_to_embedding(a)
        emb2 = text_to_embedding(b)
        import numpy as np
        # 归一化后点积，更稳定
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        sim = float(np.dot(emb1, emb2.T))
        return sim
    except Exception:
        return 0.0

def parse_generated(text: str):
    """从模型输出中解析问答对"""
    q_marker, a_marker = "问题：", "答案："
    q_idx = text.find(q_marker)
    a_idx = text.find(a_marker)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        return None, None
    question = text[q_idx + len(q_marker):a_idx].strip()
    answer = text[a_idx + len(a_marker):].strip()
    return question, answer

def validate_pair(source_text, question, answer):
    """对生成的问答对进行多维度质量校验"""
    if not question or not answer:
        return False, "生成空问答"
    if REQUIRED_CHINESE_PUNCT not in question:
        return False, "问题缺少问号"
    if not (MIN_QUESTION_LEN <= len(question) <= MAX_QUESTION_LEN):
        return False, f"问题长度异常({len(question)})"
    if not (MIN_ANSWER_LEN <= len(answer) <= MAX_ANSWER_LEN):
        return False, f"答案长度异常({len(answer)})"
    
    sim = semantic_similarity(source_text, question + " " + answer)
    if sim < SEMANTIC_SIMILARITY_MIN:
        return False, f"与原文相关度低({sim:.2f})"
    
    return True, "ok"

# ================================================================
# ========== vLLM 批量生成函数 (新增) ==========
# ================================================================

def generate_batch_qa(model: LLM, prompts: list[str]) -> list[str]:
    """使用 vLLM 对一个批次的 prompts 进行高效生成"""
    try:
        sampling_params = SamplingParams(
            temperature=TEMPERATURE_QA,
            top_p=TOP_P_QA,
            max_tokens=MAX_NEW_TOKENS_QA
        )
        outputs = model.generate(prompts, sampling_params, use_tqdm=False)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        return generated_texts
    except Exception as e:
        logging.warning(f"⚠️ vLLM 批量生成失败: {e}")
        return [""] * len(prompts)

# ================================================================
# ========== 主流程 (完全重构为批量处理模式) ==========
# ================================================================

def process_batch(batch_data, model, f_ok, f_fail):
    """处理一个批次的数据：生成、解析、质检、写入"""
    if not batch_data:
        return 0, 0

    # 准备批量数据
    batch_prompts = [item['prompt'] for item in batch_data]
    
    # 批量生成
    raw_outputs = generate_batch_qa(model, batch_prompts)

    success_count, fail_count = 0, 0

    # 遍历批次结果并处理
    for i, raw_output in enumerate(raw_outputs):
        item_data = batch_data[i]
        question, answer = parse_generated(raw_output)
        ok, reason = validate_pair(item_data['source_text'], question, answer)

        record = {
            "id": item_data["id"],
            "question": question,
            "answer": answer,
            "status": "success" if ok else "failed",
            "fail_reason": None if ok else reason,
            "question_type": item_data["q_type"],
            "prompt_key": item_data["prompt_key"]
        }

        if ok:
            f_ok.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1
        else:
            f_fail.write(json.dumps(record, ensure_ascii=False) + "\n")
            fail_count += 1
            
    return success_count, fail_count

def generate_qa_pairs():
    """主函数，负责读取、批处理和写入"""
    if not os.path.exists(REWRITTEN_INPUT_PATH):
        logging.error(f"❌ 改写结果文件不存在：{REWRITTEN_INPUT_PATH}")
        return

    model, _ = load_qa_model()

    try:
        total_lines = sum(1 for _ in open(REWRITTEN_INPUT_PATH, 'r', encoding='utf-8'))
    except Exception as e:
        logging.error(f"❌ 无法读取输入文件行数: {e}")
        return
        
    logging.info(f"📄 输入改写数据总条数：{total_lines}")
    logging.info(f"🚀 使用 vLLM 引擎，批处理大小 (Batch Size): {BATCH_SIZE_QA}")

    total_success = 0
    total_fail = 0
    batch_data = []

    with open(REWRITTEN_INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(QA_OUTPUT_PATH, 'w', encoding='utf-8') as f_ok, \
         open(QA_FAILED_PATH, 'w', encoding='utf-8') as f_fail:

        for line in tqdm(f_in, total=total_lines, desc="QA 生成进度"):
            try:
                item = json.loads(line)
                source_text = (item.get("rewritten_text") or item.get("original_text") or item.get("text") or "").strip()

                if not source_text:
                    total_fail += 1
                    f_fail.write(json.dumps({"id": item.get("id"), "reason": "空源文本"}, ensure_ascii=False) + "\n")
                    continue
                
                if len(source_text) > MAX_SOURCE_CHARS:
                    source_text = source_text[:MAX_SOURCE_CHARS]

                q_type = detect_question_type(source_text)
                prompt_key = TYPE_PROMPT_MAPPING.get(q_type, "generic_q")
                prompt_template = get_random_prompt(prompt_key)
                prompt = prompt_template.format(text=source_text)
                
                batch_data.append({
                    "id": item.get("id"),
                    "prompt": prompt,
                    "source_text": source_text,
                    "q_type": q_type,
                    "prompt_key": prompt_key
                })
                
                if len(batch_data) >= BATCH_SIZE_QA:
                    s, f = process_batch(batch_data, model, f_ok, f_fail)
                    total_success += s
                    total_fail += f
                    batch_data = []

            except json.JSONDecodeError:
                total_fail += 1
                f_fail.write(json.dumps({"id": None, "reason": "JSON解析失败"}, ensure_ascii=False) + "\n")
                continue
            except Exception as e:
                logging.error(f"处理行时发生未知异常: {e}", exc_info=True)
                total_fail += 1
                continue
        
        # 处理最后一个不完整的批次
        if batch_data:
            s, f = process_batch(batch_data, model, f_ok, f_fail)
            total_success += s
            total_fail += f

    logging.info(f"✅ QA生成完成 | 成功 {total_success} | 失败 {total_fail}")
    logging.info(f"📄 成功问答对输出至: {QA_OUTPUT_PATH}")
    logging.info(f"📄 失败记录输出至: {QA_FAILED_PATH}")


def main():
    # 确保日志目录存在
    os.makedirs(os.path.dirname(QA_LOG_PATH), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(QA_LOG_PATH, mode='w', encoding="utf-8")
        ]
    )
    logging.info("🚀 启动QA生成流程 (vLLM 高性能版)")
    generate_qa_pairs()
    logging.info("🎉 QA生成流程结束")

if __name__ == "__main__":
    main()