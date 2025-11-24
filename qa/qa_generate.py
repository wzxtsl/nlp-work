# qa/qa_generate.py (vLLM 高性能版 + 严格质检)

import os
import json
import logging
from tqdm import tqdm
import random

from vllm import LLM, SamplingParams

from qa.qa_config import (
    REWRITTEN_INPUT_PATH, QA_OUTPUT_PATH, QA_FAILED_PATH, QA_LOG_PATH,
    QA_MODEL_ID, MAX_NEW_TOKENS_QA, TEMPERATURE_QA, TOP_P_QA, BATCH_SIZE_QA,
    MAX_SOURCE_CHARS, MIN_QUESTION_LEN, MAX_QUESTION_LEN, MIN_ANSWER_LEN,
    MAX_ANSWER_LEN, REQUIRED_CHINESE_PUNCT, SEMANTIC_SIMILARITY_MIN,
    TYPE_KEYWORDS, TYPE_PROMPT_MAPPING, NUM_QA_ATTEMPTS_PER_TEXT
)
from qa.prompt_templates import get_random_prompt

try:
    from rewrite.model_utils import text_to_embedding
    _HAS_EMB = True
    logging.info("✅ 成功导入语义相似度计算函数 (text_to_embedding)")
except ImportError:
    _HAS_EMB = False
    logging.warning("⚠️ 未找到语义相似度计算函数，相关质检将跳过。")

# ================================================================
# ========== 核心功能函数 ==========
# ================================================================

def load_qa_model():
    # ... (此函数保持不变) ...
    logging.info(f"📥 正在加载QA生成模型 (vLLM Engine): {QA_MODEL_ID}")
    try:
        model = LLM(model=QA_MODEL_ID, trust_remote_code=True)
        tokenizer = model.get_tokenizer()
        logging.info("✅ QA模型 (vLLM) 加载完成")
        return model, tokenizer
    except Exception as e:
        logging.error(f"❌ vLLM 模型加载失败: {e}", exc_info=True)
        exit(1)

def detect_question_type(source_text: str) -> str:
    # ... (此函数保持不变) ...
    lower_text = source_text.lower()
    for q_type, kws in TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in source_text or kw in lower_text:
                return q_type
    if any(ch.isdigit() for ch in source_text) and any(sym in source_text for sym in ["=", "+", "-", "→", "%"]):
        return "calculate"
    return "fallback"

def semantic_similarity(a: str, b: str) -> float:
    # ... (此函数保持不变) ...
    if not _HAS_EMB: return 1.0
    try:
        emb1 = text_to_embedding(a)
        emb2 = text_to_embedding(b)
        import numpy as np
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        sim = float(np.dot(emb1, emb2.T))
        return sim
    except Exception:
        return 0.0

# ========== 【【核心修改点 1】】: 升级解析器 ==========
def parse_generated(text: str) -> tuple:
    """
    【升级版】从模型输出中解析【单个】问答对，并主动拒绝不良格式。
    返回: (question, answer_or_error_reason)
    """
    q_marker, a_marker = "问题：", "答案："
    
    # 规则1：检查是否包含多余的 "问题：" 标记
    if text.count(q_marker) > 1:
        return None, "解析失败: 输出包含多个QA对"

    q_idx = text.find(q_marker)
    a_idx = text.find(a_marker)

    # 规则2：检查基本结构是否存在
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        return None, "解析失败: 结构不符"
        
    question = text[q_idx + len(q_marker):a_idx].strip()
    answer = text[a_idx + len(a_marker):].strip()

    # 规则3：检查答案是否包含无意义的占位符或为空
    if not answer or "..." in answer:
        return question, "解析失败: 答案为空或未完成"

    return question, answer

# ========== 【【核心修改点 2】】: 升级质检员 ==========
def validate_pair(source_text, question, answer):
    """
    【升级版】对生成的问答对进行多维度质量校验。
    """
    # 规则0：前置解析已失败
    if not question:
        return False, answer or "解析失败: 未找到问题"
    if answer and "解析失败:" in answer:
        return False, answer
        
    # 规则1：问题缺少问号
    if REQUIRED_CHINESE_PUNCT and REQUIRED_CHINESE_PUNCT not in question:
        return False, "问题缺少问号"
        
    # 规则2：问题长度异常
    if not (MIN_QUESTION_LEN <= len(question) <= MAX_QUESTION_LEN):
        return False, f"问题长度异常({len(question)})"
        
    # 规则3：答案长度异常
    if not (MIN_ANSWER_LEN <= len(answer) <= MAX_ANSWER_LEN):
        return False, f"答案长度异常({len(answer)})"
    
    # 规则4：与原文相关度低
    sim = semantic_similarity(source_text, question + " " + answer)
    if sim < SEMANTIC_SIMILARITY_MIN:
        return False, f"与原文相关度低({sim:.2f})"
    
    return True, "ok"


# ================================================================
# ========== vLLM 批量生成函数 (保持不变) ==========
# ================================================================
def generate_batch_qa(model: LLM, prompts: list[str]) -> list[str]:
    # ... (此函数保持不变) ...
    try:
        sampling_params = SamplingParams(temperature=TEMPERATURE_QA, top_p=TOP_P_QA, max_tokens=MAX_NEW_TOKENS_QA)
        outputs = model.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]
    except Exception as e:
        logging.warning(f"⚠️ vLLM 批量生成失败: {e}")
        return [""] * len(prompts)


# ================================================================
# ========== 主流程 (保持不变，已支持多轮生成) ==========
# ================================================================

def process_batch(batch_data, model, f_ok, f_fail):
    # ... (此函数现在与升级后的 parse/validate 无缝对接，无需修改) ...
    if not batch_data: return 0, 0
    batch_prompts = [item['prompt'] for item in batch_data]
    raw_outputs = generate_batch_qa(model, batch_prompts)
    success_count, fail_count = 0, 0
    seen_in_batch = set() # 批次内去重

    for i, raw_output in enumerate(raw_outputs):
        item_data = batch_data[i]
        
        # 【注意】这里的 question 和 answer 已经经过了升级版的 parse_generated
        question, answer = parse_generated(raw_output)
        
        # 【注意】这里的 validate_pair 是升级版的
        ok, reason = validate_pair(item_data['source_text'], question, answer)

        # (可选) 批次内去重逻辑
        if ok:
            qa_pair_str = f"{question}|{answer}"
            if qa_pair_str in seen_in_batch:
                ok, reason = False, "批次内重复"
            else:
                seen_in_batch.add(qa_pair_str)

        record = {
            "id": item_data["id"], "question": question, "answer": answer,
            "status": "success" if ok else "failed", "fail_reason": None if ok else reason,
            "question_type": item_data["q_type"], "prompt_key": item_data["prompt_key"]
        }

        if ok:
            f_ok.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1
        else:
            f_fail.write(json.dumps(record, ensure_ascii=False) + "\n")
            fail_count += 1
            
    return success_count, fail_count

def generate_qa_pairs():
    # ... (此函数保持不变) ...
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
    logging.info(f"🚀 每个源文本将尝试生成 {NUM_QA_ATTEMPTS_PER_TEXT} 次 QA")
    total_success, total_fail, batch_data = 0, 0, []
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
                for _ in range(NUM_QA_ATTEMPTS_PER_TEXT):
                    q_type = detect_question_type(source_text)
                    prompt_key = TYPE_PROMPT_MAPPING.get(q_type, "generic_q")
                    prompt_template = get_random_prompt(prompt_key)
                    prompt = prompt_template.format(text=source_text)
                    batch_data.append({
                        "id": item.get("id"), "prompt": prompt, "source_text": source_text,
                        "q_type": q_type, "prompt_key": prompt_key
                    })
                if len(batch_data) >= BATCH_SIZE_QA:
                    s, f = process_batch(batch_data, model, f_ok, f_fail)
                    total_success += s
                    total_fail += f
                    batch_data = []
            except Exception as e:
                logging.error(f"处理行时发生未知异常: {e}", exc_info=True)
                total_fail += 1
                continue
        if batch_data:
            s, f = process_batch(batch_data, model, f_ok, f_fail)
            total_success += s
            total_fail += f
    logging.info(f"✅ QA生成完成 | 成功 {total_success} | 失败 {total_fail}")
    logging.info(f"📄 成功问答对输出至: {QA_OUTPUT_PATH}")
    logging.info(f"📄 失败记录输出至: {QA_FAILED_PATH}")


def main():
    # ... (此函数保持不变) ...
    os.makedirs(os.path.dirname(QA_LOG_PATH), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(QA_LOG_PATH, mode='w', encoding="utf-8")]
    )
    logging.info("🚀 启动QA生成流程 (vLLM 高性能版)")
    generate_qa_pairs()
    logging.info("🎉 QA生成流程结束")

if __name__ == "__main__":
    main()
