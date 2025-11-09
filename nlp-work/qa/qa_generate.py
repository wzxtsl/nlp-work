import os
import json
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from qa.qa_config import (
    REWRITTEN_INPUT_PATH, QA_OUTPUT_PATH, QA_FAILED_PATH, QA_LOG_PATH,
    QA_MODEL_ID, MAX_NEW_TOKENS_QA, TEMPERATURE_QA, TOP_P_QA, BATCH_SIZE_QA,
    MAX_SOURCE_CHARS, MIN_QUESTION_LEN, MAX_QUESTION_LEN, MIN_ANSWER_LEN,
    MAX_ANSWER_LEN, REQUIRED_CHINESE_PUNCT, SEMANTIC_SIMILARITY_MIN,
    TYPE_KEYWORDS, TYPE_PROMPT_MAPPING
)
from qa.prompt_templates import QA_PROMPTS

# 复用已有 embedding 方法（如果存在）
try:
    from rewrite.model_utils import text_to_embedding
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ========== 模型加载 ==========
def load_qa_model():
    logging.info(f"📥 正在加载QA生成模型: {QA_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QA_MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        load_in_4bit=True
    )
    model.eval()
    logging.info("✅ QA模型加载完成")
    return model, tokenizer

# ========== 问题类型猜测 ==========
def detect_question_type(source_text: str) -> str:
    lower_text = source_text.lower()
    for q_type, kws in TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in source_text or kw in lower_text:
                return q_type
    # 简单公式/数字判断归入 calculate
    if any(ch.isdigit() for ch in source_text) and any(sym in source_text for sym in ["=", "+", "-", "→", "%"]):
        return "calculate"
    return "fallback"

# ========== 语义相关度 ==========
def semantic_similarity(a: str, b: str) -> float:
    if not _HAS_EMB:
        return 1.0  # 没有嵌入函数时直接放行
    try:
        emb1 = text_to_embedding(a)
        emb2 = text_to_embedding(b)
        import numpy as np
        sim = float((emb1 @ emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return sim
    except Exception:
        return 0.0

# ========== 生成单个QA ==========
def generate_single_qa(model, tokenizer, prompt: str) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)["input_ids"]
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS_QA,
                temperature=TEMPERATURE_QA,
                top_p=TOP_P_QA,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        gen = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()
        return gen
    except Exception as e:
        logging.warning(f"⚠️ 生成失败: {e}")
        return ""

# ========== 解析输出 ==========
def parse_generated(text: str):
    # 寻找“问题：”和“答案：”分隔
    q_marker, a_marker = "问题：", "答案："
    q_idx = text.find(q_marker)
    a_idx = text.find(a_marker)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        return None, None
    question = text[q_idx + len(q_marker):a_idx].strip().strip("\n")
    answer = text[a_idx + len(a_marker):].strip()
    return question, answer

# ========== 质量校验 ==========
def validate_pair(source_text, question, answer):
    if not question or not answer:
        return False, "空问答"
    if REQUIRED_CHINESE_PUNCT not in question:
        return False, "缺少问号"
    if not (MIN_QUESTION_LEN <= len(question) <= MAX_QUESTION_LEN):
        return False, "问题长度异常"
    if not (MIN_ANSWER_LEN <= len(answer) <= MAX_ANSWER_LEN):
        return False, "答案长度异常"
    sim = semantic_similarity(source_text, question + " " + answer)
    if sim < SEMANTIC_SIMILARITY_MIN:
        return False, f"相关度低({sim:.2f})"
    return True, "ok"

# ========== 主流程 ==========
def generate_qa_pairs():
    if not os.path.exists(REWRITTEN_INPUT_PATH):
        logging.error(f"❌ 改写结果文件不存在：{REWRITTEN_INPUT_PATH}")
        return

    model, tokenizer = load_qa_model()

    total_lines = sum(1 for _ in open(REWRITTEN_INPUT_PATH, 'r', encoding='utf-8'))
    logging.info(f"📄 输入改写数据条数：{total_lines}")

    success_count = 0
    fail_count = 0

    with open(REWRITTEN_INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(QA_OUTPUT_PATH, 'w', encoding='utf-8') as f_ok, \
         open(QA_FAILED_PATH, 'w', encoding='utf-8') as f_fail, \
         open(QA_LOG_PATH, 'w', encoding='utf-8') as f_log:

        for line in tqdm(f_in, desc="生成QA"):
            try:
                item = json.loads(line)
                source_text = item.get("rewritten_text") or item.get("original_text") or item.get("text") or ""
                source_text = source_text.strip()
                if not source_text:
                    fail_count += 1
                    f_fail.write(json.dumps({"id": item.get("id"), "reason": "空源文本"}, ensure_ascii=False) + "\n")
                    continue
                # 截断超长文本
                if len(source_text) > MAX_SOURCE_CHARS:
                    source_text = source_text[:MAX_SOURCE_CHARS]

                q_type = detect_question_type(source_text)
                prompt_key = TYPE_PROMPT_MAPPING.get(q_type, "generic_q")
                prompt_template = QA_PROMPTS[prompt_key]
                prompt = prompt_template.format(text=source_text)

                raw_output = generate_single_qa(model, tokenizer, prompt)
                question, answer = parse_generated(raw_output)
                ok, reason = validate_pair(source_text, question, answer)

                record = {
                    "id": item.get("id"),
                    "question": question,
                    "answer": answer,
                    "status": "success" if ok else "failed",
                    "fail_reason": None if ok else reason,
                    "question_type": q_type,
                    "prompt_key": prompt_key
                }

                if ok:
                    f_ok.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success_count += 1
                else:
                    f_fail.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fail_count += 1

                # 简单日志
                if (success_count + fail_count) % 200 == 0:
                    f_log.write(f"进度: 成功={success_count}, 失败={fail_count}\n")

            except Exception as e:
                fail_count += 1
                f_fail.write(json.dumps({"id": None, "reason": f"异常: {str(e)}"}, ensure_ascii=False) + "\n")
                continue

    logging.info(f"✅ QA生成完成 | 成功 {success_count} | 失败 {fail_count}")
    logging.info(f"输出文件：{QA_OUTPUT_PATH}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(QA_LOG_PATH, encoding="utf-8")
        ]
    )
    logging.info("🚀 启动QA生成流程")
    generate_qa_pairs()
    logging.info("🎉 QA生成流程结束")

if __name__ == "__main__":
    main()
