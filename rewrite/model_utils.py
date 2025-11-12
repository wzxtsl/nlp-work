# in rewrite/model_utils.py

import torch
import os
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams
from rewrite_config import *

# ================================================================
# ========== 语义相似度部分 (不再加载模型) ==========
# ================================================================

def text_to_embedding(text, sim_model, sim_tokenizer):
    """【已修改】用传入的模型将文本转为嵌入向量"""
    inputs = sim_tokenizer(
        text, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = sim_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
        embedding = torch.sum(hidden_states * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
        return embedding.cpu().numpy()

def calculate_semantic_similarity(text1, text2, sim_model, sim_tokenizer):
    """【已修改】用传入的模型计算语义相似度"""
    try:
        emb1 = text_to_embedding(text1, sim_model, sim_tokenizer)
        emb2 = text_to_embedding(text2, sim_model, sim_tokenizer)
        cos_sim = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim[0][0])
    except Exception as e:
        print(f"❌ 计算相似度失败：{str(e)}")
        return 0.0

def calculate_redundancy_ratio(text):
    """计算冗余词占比"""
    redundant_words = {"的", "了", "在", "是", "啊", "吧", "呢", "其实", "这个", "那个", "非常", "特别"}
    words = [char for char in text]
    if not words:
        return 0.0
    redundant_count = sum(1 for word in words if word in redundant_words)
    return redundant_count / len(words)

# ================================================================
# ========== vLLM 改写模型加载与生成 (保持不变) ==========
# ================================================================

def load_rewrite_model():
    """使用 vLLM 加载改写用的大模型"""
    print(f"正在加载改写模型 (vLLM Engine)：{REWRITE_MODEL_ID}")
    try:
        model = LLM(
            model=REWRITE_MODEL_ID, 
            trust_remote_code=True,
            gpu_memory_utilization=0.7, # 保持对显存使用的限制
            max_model_len=8192          # 保持对最大长度的限制
        )
        tokenizer = model.get_tokenizer()
        print(f"✅ 改写模型 (vLLM) 加载完成")
        return model, tokenizer
    except Exception as e:
        print(f"❌ vLLM 加载改写模型失败：{str(e)}")
        exit(1)

def generate_rewrites_batch(model: LLM, prompts: list[str]) -> list[str]:
    """使用 vLLM 对一个批次的 prompts 进行高效生成"""
    try:
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=0.95,
            max_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.1
        )
        outputs = model.generate(prompts, sampling_params, use_tqdm=False)
        rewritten_texts = [output.outputs[0].text.strip() for output in outputs]
        return rewritten_texts
    except Exception as e:
        print(f"❌ vLLM 批量生成改写结果失败：{str(e)}")
        return [""] * len(prompts)