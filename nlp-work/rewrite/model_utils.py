import torch
import os
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewrite_config import MAX_SEQ_LENGTH
from rewrite_config import *

# ========== 复用已有模型计算语义相似度（核心修改） ==========
# 加载你已有的GPT2中文模型（无需下载新模型！）
try:
    print(f"正在加载语义相似度模型（复用已有GPT2模型）...")
    # 复用rewrite_config中的MODEL_ID（uer/gpt2-chinese-cluecorpussmall）
    sim_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if sim_tokenizer.pad_token is None:
        sim_tokenizer.pad_token = sim_tokenizer.eos_token
    
    # 加载模型（仅用于编码，不生成，显存占用极低）
    sim_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    sim_model.eval()
    print(f"✅ 语义相似度模型加载成功（复用GPT2模型，无额外下载）")
except Exception as e:
    print(f"❌ 加载模型失败：{str(e)}")
    exit(1)

def text_to_embedding(text):
    """用GPT2模型将文本转为嵌入向量（替代sentence-transformers）"""
    inputs = sim_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = sim_model(**inputs, output_hidden_states=True)
        # 取最后一层隐藏状态的均值作为文本嵌入
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
        embedding = torch.sum(hidden_states * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
        return embedding.cpu().numpy()

# ========== 改写模型加载（保持不变） ==========
def load_rewrite_model():
    """加载改写用的大模型（自动适配设备+量化省显存）"""
    print(f"正在加载改写模型：{REWRITE_MODEL_ID}（请耐心等待...）")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            REWRITE_MODEL_ID,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            REWRITE_MODEL_ID,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"✅ 改写模型加载完成，运行设备：{model.device}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 加载改写模型失败：{str(e)}")
        print("可能原因：模型名错误/网络问题/显存不足")
        exit(1)

# ========== 生成改写结果（保持不变） ==========
def generate_rewrite(model, tokenizer, prompt):
    try:
        if "qwen" in REWRITE_MODEL_ID.lower():
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        else:
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(model.device)["input_ids"]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        rewritten_text = tokenizer.decode(
            outputs[0][len(input_ids[0]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        return rewritten_text if rewritten_text else "改写失败：生成结果为空"
    except Exception as e:
        print(f"❌ 生成改写结果失败：{str(e)}")
        return None

# ========== 工具函数（修改相似度计算逻辑） ==========
def calculate_semantic_similarity(text1, text2):
    """用已有GPT2模型计算语义相似度（替代sentence-transformers）"""
    try:
        # 转为嵌入向量
        emb1 = text_to_embedding(text1)
        emb2 = text_to_embedding(text2)
        
        # 计算余弦相似度
        cos_sim = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim[0][0])
    except Exception as e:
        print(f"❌ 计算相似度失败：{str(e)}")
        return 0.0

def is_question_style(text):
    """判断文本是否符合考题风格（含考题关键词）"""
    return any(kw in text for kw in QUESTION_KEYWORDS)

def calculate_redundancy_ratio(text):
    """计算冗余词占比（简单有效，针对中文）"""
    redundant_words = {"的", "了", "在", "是", "啊", "吧", "呢", "其实", "这个", "那个", "非常", "特别"}
    words = text.strip().split()
    if not words:
        return 0.0
    redundant_count = sum(1 for word in words if word in redundant_words)
    return redundant_count / len(words)