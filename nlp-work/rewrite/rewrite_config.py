import sys
from pathlib import Path

# 把项目根目录（假设 rewrite 文件夹和 config.py 在同一级）加入搜索路径
sys.path.append(str(Path(__file__).parent.parent))

# 从已有config.py导入关键配置
from config import MODEL_ID, DEVICE, OUTPUT_PATH
import os

# ========== 路径配置 ==========
INPUT_DATA_PATH = os.path.join(OUTPUT_PATH, "clmmu_kept_data_final.jsonl")  # 上一步输出
REWRITTEN_OUTPUT_PATH = os.path.join("data", "rewrite_output", "rewritten_data.jsonl")
FAILED_OUTPUT_PATH = os.path.join("data", "rewrite_output", "rewrite_failed.jsonl")
LOG_PATH = os.path.join("data", "rewrite_output", "rewrite_log.log")

# ========== 改写触发条件（核心：用百分数动态计算） ==========
MODERN_PERPLEXITY_PERCENTILE = 80  # 现代文取80%分位数（高于此值需要改写）
REDUNDANCY_RATIO_THRESHOLD = 0.2   # 冗余词占比阈值
NON_QUESTION_THRESHOLD = 0.5       # 非考题风格阈值

# ========== 古文处理配置（全程跳过） ==========
SKIP_CLASSIC_CHINESE = True  # 古文不参与改写

# ========== 改写用大模型配置 ==========
REWRITE_MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen1.5-1.8B-Chat"  # 中文小模型，省显存
BATCH_SIZE = 2
MAX_NEW_TOKENS = 512
MAX_SEQ_LENGTH = 512
TEMPERATURE = 0.7

# ========== 质量评估阈值 ==========
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
PERPLEXITY_REDUCTION_RATIO = 0.9  # 改写后困惑度至少降低10%

# ========== 考题风格关键词 ==========
QUESTION_KEYWORDS = {
    "下列", "求解", "证明", "简述", "是什么", "为什么", "正确的是", "错误的是",
    "若...则", "已知...求", "试述", "分析", "推导", "计算", "判断"
}