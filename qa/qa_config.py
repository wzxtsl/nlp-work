import sys
from pathlib import Path
import os

# 允许从项目根导入已有 config（与 rewrite_config 同结构）
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config import MODEL_ID, DEVICE, OUTPUT_PATH  # 复用通用配置
except Exception:
    # 兜底：在编辑器无法解析 config 时给出合理默认值（运行期若有 config 将被上层引用覆盖）
    MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"
    try:
        import torch  # 仅用于检测设备
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception:
        DEVICE = "cpu"
    OUTPUT_PATH = os.path.join("data", "output")

# ========== 路径配置 ==========
# QA 输入来自改写阶段的结果；如果改写失败则回退到原文本
REWRITTEN_INPUT_PATH = os.path.join("data", "rewrite_output", "rewritten_data.jsonl")
QA_OUTPUT_PATH = os.path.join("data", "qa_output", "qa_pairs.jsonl")
QA_FAILED_PATH = os.path.join("data", "qa_output", "qa_failed.jsonl")
QA_LOG_PATH = os.path.join("data", "qa_output", "qa_log.log")
os.makedirs(os.path.dirname(QA_OUTPUT_PATH), exist_ok=True)

# ========== 模型配置（复用改写模型或基础模型） ==========
# 如果希望与改写一致，可将其设置为与 REWRITE_MODEL_ID 一致；否则默认使用基础模型进行生成
QA_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"  # 与改写一致，便于风格统一
MAX_NEW_TOKENS_QA = 512
TEMPERATURE_QA = 0.7
TOP_P_QA = 0.95
BATCH_SIZE_QA = 32  # QA生成通常较短，可适度提高批量
MAX_SOURCE_CHARS = 1600  # 过长文本先做截断，避免提示词过长

# ========== 过滤与质量控制 ==========
MIN_QUESTION_LEN = 8
MAX_QUESTION_LEN = 180
MIN_ANSWER_LEN = 4
MAX_ANSWER_LEN = 800
REQUIRED_CHINESE_PUNCT = "？"  # 优先生成以问号结尾的中文问题
SEMANTIC_SIMILARITY_MIN = 0.55  # 问题与原文语义相关度（粗略阈值，复用 embedding）

# ========== 问题类型识别用关键词 ==========
TYPE_KEYWORDS = {
    "definition": ["定义", "是指", "称为", "叫做", "是什么"],
    "reason": ["为什么", "原因", "原理", "依据"],
    "compare": ["区别", "不同", "比较", "差异"],
    "apply": ["如何", "怎样", "怎么", "步骤", "方法"],
    "calculate": ["计算", "求", "推导", "公式"],
}

# 为不同类型选择不同 prompt 模板 key 的映射
TYPE_PROMPT_MAPPING = {
    "definition": "definition_q",
    "reason": "reason_q",
    "compare": "compare_q",
    "apply": "apply_q",
    "calculate": "calculate_q",
    "fallback": "generic_q",
}
