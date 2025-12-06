"""
项目通用配置

提供改写与QA阶段共享的基础常量：模型ID、设备、输出目录。
如果你有自定义配置，可以在此文件中修改。
"""

import os

# 基础语言模型（用于困惑度计算与部分嵌入复用）
MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"

# 推理设备
try:
    import torch
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = "cpu"

# 筛选阶段输出目录（供改写阶段自动读取）
OUTPUT_PATH = os.path.join("data", "filter_output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

