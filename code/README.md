# Qwen2.5-0.5B微调与评估工具集

本目录包含用于Qwen2.5-0.5B模型微调和评估的 Python 脚本集合。使用LoRA技术进行高效微调，可自行配置LoRA参数与微调数据采样率，并提供全面的模型评估功能。

## 文件说明

### 1. 训练脚本

#### `train-0.5B.py`
- **功能**: 0.5GB数据微调Qwen2.5-0.5B模型（可选 实验结果对比中未使用）
- **特点**:
  - 使用 LoRA 技术进行高效微调 - 支持 BF16/FP16 混合精度训练 - 数据采样功能，可控制训练数据量
 - 自动检查点恢复和模型保存

#### `train-qa.py`
- **功能**:针对 question-answer 格式的 QA 数据微调(对应数据处理流水线结果文件)
- **数据格式**:支持 `question`和 `answer` 字段的 JSON/JSONL 文件

#### `train-pr.py`
- **功能**: 针对 prompt-response 格式的 QA 数据微调(对应对比实验使用的现有问答对数据集)
- **数据格式**: 支持 `prompt` 和 `response` 字段的 JSON/JSONL 文件

#### `train-io.py`
- **功能**: 针对 instruction-output 格式的 QA 数据微调(对应对比实验使用的现有问答对数据集)
- **数据格式**: 支持 `instruction` 和 `output` 字段的 JSON/JSONL 文件

### 2. 评估脚本

#### `test-fine-tuning.py`
- **功能**: 综合模型评估工具- **评估维度**:
  - 基础生成能力测试 - 问答准确率评估（精确匹配、部分匹配）
  - 事实性问答能力 - 推理问答能力
 - 多轮对话连贯性
  - CMMLU 中文多学科评测 - 困惑度计算
  - ROUGE 和 BLEU 文本质量指标

## 快速开始

### 环境要求

#### 基础依赖安装
```bash
pip install torch transformers peft datasets pandas numpy jieba rouge-chinese nltk scikit-learn
```

#### 完整环境依赖（conda list）
以下是已验证可运行的环境配置：

```bash
#主要依赖包
absl-py==2.3.1
accelerate==0.30.1
datasets==2.14.6
jieba==0.42.1
nltk==3.9.2
numpy==1.26.4
pandas==2.3.3
peft==0.8.2
rouge-chinese==1.0.3
scikit-learn==1.7.2
torch==2.2.2
transformers==4.41.2
tqdm==4.66.1

# 其他相关依赖
filelock==3.20.0
huggingface-hub==0.36.0
tokenizers==0.19.1
safetensors==0.6.2
pyyaml==6.0.3
requests==2.32.5
```

#### Conda 环境创建（推荐）
```bash
conda create -n qwen05 python=3.10.12
conda activate qwen05
pip install -r requirements.txt  # 可创建requirements文件包含上述依赖
```

### 训练步骤

1. **准备数据**: 将训练数据放入相应的数据目录（如 `QA-data`, `QA-data-2`, `QA-data-3`）

2. **配置模型路径**: 修改脚本中的 `model_path` 为相应Qwen2.5-0.5B 模型路径

3. **运行训练**:
```bash
# 使用 question-answer 格式数据训练
python train-qa.py

# 使用 prompt-response 格式数据训练
python train-pr.py

# 使用 instruction-output 格式数据训练
python train-io.py

# 0.5GB 数据微调
python train-0.5B.py
```

### 评估步骤

1. **合并模型** (如果需要):
```bash
python test-fine-tuning.py
# 脚本会自动检测并合并 LoRA 权重
```

2. **运行综合评估**:
```bash
python test-fine-tuning.py
```

## 配置说明

### 训练参数
- `per_device_train_batch_size`: 4-8（根据 GPU 显存调整）
- `learning_rate`: 2e-4 ~ 3e-4
- `num_epochs`: 5-10
- `max_length`: 512-1024### 数据格式要求
所有训练脚本支持 JSON和 JSONL 格式，字段名称根据脚本类型有所不同：
- `train-qa.py`: `question`, `answer`
- `train-pr.py`: `prompt`, `response`
- `train-io.py`: `instruction`, `output`

### 硬件要求
- GPU显存: ≥ 4GB- 系统内存: ≥ 8GB
- 推荐使用支持 BF16 的 GPU（如 RTX 30/40 系列）

## 评估指标

### 问答能力评估
- **精确匹配率**: 生成答案与标准答案完全一致的比例
- **部分匹配率**: 语义相似度 > 0.7 的比例
- **事实性准确率**: 事实性问题的正确率
- **推理准确率**: 推理问题的正确率

### 文本质量指标
- **困惑度**: 语言模型困惑度
- **ROUGE分数**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU分数**: 机器翻译质量评估

### CMMLU 评估
覆盖 57 个中文学科领域，包括：
-人文社科: 中国历史、文学、哲学等
- 自然科学: 数学、物理、化学、生物等
-工程技术: 计算机科学、电子工程等
- 医学法律: 临床知识、法律等

## 训练后模型
### 数据流水线得到QA问答对得到完整模型结果
通过网盘分享的文件：qwen2.5-0.5b-qa-r16.zip
链接: https://pan.baidu.com/s/1YdoWFrxp0DYcOL6LnDWOVA?pwd=bjwq 提取码: bjwq 复制这段内容后打开百度网盘手机App，操作更方便哦 
--来自百度网盘超级会员v9的分享
### 指令-输出问答对得到完整模型结果
通过网盘分享的文件：qwen2.5-0.5b-io-r16.zip
链接: https://pan.baidu.com/s/1V048CRDIEAvVmmWbizUvrQ?pwd=7tdk 提取码: 7tdk 复制这段内容后打开百度网盘手机App，操作更方便哦 
--来自百度网盘超级会员v9的分享
### 提示词-回应问答对得到完整模型结果
通过网盘分享的文件：qwen2.5-0.5b-pr-r16.zip
链接: https://pan.baidu.com/s/1WnkBZQx3l6Bxtgpz7Rsmhw?pwd=khuc 提取码: khuc 复制这段内容后打开百度网盘手机App，操作更方便哦 
--来自百度网盘超级会员v9的分享
