# QA 生成模块

本模块基于筛选与改写后的文本，自动生成高质量的中文问答对，输出为 JSONL 便于后续训练或评测。

## 流程概览

- 输入：`data/rewrite_output/rewritten_data.jsonl`
  - 字段：
    - `rewritten_text`（优先使用）
    - `original_text`（当改写失败时回退）
    - `id`（可选）
- 生成：根据源文本内容识别问题类型，使用相应 Prompt 生成一问一答。
- 质控：
  - 问题必须有中文问号（“？”），并满足长度阈值；
  - 答案长度在阈值范围；
  - 若可用，使用已有 embedding 粗测问答与原文的语义相关度（阈值可配）。
- 输出：
  - 成功问答：`data/qa_output/qa_pairs.jsonl`
  - 失败记录：`data/qa_output/qa_failed.jsonl`
  - 日志：`data/qa_output/qa_log.log`

## 目录结构

```
qa/
  ├─ __init__.py
  ├─ qa_config.py         # 配置：路径、模型、阈值
  ├─ prompt_templates.py  # QA 生成模板
  ├─ qa_generate.py       # 主流程脚本
  └─ README.md            # 本说明
```

## 依赖

- transformers
- torch（或兼容的后端）
- tqdm

建议使用与改写相同的模型（默认 Qwen1.5-1.8B-Chat，4-bit 加载）。

## 运行方式

- 集成运行（推荐）：
  - 直接运行项目根下的 `run_pipeline.py`，会在筛选→改写后自动执行 QA 生成。
- 单独运行 QA：

```powershell
# 在项目根目录 nlp-work\nlp-work 下
python qa\qa_generate.py
```

## 配置说明（qa_config.py）

- 路径：
  - `REWRITTEN_INPUT_PATH`：改写结果输入（默认 `data/rewrite_output/rewritten_data.jsonl`）
  - `QA_OUTPUT_PATH`：成功问答输出
  - `QA_FAILED_PATH`：失败记录输出
- 模型与生成：
  - `QA_MODEL_ID`：QA 生成用模型（默认与改写一致）
  - `MAX_NEW_TOKENS_QA`、`TEMPERATURE_QA`、`TOP_P_QA`、`BATCH_SIZE_QA`
- 过滤/质控：
  - `MIN_QUESTION_LEN`、`MAX_QUESTION_LEN`、`MIN_ANSWER_LEN`、`MAX_ANSWER_LEN`
  - `REQUIRED_CHINESE_PUNCT`：要求问题含“？”
  - `SEMANTIC_SIMILARITY_MIN`：问答与原文的相关性阈值（需可用的 embedding）

## 输入格式示例（rewritten_data.jsonl）

```json
{"id": "abc123", "original_text": "牛顿第二定律表述...", "rewritten_text": "将牛顿第二定律更规范地表述为...", "status": "success"}
{"id": "def456", "original_text": "勾股定理...", "rewritten_text": null, "status": "skipped"}
```

## 输出格式示例（qa_pairs.jsonl）

```json
{"id": "abc123", "question": "牛顿第二定律的数学表达式是什么？", "answer": "F = m a。", "status": "success", "question_type": "definition", "prompt_key": "definition_q"}
```

## 常见问题（FAQ）

1. 运行报缺少 `torch/transformers/tqdm`？
   - 请先安装依赖，再运行：

```powershell
pip install transformers torch tqdm
```

2. 模型显存不够？
   - 默认 4-bit 加载，若仍不足可改小模型或降低 `MAX_NEW_TOKENS_QA`，或切换到 CPU（速度较慢）。

3. 想要每条文本生成多道题？
   - 可在 `qa_generate.py` 中循环生成多次并去重，或扩展 Prompt 与解析逻辑输出多问多答。

4. 如何支持选择题/判断题？
   - 在 `prompt_templates.py` 中新增相应模板，并在 `detect_question_type` 与 `TYPE_PROMPT_MAPPING` 中加入映射即可。
