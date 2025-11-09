# 文本改写改写工具使用说明


## 工具简介
本工具用于对文本数据进行自动改写优化，主要针对高困惑度文本（提升流畅度）和冗余文本（精简表达），同时确保改写前后语义一致。支持批量处理，可直接对接上游筛选后的数据集，输出包含完整改写结果的结构化数据。


## 环境依赖
- Python 3.8+
- 必要依赖库：
  ```bash
  pip install torch transformers tqdm numpy
  ```
- 建议配置：
  - 显存：≥14GB（支持批量处理）
  - 磁盘空间：≥10GB（全量数据处理）


## 目录结构
```
nlp-work/
├── rewrite/
│   ├── rewrite.py          # 主程序（核心改写逻辑）
│   ├── rewrite_config.py   # 配置参数（模型、阈值等）
│   ├── model_utils.py      # 模型加载/生成工具函数
│   └── prompt_templates.py # 改写提示词模板
├── filter.py               # 数据筛选
├── run_pipeline.py         # 管道
└── README.md               # 本说明文件
```


## 快速开始
1. **准备输入数据**：  
   将上游筛选后的JSONL格式数据放入 `data/input/` 目录，确保每条数据包含：
   - `id`：唯一标识
   - `text`：原始文本
   - `perplexity`：困惑度（用于筛选高困惑度文本）
   - `text_type`：文本类型（`modern_chinese` 或 `classic_chinese`）

2. **运行测试模式**：  
   ```bash
   python run_pipeline.py 
   ```

3. **查看结果**：  
   输出文件位于 `data/rewrite_output/rewritten_data.jsonl`，包含所有数据的改写状态（未改写/成功/失败）及详细指标。


## 核心功能配置
### 1. 切换模型
如需更换改写模型或困惑度计算模型，修改 `rewrite_config.py` 中的模型ID：
```python
# rewrite_config.py
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"  # 同时用于改写和困惑度计算
REWRITE_MODEL_ID = MODEL_ID          # 改写模型（可单独指定）
```
- 支持Hugging Face Hub上的所有因果语言模型（如 `baichuan-7b`、`chatglm3-6b` 等）
- 模型切换后需确保 `model_utils.py` 中的加载逻辑兼容（主要适配 `generate_rewrite` 函数）


### 2. 调整批量大小（Batch Size）
根据显存大小修改批量处理规模，在 `rewrite.py` 主流程中调整：
```python
# rewrite.py 中 main 函数内
adjusted_batch_size = 6  # 14GB显存推荐值
# 若显存≥20GB，可尝试增大至 8-10；若显存不足，降低至 2-4
```


### 3. 改写阈值配置
所有阈值参数在 `rewrite_config.py` 中定义，可根据需求调整：
```python
# rewrite_config.py
MODERN_PERPLEXITY_PERCENTILE = 80  # 高困惑度阈值（取现代文困惑度的80%分位数）
SEMANTIC_SIMILARITY_THRESHOLD = 0.8  # 语义相似度最低阈值（低于此值视为失败）
PERPLEXITY_REDUCTION_RATIO = 0.9  # 困惑度降低比例（需降至原90%以下）
REDUNDANCY_RATIO_THRESHOLD = 0.2  # 冗余度阈值（高于此值视为冗余文本）
SKIP_CLASSIC_CHINESE = True  # 是否跳过古文（True/False）
```


### 4. 切换全量/测试模式
- **测试模式**（默认）：仅处理前10000条数据，适合快速验证效果  
  无需修改代码，直接运行即可。

- **全量模式**：处理所有数据，适合正式运行  
  打开 `rewrite.py`，取消注释"完整模式"代码块，并注释"测试模式"代码块：
  ```python
  # 注释测试模式代码
  # print(f"开始改写测试（前10000条数据）...")
  # ...（测试模式处理逻辑）

  # 取消注释全量模式代码
  print(f"开始全量数据改写...")
  # ...（全量模式处理逻辑）
  ```


## 输出结果说明
输出文件为JSONL格式（每行一条JSON），包含以下核心字段：
| 字段名               | 说明                                  |
|----------------------|---------------------------------------|
| `id`                 | 原始文本唯一标识                      |
| `original_text`      | 原始文本内容                          |
| `rewritten_text`     | 改写后的文本（未改写则为None）        |
| `status`             | 处理状态（`skipped`/`success`/`failed`/`error`） |
| `reason`             | 状态说明（如"古文无需改写"、"困惑度未降低"等） |
| `orig_perplexity`    | 原始文本困惑度（仅高困惑度文本有值）  |
| `rew_perplexity`     | 改写后文本困惑度（仅成功案例有值）    |


## 常见问题解决
1. **显存不足（OOM）**：  
   - 降低 `adjusted_batch_size` 至4或2  
   - 启用4bit量化（在 `model_utils.py` 的 `load_rewrite_model` 中添加 `load_in_4bit=True`）

2. **磁盘空间不足**：  
   - 清理输出目录历史文件  
   - 启用空间优化模式（使用精简输出字段的代码版本）

3. **模型加载失败**：  
   - 检查网络连接（首次运行需下载模型）  
   - 确认模型ID正确，且支持因果语言模型（CausalLM）格式  
   - 添加 `trust_remote_code=True` 加载自定义模型

4. **改写效果不佳**：  
   - 调整 `prompt_templates.py` 中的提示词模板，增强指令明确性  
   - 降低 `PERPLEXITY_REDUCTION_RATIO` 阈值（如0.85），放宽困惑度要求  


## 使用提示
1. **测试优先**：新配置或新模型下，建议先运行测试模式（10000条数据）验证效果，再执行全量处理。
2. **日志查看**：运行过程中的警告和错误会记录在 `data/rewrite_output/rewrite_log.log`，可用于排查问题。
3. **参数备份**：修改配置前建议备份 `rewrite_config.py`，避免参数混乱。
4. **批量调整**：全量处理大规模数据时，可分时段运行（工具支持断点续跑需额外开发，当前建议一次性运行）。

本文件后半部分为rewrite的说明，其余文件自行理解☺☺☺。
