# 高质量中文指令数据集精炼管道

本项目是一个端到端的数据处理流水线，旨在将海量的、原始的中文网络文本，通过多阶段的**筛选、改写与增强、以及指令生成**，转化为可直接用于大型语言模型（LLM）**指令微调（Instruction Fine-Tuning）**的高质量数据集。

整个管道经过深度优化，集成了 **vLLM** 高性能推理引擎，能够在消费级硬件（如NVIDIA 4090）上高效处理TB级数据。

## 流程概览

-   **多层级数据筛选**：集成长度、敏感词、口语化、精确去重（MD5）和语义去重（Minhash LSH）等多道过滤关卡。
-   **AI驱动的质量评估**：使用轻量级模型（`uer/gpt2-chinese-cluecorpussmall`）计算困惑度，智能筛选高质量文本。
-   **双模文本改写与增强**：
    -   **修复模式**：针对高困惑度（不通顺）和高冗余度（啰嗦）的文本进行优化。
    -   **增强模式**：为缺乏上下文的简短陈述句，智能地**注入因果逻辑链**，直接创造推理训练样本。
-   **多样化QA指令生成**：
    -   根据文本内容智能判断提问角度（定义、原因、比较等）。
    -   采用**随机化Prompt模板**策略，生成多样化、不刻板的问答对。
-   **高性能实现**：所有生成任务（改写、QA生成）均采用 **vLLM** 引擎，通过**持续批处理 (Continuous Batching)** 实现数十倍的性能提升。
-   **完全自动化**：通过 `run_pipeline.py` 一键启动，完成从原始数据到最终QA数据集的全流程处理。

## 目录结构

```nlp-work/
├── data/
│   ├── input/                # -> 1. 存放原始数据 (需手动创建)
│   ├── filter_output/               # <- 2. 筛选后的高质量文本
│   ├── rewrite_output/       # <- 3. 改写与增强后的文本
│   └── qa_output/            # <- 4. 最终生成的QA指令数据集
├── config.py                 # 全局共享配置文件
├── filter.py                 # 阶段一：数据筛选脚本
├── rewrite/                  # 阶段二：数据改写与增强模块
│   ├── rewrite.py
│   ├── rewrite_config.py
│   ├── model_utils.py
│   └── prompt_templates.py
├── qa/                       # 阶段三：QA指令生成模块
│   ├── qa_generate.py
│   ├── qa_config.py
│   └── prompt_templates.py
├── run_pipeline.py           # 自动化流水线总控制器
├── finetune_qwen.py          # (示例)下游微调脚本
└── README.md                 # 本说明文件
```
## 环境依赖
Python 3.10+
PyTorch 2.1+
NVIDIA GPU (推荐 Ampere 架构及以上, 如 30/40 系列)
CUDA 12.1+
核心依赖库安装：
```
pip install torch transformers datasets tqdm numpy datasketch psutil
# 强烈推荐安装 vLLM 以获得数十倍的性能提升
pip install vllm
```

## 快速开始
1. 准备原始数据
将你的原始 .jsonl 格式数据（每行一个JSON，至少包含 text 字段）放入 data/input/ 目录下。

```
# 在项目根目录创建所需文件夹
mkdir -p data/input
# 将你的数据文件移动到该目录
mv /path/to/your/data.jsonl data/input/
```
2. (重要) 配置镜像与环境变量
为了避免模型下载缓慢或失败，强烈建议在运行前设置Hugging Face镜像。
在你的终端中执行（该设置仅对当前终端有效）：
```
export HF_ENDPOINT=https://hf-mirror.com
```
3. 配置参数（可选）
你可以根据需求，调整每个阶段的配置文件，以控制数据处理的策略和产出量：
筛选宽松度: filter.py (如 LSH_THRESHOLD)
改写策略: rewrite/rewrite_config.py (如 LOGIC_INJECTION_SAMPLING_RATE)
QA生成策略: qa/qa_config.py (如 BATCH_SIZE_QA, SEMANTIC_SIMILARITY_MIN)
4. 一键启动流水线
```
python run_pipeline.py
```
脚本将依次执行数据筛选 -> 数据改写 -> QA生成。最终的高质量QA数据集将保存在 data/qa_output/qa_pairs.jsonl。

## 各模块详解
### 阶段一：数据筛选 (filter.py)
-   ***目标***：从海量原始数据中，通过一系列严格的规则和AI评估，筛选出干净、无害、高质量的文本。
-   ***核心模型***：uer/gpt2-chinese-cluecorpussmall (用于计算困惑度)。
-   ***工作流程：***
    -   1. 基础过滤：长度、敏感词、口语化内容。
    -   2. 双重去重：MD5（精确）+ Minhash LSH（语义）。
    -   3. AI质检：基于困惑度百分位，保留流畅的现代文和“地道”的古文。
-   ***产出***：data/output/clmmu_kept_data_final.jsonl，一个附加了困惑度等元数据的高质量文本集。

### 阶段二：数据改写 (rewrite.py)
-   ***目标***：对筛选后的文本进行“精加工”，修复表达缺陷并注入逻辑价值。
-   ***核心模型***：Qwen/Qwen1.5-1.8B-Chat (通过 vLLM 加载)。
-   ***工作流程***：
    -   1. 智能诊断：识别“高困惑度”、“高冗余度”或“缺乏逻辑”的文本。
    -   2. 按需改写：根据诊断结果，选择对应的Prompt模板进行优化或增强。
    -   3. 严格质检：确保改写后的文本忠于原意且真正达到了优化/增强的目标。
-   ***产出***：data/rewrite_output/rewritten_data.jsonl，记录了所有文本的改写状态和结果。

### 阶段三：QA生成 (qa_generate.py)
-   ***目标***：将精炼后的陈述性文本，创造性地转化为结构化的问答对，用于指令微调。
-   ***核心模型***：Qwen/Qwen1.5-1.8B-Chat (通过 vLLM 加载)。
-   ***工作流程***：
    -   1. 智能选材：优先使用改写成功的文本。
    -   2. 主题分析：通过关键词匹配，判断文本适合生成的问题类型。
    -   3. 多样化生成：从多种措辞的Prompt模板中随机选择一个，调用vLLM进行批量生成。
    -   4. 多维度质检：对生成的QA对进行格式、长度、内容相关性等多重校验。
-   ***产出***：data/qa_output/qa_pairs.jsonl，最终可用于微调的高质量指令数据集。

## 常见问题 (FAQ)
1. 运行报错 Repo id must be in the form ...
问题: 你在配置文件中（如 rewrite_config.py 或 qa_config.py）将模型ID错误地设置为了一个本地路径。
解决: 请确保所有 MODEL_ID 相关的变量都设置为Hugging Face Hub上的标准模型ID，例如 "Qwen/Qwen1.5-1.8B-Chat"，而不是本地缓存路径。

2. vLLM 报错显存不足 (ValueError: ... KV cache is needed ...)
问题: GPU上同时加载了多个模型，导致vLLM启动时没有足够的连续显存。
解决: 在 vLLM 加载模型的代码中（如 rewrite/model_utils.py -> load_rewrite_model），为 LLM() 添加参数 gpu_memory_utilization 和 max_model_len 来限制其资源占用。

```
model = LLM(
    model=REWRITE_MODEL_ID,
    trust_remote_code=True,
    gpu_memory_utilization=0.7, # 使用70%的显存
    max_model_len=8192          # 限制最大序列长度
)
```
3. 如何处理更大的数据集（>10G）？
当前 filter.py 脚本采用了将中间数据缓存在内存中的策略。对于超大规模数据集，这可能会导致内存溢出。
优化建议: 修改 filter.py，将 preprocess_and_md5_deduplicate 和 minhash_lsh_deduplicate 的结果分块写入临时文件，然后在 layered_perplexity_filter 中逐块读取处理，以将内存占用降低为流式处理。

4. 如何提升最终QA数据集的产出量？
最终产出量主要由 QA生成阶段的成功率 决定。
提升方法:
放宽质检标准: 在 qa/qa_config.py 中，适度降低 SEMANTIC_SIMILARITY_MIN，或放宽 MIN/MAX_*_LEN 的范围。
增加生成次数: 修改 qa/qa_generate.py 的主循环，让每个源文本尝试生成多次QA对。这是最直接的“倍增器”，但会增加处理时间。

☺☺☺其他部分自行理解☺☺☺

