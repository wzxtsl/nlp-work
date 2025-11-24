import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========== 共享函数定义 ==========
def load_model_and_tokenizer(model_path, precision="bf16"):
    """加载 Qwen2.5-0.5B 模型"""
    logger.info(f"加载本地 Qwen2.5-0.5B 模型（{precision}模式）: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"已设置pad_token为: {tokenizer.eos_token}")
    
    # 精度选择
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        logger.info("使用BF16精度（GPU支持）")
    elif precision == "fp16" or (precision == "bf16" and not torch.cuda.is_bf16_supported()):
        dtype = torch.float16
        logger.info("使用FP16精度（BF16不支持时自动降级）")
    else:
        dtype = torch.float32
        logger.warning("未启用混合精度，训练速度可能较慢")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False
    )
    
    model.gradient_checkpointing_enable()
    return model, tokenizer

def setup_lora(model):
    """LoRA配置（适配Qwen2.5-0.5B）"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # 计算可训练参数
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_ratio = (trainable_params / all_param) * 100
    logger.info(f"可训练参数: {trainable_params} / {all_param} ({trainable_ratio:.4f}%)")
    
    if trainable_ratio < 0.01:
        raise ValueError("LoRA未挂载到模型！请检查Qwen2.5-0.5B的模块名是否正确")
    
    return model

def find_latest_checkpoint(output_dir):
    """查找最新的检查点"""
    if not os.path.exists(output_dir):
        return None
        
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoint_dirs:
        return None
        
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
    logger.info(f"找到最新检查点: {latest_checkpoint}")
    return latest_checkpoint

# ========== 数据集类（适配QA数据格式） ==========
class QADataset(Dataset):
    """QA问答数据集加载器（适配question+answer字段）"""
    
    def __init__(self, data_dir, tokenizer, max_length=512, split_ratio=0.95, 
                 mode='train', num_workers=4, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        self.texts = []
        # 支持读取json和jsonl文件
        data_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.json', '.jsonl'))])
        if not data_files:
            raise FileNotFoundError(f"在{data_dir}目录下未找到.json或.jsonl文件")
        logger.info(f"找到 {len(data_files)} 个QA数据文件")
        
        # 多进程加载数据
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            all_texts = []
            for file in data_files:
                file_path = os.path.join(data_dir, file)
                logger.info(f"加载文件: {file_path}")
                # 根据文件后缀选择读取方式
                if file.endswith('.jsonl'):
                    futures = [executor.submit(self._parse_jsonl_line, line) for line in open(file_path, 'r', encoding='utf-8') if line.strip()]
                else:  # .json文件（假设是列表格式）
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data_list = json.load(f)
                    futures = [executor.submit(self._parse_qa_dict, item) for item in data_list if isinstance(item, dict)]
                
                for future in futures:
                    qa_text = future.result()
                    if qa_text:
                        all_texts.append(qa_text)
        
        # 限制最大样本数
        if max_samples and len(all_texts) > max_samples:
            all_texts = all_texts[:max_samples]
        
        if not all_texts:
            raise ValueError("未加载到有效QA数据，请检查数据格式")
        logger.info(f"加载到有效QA样本数: {len(all_texts)}")
        
        # 数据集划分（训练集/验证集）
        split_idx = int(len(all_texts) * split_ratio)
        if mode == 'train':
            self.texts = all_texts[:split_idx]
        else:
            self.texts = all_texts[split_idx:]
        
        logger.info(f"最终{mode}集大小: {len(self.texts)}")
    
    def _parse_jsonl_line(self, line):
        """解析jsonl文件的单行数据"""
        try:
            data = json.loads(line.strip())
            return self._parse_qa_dict(data)
        except json.JSONDecodeError:
            logger.warning(f"跳过无效jsonl行: {line[:50]}...")
            return None
    
    def _parse_qa_dict(self, data):
        """解析QA字典（提取question和answer并组装）"""
        # 提取question和answer字段（兼容大小写）
        question = data.get('question') or data.get('Question')
        answer = data.get('answer') or data.get('Answer')
        
        # 过滤无效数据
        if not (isinstance(question, str) and isinstance(answer, str)):
            return None
        if len(question) < 5 or len(answer) < 5:  # 过滤过短的问答
            return None
        if '...' in answer[:10]:  # 过滤未完成的答案
            return None
        
        # 组装QA格式（让模型学习“问题→答案”的映射）
        qa_text = f"问题：{question.strip()} 答案：{answer.strip()}{self.tokenizer.eos_token}"
        return qa_text
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
        }

# ========== 主训练函数（适配QA数据） ==========
def qa_finetuning():
    """QA数据微调Qwen2.5-0.5B配置"""
    config = {
        "data_dir": "./QA-data",  # 你的QA数据目录
        "output_dir": "./qwen2.5-0.5b-qa-lora-r32",  # 输出目录（可自定义）
        "model_path": "./Qwen2.5-0.5B",  # 你的模型路径
        "max_length": 512,  # 根据QA长度调整（建议512-1024）
        "split_ratio": 0.95,  # 训练集:验证集=95:5
        "num_epochs": 5,  # QA数据建议增加epoch（8-10轮）
        "precision": "bf16",
        "resume_from_checkpoint": True,
        "max_steps": -1,  # 设为-1表示按epoch训练（优先num_epochs）
        "max_samples": None  # 不限制样本数（如果数据多可设为10000等）
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 优化后的QA训练参数
    train_params = {
        "per_device_train_batch_size": 4,  # 根据GPU显存调整（0.5B模型建议4-8）
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 2,  # 显存不足时增大
        "learning_rate": 2e-4,  # QA微调建议学习率（2e-4~3e-4）
        "warmup_ratio": 0.05,  # 预热比例（QA数据无需过长预热）
        "logging_steps": 20,
        "save_steps": 200,
        "eval_steps": 200,
        "save_total_limit": 3,  # 保留最新3个检查点
    }
    
    # 检查GPU
    if not torch.cuda.is_available():
        logger.error("未检测到GPU，无法训练（Qwen2.5-0.5B需至少4GB显存）")
        return
    
    # 加载模型和LoRA
    try:
        model, tokenizer = load_model_and_tokenizer(config["model_path"], precision=config["precision"])
        model = setup_lora(model)
    except Exception as e:
        logger.error(f"模型初始化失败: {e}", exc_info=True)
        return
    
    # 加载QA数据集
    try:
        logger.info("加载QA训练集...")
        train_dataset = QADataset(
            data_dir=config["data_dir"],
            tokenizer=tokenizer,
            max_length=config["max_length"],
            mode='train',
            num_workers=4,
            max_samples=config["max_samples"]
        )
        logger.info("加载QA验证集...")
        val_dataset = QADataset(
            data_dir=config["data_dir"],
            tokenizer=tokenizer,
            max_length=config["max_length"],
            mode='val', 
            num_workers=4,
            max_samples=config.get("max_samples_val", 500)  # 验证集最多500样本
        )
    except Exception as e:
        logger.error(f"数据集加载失败: {e}", exc_info=True)
        return
    
    # 计算训练信息
    total_batch_size = train_params["per_device_train_batch_size"] * train_params["gradient_accumulation_steps"]
    steps_per_epoch = len(train_dataset) // total_batch_size
    logger.info("训练信息:")
    logger.info(f"  - 训练样本数: {len(train_dataset)}")
    logger.info(f"  - 验证样本数: {len(val_dataset)}")
    logger.info(f"  - 每epoch步数: {steps_per_epoch}")
    logger.info(f"  - 总epoch数: {config['num_epochs']}")
    logger.info(f"  - 有效batch size: {total_batch_size}")
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=False,
        num_train_epochs=config["num_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=train_params["per_device_train_batch_size"],
        per_device_eval_batch_size=train_params["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_params["gradient_accumulation_steps"],
        warmup_ratio=train_params["warmup_ratio"],
        logging_steps=train_params["logging_steps"],
        eval_steps=train_params["eval_steps"],
        save_steps=train_params["save_steps"],
        evaluation_strategy="steps",
        learning_rate=train_params["learning_rate"],
        fp16=(config["precision"] == "fp16"),
        bf16=(config["precision"] == "bf16"),
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        save_total_limit=train_params["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        remove_unused_columns=True,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_dir=os.path.join(config["output_dir"], "logs"),
    )
    
    # 数据整理器（LM任务）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 训练执行
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 检查点恢复
    resume_from_checkpoint = None
    if config["resume_from_checkpoint"]:
        latest_checkpoint = find_latest_checkpoint(config["output_dir"])
        if latest_checkpoint:
            resume_from_checkpoint = latest_checkpoint
            logger.info(f"从检查点恢复训练: {resume_from_checkpoint}")
        else:
            logger.info("未找到检查点，从头开始训练")
    
    try:
        logger.info("开始QA数据微调训练...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        logger.info("训练被中断，保存当前进度...")
        trainer.save_model()
        return
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        return
    
    # 保存最终模型
    logger.info("训练完成，保存最终模型...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    # 保存训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["val_samples"] = len(val_dataset)
    with open(os.path.join(config["output_dir"], "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"模型保存至: {config['output_dir']}")
    logger.info("QA数据微调完成！")

if __name__ == "__main__":
    qa_finetuning()