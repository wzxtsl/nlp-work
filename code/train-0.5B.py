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
    """LoRA配置"""
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
        raise ValueError("LoRA未挂载到模型！请检查模块名是否正确")
    
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

def get_training_status(checkpoint_path):
    """获取检查点的训练状态"""
    trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_file):
        with open(trainer_state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
            return state
    return None

# ========== 数据集类 ==========

class SampledCCI3HQDataset(Dataset):
    """采样数据集加载器 - 只加载部分数据"""
    
    def __init__(self, data_dir, tokenizer, max_length=512, split_ratio=0.95, 
                 mode='train', num_workers=4, sampling_ratio=0.1, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.sampling_ratio = sampling_ratio
        
        self.texts = []
        jsonl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
        if not jsonl_files:
            raise FileNotFoundError(f"在{data_dir}目录下未找到.jsonl文件")
        logger.info(f"找到 {len(jsonl_files)} 个数据文件，采样比例: {sampling_ratio}")
        
        # 计算目标样本数
        total_estimated_samples = 0
        for jsonl_file in jsonl_files:
            file_path = os.path.join(data_dir, jsonl_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            total_estimated_samples += line_count
        
        target_samples = int(total_estimated_samples * sampling_ratio)
        if max_samples:
            target_samples = min(target_samples, max_samples)
        
        logger.info(f"目标采样数量: {target_samples}")
        
        # 多进程加载并采样数据
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            all_texts = []
            for jsonl_file in jsonl_files:
                file_path = os.path.join(data_dir, jsonl_file)
                logger.info(f"加载文件: {file_path}")
                futures = [executor.submit(self._load_line, line) for line in open(file_path, 'r', encoding='utf-8') if line.strip()]
                for future in futures:
                    text = future.result()
                    if text:
                        all_texts.append(text)
            
            # 随机采样
            if len(all_texts) > target_samples:
                indices = np.random.choice(len(all_texts), target_samples, replace=False)
                self.texts = [all_texts[i] for i in indices]
            else:
                self.texts = all_texts
        
        if not self.texts:
            raise ValueError("未加载到有效文本数据")
        logger.info(f"采样后 {mode}集大小: {len(self.texts)}")
        
        # 数据集划分
        split_idx = int(len(self.texts) * split_ratio)
        if mode == 'train':
            self.texts = self.texts[:split_idx]
        else:
            self.texts = self.texts[split_idx:]
        
        logger.info(f"最终{mode}集大小: {len(self.texts)}")
    
    def _load_line(self, line):
        try:
            data = json.loads(line.strip())
            text = self.extract_text(data)
            return text if text and len(text) > 10 else None
        except json.JSONDecodeError:
            return None
    
    def extract_text(self, data):
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            for field in ['text', 'content', 'article', 'prompt', 'answer']:
                if field in data and isinstance(data[field], str) and len(data[field]) > 10:
                    return data[field]
            all_texts = [str(v) for k, v in data.items() if isinstance(v, str) and len(v) > 5]
            return " ".join(all_texts) if all_texts else None
        return None
    
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

# ========== 主训练函数 ==========

def main_1gb_finetuning():
    """1GB数据微调专用配置"""
    config = {
        "data_dir": "./data",
        "output_dir": "/root/shared-nvme/kmr/NLP/qwen2.5-0.5b-05gb-lora-r32",
        "model_path": "./Qwen2.5-0.5B",
        "max_length": 512,
        "split_ratio": 0.95,
        "num_epochs": 5,           # 增加epoch数
        "precision": "bf16",
        "resume_from_checkpoint": True,
        "max_steps": 8000,         # 减少总步数
        "sampling_ratio": 0.05,     # 使用10%数据 ≈ 1GB
        "max_samples": 25000       # 最大样本数限制
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 优化后的训练参数
    train_params = {
        "per_device_train_batch_size": 8,    # 增大batch size
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,     # 减少累积步数
        "learning_rate": 3e-4,               # 稍高学习率
        "warmup_ratio": 0.1,                 # 使用比例而非固定步数
        "logging_steps": 50,
        "save_steps": 500,                   # 更频繁保存
        "eval_steps": 500,                   # 更频繁评估
        "save_total_limit": 5,               # 减少检查点数量
        "max_steps": config["max_steps"],
    }
    
    # 检查GPU
    if not torch.cuda.is_available():
        logger.error("未检测到GPU，无法训练")
        return
    
    # 加载模型和LoRA
    try:
        model, tokenizer = load_model_and_tokenizer(config["model_path"], precision=config["precision"])
        model = setup_lora(model)
    except Exception as e:
        logger.error(f"模型初始化失败: {e}", exc_info=True)
        return
    
    # 加载采样数据集
    try:
        logger.info("加载采样训练集...")
        train_dataset = SampledCCI3HQDataset(
            data_dir=config["data_dir"],
            tokenizer=tokenizer,
            max_length=config["max_length"],
            mode='train',
            num_workers=4,
            sampling_ratio=config["sampling_ratio"],
            max_samples=config["max_samples"]
        )
        logger.info("加载采样验证集...")
        val_dataset = SampledCCI3HQDataset(
            data_dir=config["data_dir"],
            tokenizer=tokenizer,
            max_length=config["max_length"],
            mode='val', 
            num_workers=4,
            sampling_ratio=config["sampling_ratio"],
            max_samples=config.get("max_samples_val", 5000)  # 验证集样本限制
        )
    except Exception as e:
        logger.error(f"数据集加载失败: {e}", exc_info=True)
        return
    
    # 计算训练信息
    total_batch_size = train_params["per_device_train_batch_size"] * train_params["gradient_accumulation_steps"]
    steps_per_epoch = len(train_dataset) // total_batch_size
    total_steps = steps_per_epoch * config["num_epochs"]
    
    # 如果计算的总步数小于配置的max_steps，使用计算值
    effective_max_steps = min(total_steps, train_params["max_steps"])
    
    logger.info("训练信息:")
    logger.info(f"  - 训练样本数: {len(train_dataset)}")
    logger.info(f"  - 验证样本数: {len(val_dataset)}")
    logger.info(f"  - 每epoch步数: {steps_per_epoch}")
    logger.info(f"  - 总训练步数: {effective_max_steps}")
    logger.info(f"  - 有效batch size: {total_batch_size}")
    logger.info(f"  - 检查点频率: 每 {train_params['save_steps']} 步")
    logger.info(f"  - 评估频率: 每 {train_params['eval_steps']} 步")
    
    # 计算预热步数
    warmup_steps = int(effective_max_steps * train_params["warmup_ratio"])
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=False,
        num_train_epochs=config["num_epochs"],
        max_steps=effective_max_steps,
        per_device_train_batch_size=train_params["per_device_train_batch_size"],
        per_device_eval_batch_size=train_params["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_params["gradient_accumulation_steps"],
        warmup_steps=warmup_steps,
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
        save_strategy="steps",
        logging_dir=os.path.join(config["output_dir"], "logs"),
    )
    
    # 数据整理器
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
    
    # 检查点恢复逻辑
    resume_from_checkpoint = None
    if config["resume_from_checkpoint"]:
        latest_checkpoint = find_latest_checkpoint(config["output_dir"])
        if latest_checkpoint:
            resume_from_checkpoint = latest_checkpoint
            logger.info(f"将从检查点恢复训练: {resume_from_checkpoint}")
        else:
            logger.info("未找到检查点，从头开始训练")
    
    try:
        logger.info("开始1GB数据微调训练...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断，保存当前进度...")
        trainer.save_model()
        return
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        return
    
    # 保存最终结果
    logger.info("训练完成，保存最终模型...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["val_samples"] = len(val_dataset)
    metrics["sampling_ratio"] = config["sampling_ratio"]
    
    with open(os.path.join(config["output_dir"], "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"模型保存至: {config['output_dir']}")
    logger.info("1GB数据微调完成！")

if __name__ == "__main__":
    main_1gb_finetuning()
