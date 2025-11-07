import os
import json
import torch
import logging
import mmap
import time
import hashlib
import psutil
import threading
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

# 恢复 matplotlib 用于生成困惑度分布图（关键）
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')  # 无GUI环境兼容（避免报错）
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ 未安装 matplotlib，无法生成困惑度分布图")
    MATPLOTLIB_AVAILABLE = False

# 忽略无关警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ========== 核心配置（保留核心文件+分布图） ==========
INPUT_DIR = "data"
INTERMEDIATE_PATH = "data/intermediate"
OUTPUT_PATH = "data/output"
os.makedirs(INTERMEDIATE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 复用中间文件配置（必须启用）
USE_EXISTING_INTERMEDIATE = True
EXISTING_MINHASH_FILE = os.path.join(INTERMEDIATE_PATH, "minhash_dedup_with_sensitive_filter.jsonl")

# 批量配置（保持不变）
BATCH_SIZE_PREPROCESS = 1024
BATCH_SIZE_PERPLEXITY = 32
BATCH_SIZE_MINHASH = 5000

# 基础配置（保持不变，兼容中间文件）
MIN_CHAR_LEN = 8
MAX_CHAR_LEN = 12000
MAX_SEQ_LENGTH = 512
MINHASH_NUM_PERM = 128
LSH_THRESHOLD = 0.76

# 模型配置（保持不变）
MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== 核心筛选配置（保持不变，兼容中间文件） ==========
COLLOQUIAL_WORDS = [
    "卧槽", "牛逼", "哈哈哈", "嘻嘻", "嘿嘿", "老铁", "懂吧", "给力", "666", "yyds",
    "绝绝子", "家人们", "谁懂啊", "救命", "哭死", "笑不活", "栓Q", "拿捏", "破防", "躺平",
    "绝了", "无语", "服了", "凑活", "咋整", "唠嗑", "侃大山", "扯犊子", "瞎逼逼", "逼逼赖赖",
    "磨磨唧唧", "叽叽歪歪", "碎碎念", "吐槽", "怼人", "杠精", "内卷", "摆烂", "摸鱼",
    "划水", "打工人", "干饭人", "尾款人", "工具人", "冤种", "显眼包", "社恐", "社牛", "社死",
    "emo", "佛系", "卷王", "摆烂式", "摸鱼式", "划水式", "躺平式", "敷衍式", "糊弄学", "PUA",
    "CPU", "KTV", "yyds", "awsl", "绝绝子", "YYDS", "栓Q", "拿捏了", "破防了"
]

SENSITIVE_KEYWORDS = {
    "色情相关": ["色情", "黄色", "裸聊", "性交易", "嫖娼", "卖淫", "淫荡", "色情视频", "色情图片", "AV",
                "三级片", "春宫", "艳照", "露骨", "性行为", "性器官", "手淫", "包养", "小三",
                "二奶", "情夫", "情妇", "不正当关系", "一夜情", "约炮", "性服务", "色情直播", "色情小说"],
    "暴力相关": ["杀人", "抢劫", "强奸", "绑架", "斗殴", "故意伤害", "杀人放火", "爆炸", "投毒",
                "凶器", "枪支", "弹药", "管制刀具", "暴力", "血腥", "恐怖", "虐杀", "虐待", "施暴",
                "殴打", "群殴", "互殴", "寻衅滋事", "聚众斗殴", "故意伤害", "故意杀人", "抢劫财物"],
    "毒品相关": ["毒品", "大麻", "海洛因", "冰毒", "可卡因", "摇头丸", "K粉", "鸦片", "吗啡", "杜冷丁",
                "吸毒", "贩毒", "制毒", "吸毒者", "毒贩", "毒品交易", "毒品运输", "毒品走私"],
    "政治敏感": ["敏感政治人物", "政治敏感事件", "颠覆", "分裂", "叛国", "暴动", "骚乱", "非法集会",
                "反动", "反政府", "反社会", "极端主义", "恐怖主义", "邪教", "法轮功", "台独", "港独", "疆独"],
    "其他敏感": ["赌博", "诈骗", "传销", "非法集资", "洗钱", "偷税漏税", "贪污腐败", "行贿受贿",
                "假币", "非法交易", "黑客", "入侵", "病毒", "盗号", "诈骗短信", "诈骗电话"]
}

ACADEMIC_PATTERNS = [
    r"[A-Za-z0-9]=.*[A-Za-z0-9]", r"定义[:：]", r"定理", r"公理", r"命题", r"推论", r"原理", r"方法", r"实验", r"分析", r"结论",
]
ACADEMIC_REQUIRE = False

# 全局统计变量（只保留核心统计项）
stats = {
    "final_kept": 0,
    "classic_chinese_kept": 0,
    "modern_chinese_kept": 0,
    "perplexity_filtered": 0,
    "stage_time": {}
}

# 监控线程变量（保持不变）
monitor_running = True
gpu_util = 0
cpu_mem = 0

# ========== 工具函数（保持不变，兼容中间文件） ==========
def get_gpu_utilization():
    try:
        result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
        return int(result.strip().split("\n")[0]) if result else 0
    except:
        return 0

def get_cpu_memory():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)

def monitor_thread():
    global monitor_running, gpu_util, cpu_mem
    logging.info("🔍 监控线程启动（每30秒更新GPU/内存状态）")
    while monitor_running:
        gpu_util = get_gpu_utilization()
        cpu_mem = get_cpu_memory()
        progress = (stats["final_kept"] / (stats["final_kept"] + stats["perplexity_filtered"]) * 100) if (stats["final_kept"] + stats["perplexity_filtered"]) > 0 else 0.0
        logging.info(
            f"📊 监控状态 - GPU利用率：{gpu_util}% | CPU内存：{cpu_mem}MB | "
            f"已保留：{stats['final_kept']} | 已过滤：{stats['perplexity_filtered']} | 进度：{progress:.1f}%"
        )
        time.sleep(30)
    logging.info("🔍 监控线程停止")

# ========== 核心筛选工具函数（保持不变，兼容中间文件） ==========
def is_colloquial(text):
    for word in COLLOQUIAL_WORDS:
        if word in text:
            return True
    if re.search(r"[！？。,，；;：:]{3,}", text):
        return True
    colloquial_patterns = [
        r"[我你他她它]（们）?[也都还就才又再]?[不没没什么没什么大不了]",
        r"[这那哪]（个些）?[也都还就才又再]?[不没没什么没什么大不了]",
        r"^[哈哈嘿嘿嘻嘻呵呵]+"
    ]
    if any(re.search(pattern, text) for pattern in colloquial_patterns):
        return True
    return False

def is_sensitive(text):
    for category, words in SENSITIVE_KEYWORDS.items():
        for word in words:
            if word in text:
                return True
    sensitive_patterns = [
        r"出售.*(色情|AV|三级片)",
        r"(嫖娼|卖淫|性交易).*(价格|联系方式|地点)",
        r"(杀人|抢劫|绑架).*(方法|教程|工具)",
        r"(毒品|大麻|冰毒).*(购买|出售|运输)",
        r"(台独|港独|疆独).*(支持|宣传|分裂)"
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in sensitive_patterns):
        return True
    return False

def has_academic_features(text):
    return any(re.search(pattern, text) for pattern in ACADEMIC_PATTERNS)

# ========== 古文检测函数（保持不变） ==========
def is_classic_chinese(text):
    classic_words = [
        '之', '乎', '者', '也', '曰', '吾', '汝', '尔', '乃', '兮', '矣', '哉', '耶', '欤', '焉', '乎', '其', '而', '以', '于',
        '夫', '盖', '则', '且', '若', '何', '孰', '安', '孰与', '所以', '所', '可', '能', '必', '当', '应',
        '夏', '商', '周', '秦', '汉', '魏', '蜀', '吴', '晋', '隋', '唐', '宋', '元', '明', '清',
        '黄帝', '炎帝', '尧', '舜', '禹', '汤', '文王', '武王', '周公', '孔子', '孟子', '老子', '庄子', '墨子', '荀子', '韩非子',
        '《论语》', '《孟子》', '《大学》', '《中庸》', '《诗经》', '《尚书》', '《礼记》', '《周易》', '《春秋》',
        '《道德经》', '《庄子》', '《墨子》', '《荀子》', '《韩非子》', '《史记》', '《汉书》', '《后汉书》', '《三国志》',
        '呜呼', '嗟夫', '盖闻', '窃以为', '臣闻', '圣王', '贤君', '忠臣', '义士', '孝子', '烈女'
    ]
    modern_words = ["手机", "互联网", "电脑", "微信", "支付宝", "快递", "高铁", "空调", "电视", "网络", "APP",
                   "微博", "抖音", "快手", "直播", "电商", "网购", "外卖", "打车", "共享单车", "5G", "WiFi"]
    
    for word in modern_words:
        if word in text:
            return False
    
    total_chars = len(text)
    if total_chars == 0:
        return False
    
    classic_char_count = 0
    for word in classic_words:
        classic_char_count += text.count(word)
    density_threshold = 0.02
    
    classic_patterns = [
        r'^[\u4e00-\u9fff]{1,5}曰', r'^昔者', r'^初', r'^当是时', r'^于是', r'^呜呼', r'^嗟夫',
        r'^盖闻', r'^窃以为', r'^臣闻', r'^圣王', r'^贤君', r'^忠臣', r'^义士'
    ]
    pattern_match = any(re.match(pattern, text) for pattern in classic_patterns)
    
    is_classic = (classic_char_count / total_chars > density_threshold) or pattern_match
    return is_classic

# ========== 困惑度计算函数（保持不变） ==========
def load_perplexity_model():
    logging.info("📥 加载模型计算困惑度")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    logging.info(f"✅ 模型加载完成 | 耗时：{round(time.time() - start_time, 2)}秒")
    return tokenizer, model

def calculate_perplexity_batch_optimized(texts, tokenizer, model):
    try:
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            return_tensors="pt"
        ).to(DEVICE)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            perplexities = []
            for i in range(batch_size):
                valid_mask = (input_ids[i] != tokenizer.pad_token_id)
                valid_token_ids = input_ids[i][valid_mask]
                
                if len(valid_token_ids) <= 1:
                    perplexities.append(float('inf'))
                    continue
                
                shift_logits = logits[i, :-1, :].contiguous()
                shift_labels = valid_token_ids[1:].contiguous()
                valid_len = len(shift_labels)
                shift_logits = shift_logits[:valid_len, :]
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
            
            return perplexities
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.warning("⚠️ GPU内存不足，减小批量大小")
            half_size = len(texts) // 2
            if half_size >= 1:
                perplexities1 = calculate_perplexity_batch_optimized(texts[:half_size], tokenizer, model)
                perplexities2 = calculate_perplexity_batch_optimized(texts[half_size:], tokenizer, model)
                return perplexities1 + perplexities2
            else:
                return [calculate_perplexity(text, tokenizer, model) for text in texts]
        else:
            raise e
    except Exception as e:
        logging.warning(f"批量计算失败，回退到逐条计算: {str(e)}")
        return [calculate_perplexity(text, tokenizer, model) for text in texts]

def calculate_perplexity(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            avg_loss = loss.mean()
            return torch.exp(avg_loss).item()
    except:
        return float('inf')

# ========== 阈值计算（恢复困惑度分布图生成） ==========
def analyze_perplexity_distribution_layered(input_file, sample_size=1000):
    logging.info("🔍 开始分层困惑度分布分析（将生成分布图）")
    tokenizer, model = load_perplexity_model()
    
    texts = []
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if len(lines) > sample_size:
            import random
            lines = random.sample(lines, sample_size)
        for line in lines:
            try:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if text and len(text) >= MIN_CHAR_LEN:
                    texts.append(text)
            except:
                continue
    
    if not texts:
        logging.error("❌ 没有有效的文本数据用于困惑度分析")
        return 100.0, 30.0, None, None
    
    logging.info(f"📊 将分析 {len(texts)} 条文本的困惑度分布")
    
    classic_texts = []
    modern_texts = []
    for text in texts:
        if is_classic_chinese(text):
            classic_texts.append(text)
        else:
            modern_texts.append(text)
    
    logging.info(f"📊 文本分类结果: 古文 {len(classic_texts)} 条, 现代文 {len(modern_texts)} 条")
    
    modern_perplexities = []
    for i in tqdm(range(0, len(modern_texts), BATCH_SIZE_PERPLEXITY), desc="计算现代文困惑度"):
        batch = modern_texts[i:i+BATCH_SIZE_PERPLEXITY]
        batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
        modern_perplexities.extend(batch_perplexities)
    
    classic_perplexities = []
    for i in tqdm(range(0, len(classic_texts), BATCH_SIZE_PERPLEXITY), desc="计算古文困惑度"):
        batch = classic_texts[i:i+BATCH_SIZE_PERPLEXITY]
        batch_perplexities = calculate_perplexity_batch_optimized(batch, tokenizer, model)
        classic_perplexities.extend(batch_perplexities)
    
    modern_perplexities = np.array(modern_perplexities)
    classic_perplexities = np.array(classic_perplexities)
    
    # 过滤异常大的困惑度值（避免分布图失真）
    modern_valid = modern_perplexities[modern_perplexities < 15000]
    classic_valid = classic_perplexities[classic_perplexities < 15000]
    
    if len(modern_valid) == 0:
        logging.warning("⚠️ 现代文困惑度计算失败，使用默认值")
        modern_valid = np.array([100.0])
    
    if len(classic_valid) == 0:
        logging.warning("⚠️ 古文困惑度计算失败，使用默认值")
        classic_valid = np.array([30.0])
    
    percentiles = [0, 10, 20, 25, 30, 50, 75, 90, 95, 100]
    modern_percentiles = np.percentile(modern_valid, percentiles)
    classic_percentiles = np.percentile(classic_valid, percentiles)
    
    modern_threshold_idx = 6  # 75%分位数
    classic_threshold_idx = 1  # 10%分位数
    
    modern_threshold = modern_percentiles[modern_threshold_idx]
    classic_threshold = classic_percentiles[classic_threshold_idx]
    
    # 输出阈值日志
    logging.info("📈 现代文困惑度分布:")
    for p, val in zip(percentiles, modern_percentiles):
        logging.info(f"    {p}% 分位数: {val:.2f}")
    logging.info(f"🎯 现代文阈值: ≤{modern_threshold:.2f} ({percentiles[modern_threshold_idx]}%分位数)")
    
    logging.info("📈 古文困惑度分布:")
    for p, val in zip(percentiles, classic_percentiles):
        logging.info(f"    {p}% 分位数: {val:.2f}")
    logging.info(f"🎯 古文阈值: ≥{classic_threshold:.2f} ({percentiles[classic_threshold_idx]}%分位数)")
    
    # 恢复生成困惑度分布图（关键修改）
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']  # 兼容中文/英文
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建2x1子图，分别显示现代文和古文分布
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Perplexity Distribution (Classic vs Modern Chinese)', fontsize=16, fontweight='bold')
            
            # 现代文困惑度直方图（添加阈值线）
            ax1.hist(modern_valid, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax1.axvline(modern_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {modern_threshold:.1f} (75th)')
            ax1.set_title('Modern Chinese Perplexity', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Perplexity Value', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 古文困惑度直方图（添加阈值线）
            ax2.hist(classic_valid, bins=50, color='#A23B72', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.axvline(classic_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {classic_threshold:.1f} (10th)')
            ax2.set_title('Classic Chinese Perplexity', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Perplexity Value', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # 调整布局，避免标签重叠
            plt.tight_layout()
            
            # 保存图片（仅生成1张，无其他冗余）
            plot_path = os.path.join(OUTPUT_PATH, 'perplexity_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logging.info(f"✅ 困惑度分布图已保存：{plot_path}")
            logging.info(f"📊 图片大小：{os.path.getsize(plot_path)/1024/1024:.2f}MB")
        except Exception as e:
            logging.warning(f"⚠️ 生成困惑度分布图失败：{str(e)}")
    else:
        logging.warning("⚠️ 未安装 matplotlib，跳过分布图生成")
    
    return modern_threshold, classic_threshold, modern_percentiles, classic_percentiles

def determine_layered_thresholds(input_file):
    logging.info("🎯 开始确定分层困惑度阈值")
    try:
        modern_threshold, classic_threshold, modern_percentiles, classic_percentiles = analyze_perplexity_distribution_layered(input_file, sample_size=500)
        percentiles = [0, 10, 20, 25, 30, 50, 75, 90, 95, 100]
        modern_p = percentiles[6]
        classic_p = percentiles[1]
        logging.info(f"🤖 现代文最终阈值: ≤{modern_threshold:.2f} ({modern_p}%分位数)")
        logging.info(f"📜 古文最终阈值: ≥{classic_threshold:.2f} ({classic_p}%分位数)")
        return modern_threshold, classic_threshold
    except Exception as e:
        logging.error(f"❌ 确定分层阈值失败: {str(e)}")
        logging.info("🔄 使用默认阈值: 现代文=80 (75%分位数), 古文=30 (10%分位数)")
        return 80.0, 30.0

# ========== 分层困惑度筛选（仅保留核心文件+分布图） ==========
def layered_perplexity_filter(input_file, modern_threshold, classic_threshold):
    start_time = time.time()
    percentiles = [0, 10, 20, 25, 30, 50, 75, 90, 95, 100]
    modern_p = percentiles[6]
    classic_p = percentiles[1]
    
    logging.info(f"🚀 开始分层困惑度筛选（仅保留核心数据文件+分布图）")
    logging.info(f"   - 现代文：≤{modern_threshold:.2f} ({modern_p}%分位数)")
    logging.info(f"   - 古文：≥{classic_threshold:.2f} ({classic_p}%分位数)")
    
    tokenizer, model = load_perplexity_model()
    
    # 只创建核心输出文件（不生成过滤文件、古文单独文件）
    kept_file = os.path.join(OUTPUT_PATH, "clmmu_kept_data_final.jsonl")
    
    batch_texts = []
    batch_data = []
    
    # 只打开核心文件
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(kept_file, "w", encoding="utf-8") as f_kept:
        
        total_input = sum(1 for _ in f_in)  # 统计中间文件总行数
        f_in.seek(0)
        
        for line in tqdm(f_in, desc="分层困惑度筛选", total=total_input):
            try:
                data = json.loads(line)
                batch_texts.append(data["text"])
                batch_data.append(data)
            except:
                continue
            
            if len(batch_texts) >= BATCH_SIZE_PERPLEXITY:
                perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
                
                for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
                    data_item["perplexity"] = round(perp, 2)
                    data_item["is_classic"] = is_classic_chinese(text)
                    
                    # 筛选逻辑不变，仅写入核心文件
                    if data_item["is_classic"]:
                        if perp >= classic_threshold and perp < 10000:
                            final_data = data_item["original_data"].copy()
                            final_data["cleaned_text"] = text
                            final_data["md5"] = data_item["md5"]
                            final_data["perplexity"] = data_item["perplexity"]
                            final_data["source_file"] = data_item["source_file"]
                            final_data["text_type"] = "classic_chinese"  # 保留类型标记
                            final_data["has_academic_features"] = data_item.get("has_academic_features", False)
                            f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                            stats["final_kept"] += 1
                            stats["classic_chinese_kept"] += 1
                        else:
                            stats["perplexity_filtered"] += 1
                    else:
                        if perp <= modern_threshold and perp < 5000:
                            final_data = data_item["original_data"].copy()
                            final_data["cleaned_text"] = text
                            final_data["md5"] = data_item["md5"]
                            final_data["perplexity"] = data_item["perplexity"]
                            final_data["source_file"] = data_item["source_file"]
                            final_data["text_type"] = "modern_chinese"  # 保留类型标记
                            final_data["has_academic_features"] = data_item.get("has_academic_features", False)
                            f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                            stats["final_kept"] += 1
                            stats["modern_chinese_kept"] += 1
                        else:
                            stats["perplexity_filtered"] += 1
                
                batch_texts = []
                batch_data = []
        
        # 处理剩余数据
        if batch_texts:
            perplexities = calculate_perplexity_batch_optimized(batch_texts, tokenizer, model)
            for text, data_item, perp in zip(batch_texts, batch_data, perplexities):
                data_item["perplexity"] = round(perp, 2)
                data_item["is_classic"] = is_classic_chinese(text)
                
                if data_item["is_classic"]:
                    if perp >= classic_threshold and perp < 10000:
                        final_data = data_item["original_data"].copy()
                        final_data["cleaned_text"] = text
                        final_data["md5"] = data_item["md5"]
                        final_data["perplexity"] = data_item["perplexity"]
                        final_data["source_file"] = data_item["source_file"]
                        final_data["text_type"] = "classic_chinese"
                        f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                        stats["final_kept"] += 1
                        stats["classic_chinese_kept"] += 1
                    else:
                        stats["perplexity_filtered"] += 1
                else:
                    if perp <= modern_threshold and perp < 5000:
                        final_data = data_item["original_data"].copy()
                        final_data["cleaned_text"] = text
                        final_data["md5"] = data_item["md5"]
                        final_data["perplexity"] = data_item["perplexity"]
                        final_data["source_file"] = data_item["source_file"]
                        final_data["text_type"] = "modern_chinese"
                        f_kept.write(json.dumps(final_data, ensure_ascii=False) + "\n")
                        stats["final_kept"] += 1
                        stats["modern_chinese_kept"] += 1
                    else:
                        stats["perplexity_filtered"] += 1
    
    stats["stage_time"]["perplexity"] = round(time.time() - start_time, 2)
    logging.info(f"✅ 分层困惑度筛选完成 | 耗时：{stats['stage_time']['perplexity']}秒")
    logging.info(f"📊 分层统计 - 现代文保留: {stats['modern_chinese_kept']} | 古文保留: {stats['classic_chinese_kept']}")
    logging.info(f"📊 总计保留: {stats['final_kept']} | 过滤: {stats['perplexity_filtered']}")
    
    return kept_file  # 只返回核心文件

# ========== 主函数（仅输出核心文件+分布图，关闭日志文件） ==========
def main():
    global monitor_running
    start_time = time.time()
    
    # 日志仅输出到控制台，不写入文件（节省空间）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("🎉 启动CLMMU/CEVAL筛选流程（核心文件+困惑度分布图版）")
    logging.info(f"⚠️  配置：生成 clmmu_kept_data_final.jsonl + perplexity_distribution.png")
    logging.info(f"📋 复用的中间文件：{EXISTING_MINHASH_FILE}")
    logging.info("="*80)
    
    # 验证中间文件
    if not os.path.exists(EXISTING_MINHASH_FILE):
        logging.error(f"❌ 未找到中间文件：{EXISTING_MINHASH_FILE}")
        return
    
    # 启动监控线程
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    
    try:
        # 加载中间文件
        minhash_file = EXISTING_MINHASH_FILE
        total_intermediate = sum(1 for _ in open(minhash_file, 'r', encoding='utf-8'))
        logging.info(f"✅ 加载中间文件：{minhash_file}（总行数：{total_intermediate}条）")
        
        # 计算阈值（生成分布图）
        modern_threshold, classic_threshold = determine_layered_thresholds(minhash_file)
        
        # 筛选（仅生成核心文件）
        kept_file = layered_perplexity_filter(minhash_file, modern_threshold, classic_threshold)
        
        # 停止监控
        monitor_running = False
        monitor.join()
        
        # 最终统计
        total_time = round(time.time() - start_time, 2)
        logging.info("\n" + "="*80)
        logging.info("📊 筛选最终统计报告（核心文件+分布图版）")
        logging.info("="*80)
        logging.info(f"中间文件总行数: {total_intermediate}条")
        logging.info(f"📌 保留统计:")
        logging.info(f"   - 现代文保留: {stats['modern_chinese_kept']}条")
        logging.info(f"   - 古文保留: {stats['classic_chinese_kept']}条")
        logging.info(f"   - 总计保留: {stats['final_kept']}条")
        logging.info(f"   - 保留比例: {stats['final_kept']/total_intermediate*100:.1f}%")
        logging.info(f"📌 阈值配置:")
        logging.info(f"   - 现代文困惑度: ≤{modern_threshold:.2f} (75%分位数)")
        logging.info(f"   - 古文困惑度: ≥{classic_threshold:.2f} (10%分位数)")
        logging.info(f"📌 性能统计:")
        logging.info(f"   - 总耗时: {total_time}秒 ({total_time/60:.1f}分钟)")
        logging.info("\n📁 最终输出文件（仅2个）:")
        logging.info(f"✅ 高质量数据：{kept_file}")
        logging.info(f"📊 文件大小：{os.path.getsize(kept_file)/1024/1024:.2f}MB")
        if MATPLOTLIB_AVAILABLE and os.path.exists(os.path.join(OUTPUT_PATH, 'perplexity_distribution.png')):
            plot_path = os.path.join(OUTPUT_PATH, 'perplexity_distribution.png')
            logging.info(f"✅ 困惑度分布图：{plot_path}")
            logging.info(f"📊 图片大小：{os.path.getsize(plot_path)/1024/1024:.2f}MB")
        logging.info("="*80)
        
        # 提示删除无用文件释放空间
        logging.info("\n🗑️  可删除的无用文件（释放空间）:")
        logging.info(f"   1. 中间文件：{EXISTING_MINHASH_FILE}（筛选完成后可删）")
        logging.info(f"   2. 预处理中间文件：data/intermediate/preprocessed_md5_dedup_with_sensitive_filter.jsonl")
        
    except Exception as e:
        logging.error(f"❌ 流程执行失败: {str(e)}", exc_info=True)
    finally:
        monitor_running = False
        monitor.join(timeout=5)
        logging.info("🔚 筛选流程结束（核心文件+分布图版）")

if __name__ == "__main__":
    main()