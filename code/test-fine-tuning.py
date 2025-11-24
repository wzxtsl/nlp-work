import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel
import json
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import jieba
from rouge_chinese import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMerger:
    def __init__(self, base_model_path, lora_model_path, output_path):
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.output_path = output_path
        
    def merge_and_save(self):
        """合并LoRA权重到基础模型"""
        logger.info("开始合并模型...")
        
        # 检查LoRA模型文件是否存在
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        if not all(os.path.exists(os.path.join(self.lora_model_path, f)) for f in required_files):
            # 尝试其他可能的文件扩展名
            required_files_alt = ["adapter_config.json", "adapter_model.bin"]
            if not all(os.path.exists(os.path.join(self.lora_model_path, f)) for f in required_files_alt):
                raise FileNotFoundError(f"LoRA模型文件缺失，请检查路径: {self.lora_model_path}")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载LoRA权重并合并
        model = PeftModel.from_pretrained(
            base_model,
            self.lora_model_path,
            torch_dtype=torch.float16
        )
        
        # 合并权重
        merged_model = model.merge_and_unload()
        
        # 保存合并后的模型
        merged_model.save_pretrained(self.output_path)
        tokenizer.save_pretrained(self.output_path)
        
        logger.info(f"模型已保存到: {self.output_path}")
        return merged_model, tokenizer

class ChineseEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
    def calculate_perplexity(self, texts, batch_size=4):
        """计算困惑度"""
        logger.info("计算困惑度...")
        perplexities = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            encodings = self.tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings.input_ids)
                loss = outputs.loss
                ppl = torch.exp(loss)
                perplexities.extend([ppl.item()] * len(batch_texts))
        
        return np.mean(perplexities)
    
    def calculate_rouge(self, references, predictions):
        """计算ROUGE分数"""
        rouge = Rouge()
        
        # 确保输入是列表格式
        if isinstance(references, str):
            references = [references]
        if isinstance(predictions, str):
            predictions = [predictions]
            
        # 对中文进行分词
        refs = [' '.join(jieba.cut(ref)) for ref in references]
        preds = [' '.join(jieba.cut(pred)) for pred in predictions]
        
        try:
            scores = rouge.get_scores(preds, refs, avg=True)
            return scores
        except:
            return {"rouge-1": {"f": 0, "p": 0, "r": 0},
                   "rouge-2": {"f": 0, "p": 0, "r": 0},
                   "rouge-l": {"f": 0, "p": 0, "r": 0}}
    
    def calculate_bleu(self, references, predictions):
        """计算BLEU分数"""
        smoothie = SmoothingFunction().method4
        
        # 对中文进行分词
        refs = [[list(jieba.cut(ref))] for ref in references]
        preds = [list(jieba.cut(pred)) for pred in predictions]
        
        bleu_scores = []
        for ref, pred in zip(refs, preds):
            try:
                score = sentence_bleu(ref, pred, smoothing_function=smoothie)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        return np.mean(bleu_scores)
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]

class QAEvaluator:
    """专门评估问答能力的类"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def evaluate_qa_accuracy(self, qa_pairs):
        """评估问答准确率"""
        logger.info("评估问答准确率...")
        
        exact_matches = 0
        partial_matches = 0
        relevance_scores = []
        completeness_scores = []
        
        for i, qa in enumerate(tqdm(qa_pairs)):
            question = qa['question']
            reference_answer = qa['answer']
            
            # 生成回答
            prompt = f"问题：{question}\n回答："
            generated_answer = self.generate_text(prompt, max_length=100, temperature=0.3)
            
            # 精确匹配
            if self._exact_match(generated_answer, reference_answer):
                exact_matches += 1
            
            # 部分匹配（语义相似度）
            similarity = self._semantic_similarity(generated_answer, reference_answer)
            if similarity > 0.7:
                partial_matches += 1
            
            # 相关性评分
            relevance = self._calculate_relevance(generated_answer, question)
            relevance_scores.append(relevance)
            
            # 完整性评分
            completeness = self._calculate_completeness(generated_answer, reference_answer)
            completeness_scores.append(completeness)
        
        total = len(qa_pairs)
        return {
            'exact_match_rate': exact_matches / total,
            'partial_match_rate': partial_matches / total,
            'average_relevance': np.mean(relevance_scores),
            'average_completeness': np.mean(completeness_scores),
            'qa_accuracy_combined': (exact_matches + partial_matches) / (2 * total)
        }
    
    def evaluate_factual_qa(self, factual_questions):
        """评估事实性问答能力"""
        logger.info("评估事实性问答能力...")
        
        correct_factual = 0
        factual_scores = []
        
        for qa in tqdm(factual_questions):
            question = qa['question']
            correct_answer = qa['correct_answer']
            distractors = qa.get('distractors', [])
            
            prompt = f"问题：{question}\n回答："
            generated_answer = self.generate_text(prompt, max_length=100, temperature=0.1)
            
            # 检查事实正确性
            if self._check_factual_correctness(generated_answer, correct_answer, distractors):
                correct_factual += 1
            
            # 事实性评分
            factual_score = self._calculate_factual_score(generated_answer, correct_answer)
            factual_scores.append(factual_score)
        
        total = len(factual_questions)
        return {
            'factual_accuracy': correct_factual / total,
            'average_factual_score': np.mean(factual_scores)
        }
    
    def evaluate_reasoning_qa(self, reasoning_questions):
        """评估推理问答能力"""
        logger.info("评估推理问答能力...")
        
        correct_reasoning = 0
        reasoning_scores = []
        
        for qa in tqdm(reasoning_questions):
            question = qa['question']
            expected_reasoning = qa.get('expected_reasoning', '')
            correct_answer = qa['correct_answer']
            
            prompt = f"问题：{question}\n请推理并回答："
            generated_answer = self.generate_text(prompt, max_length=150, temperature=0.3)
            
            # 检查推理正确性
            if self._check_reasoning_correctness(generated_answer, correct_answer, expected_reasoning):
                correct_reasoning += 1
            
            # 推理质量评分
            reasoning_score = self._calculate_reasoning_quality(generated_answer, expected_reasoning)
            reasoning_scores.append(reasoning_score)
        
        total = len(reasoning_questions)
        return {
            'reasoning_accuracy': correct_reasoning / total,
            'average_reasoning_score': np.mean(reasoning_scores)
        }
    
    def evaluate_multi_turn_qa(self, multi_turn_conversations):
        """评估多轮问答能力"""
        logger.info("评估多轮问答能力...")
        
        coherence_scores = []
        context_awareness_scores = []
        
        for conversation in tqdm(multi_turn_conversations):
            turns = conversation['turns']
            context = []
            
            for turn in turns:
                question = turn['question']
                expected_answer = turn.get('expected_answer', '')
                
                # 构建包含上下文的提示
                context_str = "\n".join(context[-3:])  # 使用最近3轮作为上下文
                if context_str:
                    prompt = f"对话历史：\n{context_str}\n当前问题：{question}\n回答："
                else:
                    prompt = f"问题：{question}\n回答："
                
                generated_answer = self.generate_text(prompt, max_length=100, temperature=0.5)
                
                # 更新上下文
                context.append(f"问：{question}")
                context.append(f"答：{generated_answer}")
            
            # 评估对话连贯性
            coherence = self._evaluate_conversation_coherence(context)
            coherence_scores.append(coherence)
            
            # 评估上下文感知
            context_awareness = self._evaluate_context_awareness(context, turns)
            context_awareness_scores.append(context_awareness)
        
        return {
            'average_coherence': np.mean(coherence_scores),
            'average_context_awareness': np.mean(context_awareness_scores)
        }
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
    
    def _exact_match(self, generated, reference):
        """精确匹配检查"""
        return generated.strip() == reference.strip()
    
    def _semantic_similarity(self, text1, text2):
        """计算语义相似度"""
        # 使用TF-IDF计算余弦相似度
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]
    
    def _calculate_relevance(self, answer, question):
        """计算回答相关性"""
        question_words = set(jieba.cut(question))
        answer_words = set(jieba.cut(answer))
        
        if not question_words:
            return 0
        
        overlap = len(question_words & answer_words) / len(question_words)
        return min(1.0, overlap * 2)  # 放大效果
    
    def _calculate_completeness(self, generated, reference):
        """计算回答完整性"""
        ref_words = set(jieba.cut(reference))
        gen_words = set(jieba.cut(generated))
        
        if not ref_words:
            return 0
        
        coverage = len(ref_words & gen_words) / len(ref_words)
        return min(1.0, coverage)
    
    def _check_factual_correctness(self, generated, correct, distractors=None):
        """检查事实正确性"""
        # 检查是否包含正确答案
        if correct.lower() in generated.lower():
            return True
        
        # 检查是否包含干扰项
        if distractors:
            for distractor in distractors:
                if distractor.lower() in generated.lower():
                    return False
        
        return False
    
    def _calculate_factual_score(self, generated, correct):
        """计算事实性评分"""
        correct_lower = correct.lower()
        generated_lower = generated.lower()
        
        if correct_lower in generated_lower:
            return 1.0
        else:
            # 使用编辑距离作为备选
            from nltk.metrics import edit_distance
            distance = edit_distance(correct_lower, generated_lower[:len(correct_lower)])
            max_len = max(len(correct_lower), len(generated_lower[:len(correct_lower)]))
            return 1.0 - (distance / max_len)
    
    def _check_reasoning_correctness(self, generated, correct_answer, expected_reasoning):
        """检查推理正确性"""
        # 检查是否包含正确答案
        if correct_answer.lower() in generated.lower():
            return True
        
        # 检查是否包含关键推理词汇
        reasoning_keywords = ['因为', '所以', '因此', '由于', '原因', '结果']
        if any(keyword in generated for keyword in reasoning_keywords):
            return 0.5  # 部分正确
        
        return False
    
    def _calculate_reasoning_quality(self, generated, expected_reasoning):
        """计算推理质量评分"""
        if not expected_reasoning:
            return 0.5  # 如果没有预期推理，给中等分数
        
        return self._semantic_similarity(generated, expected_reasoning)
    
    def _evaluate_conversation_coherence(self, context):
        """评估对话连贯性"""
        if len(context) < 4:
            return 0.5
        
        # 分析对话主题一致性
        topics = []
        for i, turn in enumerate(context):
            if i % 2 == 0:  # 问题
                words = jieba.cut(turn)
                topics.append(set(words))
        
        # 计算主题一致性
        consistency_scores = []
        for i in range(len(topics)-1):
            if topics[i] and topics[i+1]:
                overlap = len(topics[i] & topics[i+1]) / len(topics[i] | topics[i+1])
                consistency_scores.append(overlap)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _evaluate_context_awareness(self, context, turns):
        """评估上下文感知能力"""
        if len(context) < 4:
            return 0.5
        
        # 检查是否在后续回答中引用了之前的上下文
        reference_count = 0
        for i in range(2, len(context), 2):  # 从第三个回答开始检查
            current_answer = context[i]
            previous_context = context[:i-1]
            
            # 检查是否引用了之前的对话
            for prev_turn in previous_context[-4:]:  # 检查最近4轮
                prev_words = set(jieba.cut(prev_turn))
                current_words = set(jieba.cut(current_answer))
                if len(prev_words & current_words) > 1:  # 至少2个词重叠
                    reference_count += 1
                    break
        
        max_references = len(context) // 2 - 1
        return reference_count / max_references if max_references > 0 else 0.5

class CMMLUEvaluator:
    def __init__(self, model, tokenizer, data_path="./cmmlu/data"):
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.device = model.device
        
    def get_all_subjects(self):
        """获取CMMLU所有科目列表"""
        test_dir = os.path.join(self.data_path, "test")
        if not os.path.exists(test_dir):
            logger.error(f"测试目录不存在: {test_dir}")
            return []
        
        csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
        subjects = [f.replace('.csv', '') for f in csv_files]
        logger.info(f"找到 {len(subjects)} 个科目")
        return sorted(subjects)
    
    def load_cmmlu_data(self, subject, split="test"):
        """加载CMMLU数据集"""
        file_path = os.path.join(self.data_path, split, f"{subject}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
            
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.warning(f"加载CSV失败 {file_path}: {e}")
            return None
    
    def evaluate_subject(self, subject, num_samples=30):
        """评估单个科目"""
        df = self.load_cmmlu_data(subject)
        if df is None or len(df) == 0:
            logger.warning(f"没有找到 {subject} 的数据")
            return None
            
        # 检查必要的列
        required_columns = ['Question', 'A', 'B', 'C', 'D', 'Answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"{subject} 缺少列: {missing_columns}")
            return None
        
        # 随机采样
        if len(df) > num_samples:
            df = df.sample(n=num_samples, random_state=42)
        else:
            logger.info(f"科目 {subject} 只有 {len(df)} 个样本，使用全部样本")
        
        correct = 0
        total = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total, desc=f"评估 {subject}"):
            question = row['Question']
            options = [row['A'], row['B'], row['C'], row['D']]
            answer = row['Answer']
            
            # 构建提示
            prompt = f"问题：{question}\n选项：\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n答案："
            
            # 生成回答
            try:
                generated = self.generate_text(prompt, max_length=10, temperature=0.1)
                
                # 检查答案 - 更灵活的匹配
                generated_upper = generated.upper().strip()
                if answer in generated_upper:
                    correct += 1
                elif len(generated_upper) > 0 and generated_upper[0] == answer:
                    correct += 1
                # 记录部分错误答案用于调试
                elif idx < 3:  # 只记录前3个错误
                    logger.debug(f"错误示例 - 问题: {question[:50]}...")
                    logger.debug(f"生成: {generated_upper}, 正确答案: {answer}")
            except Exception as e:
                logger.warning(f"生成失败: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"{subject}: {correct}/{total} = {accuracy:.4f}")
        return accuracy
    
    def generate_text(self, prompt, max_length=100, temperature=0.1):
        """生成文本（简化版）"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
    
    def comprehensive_evaluation(self, num_samples=30):
        """综合评估所有科目"""
        subjects = self.get_all_subjects()
        if not subjects:
            logger.error("未找到任何科目数据")
            return {}, 0
        
        results = {}
        valid_subjects = 0
        
        logger.info(f"开始评估 {len(subjects)} 个科目，每个科目 {num_samples} 个样本")
        
        for subject in subjects:
            accuracy = self.evaluate_subject(subject, num_samples)
            if accuracy is not None:
                results[subject] = accuracy
                valid_subjects += 1
        
        avg_accuracy = np.mean(list(results.values())) if results else 0
        logger.info(f"有效评估科目: {valid_subjects}/{len(subjects)}")
        logger.info(f"CMMLU平均准确率: {avg_accuracy:.4f}")
        
        # 按准确率排序
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        return sorted_results, avg_accuracy

def create_qa_test_dataset():
    """创建问答测试数据集"""
    
    # 基础问答对（扩充到30个）
    basic_qa_pairs = [
        {
            "question": "中国的首都是哪里？",
            "answer": "北京"
        },
        {
            "question": "太阳系有多少颗行星？", 
            "answer": "8"
        },
        {
            "question": "水的化学式是什么？",
            "answer": "H2O"
        },
        {
            "question": "谁写了《红楼梦》？",
            "answer": "曹雪芹"
        },
        {
            "question": "Python是什么编程语言？",
            "answer": "高级编程语言"
        },
        {
            "question": "珠穆朗玛峰的高度是多少？",
            "answer": "8848米"
        },
        {
            "question": "一年有多少个月份？",
            "answer": "12个"
        },
        {
            "question": "地球围绕什么天体旋转？",
            "answer": "太阳"
        },
        {
            "question": "光速是多少？",
            "answer": "299792458米/秒"
        },
        {
            "question": "《论语》的作者是谁？",
            "answer": "孔子及其弟子"
        },
        {
            "question": "计算机的基本组成部分有哪些？",
            "answer": "运算器、控制器、存储器、输入设备、输出设备"
        },
        {
            "question": "第一次世界大战发生在什么时候？",
            "answer": "1914-1918年"
        },
        {
            "question": "DNA的全称是什么？",
            "answer": "脱氧核糖核酸"
        },
        {
            "question": "钢琴有多少个琴键？",
            "answer": "88个"
        },
        {
            "question": "三角形的内角和是多少度？",
            "answer": "180度"
        },
        {
            "question": "法国的首都是哪里？",
            "answer": "巴黎"
        },
        {
            "question": "人类有多少对染色体？",
            "answer": "23对"
        },
        {
            "question": "《西游记》的作者是谁？",
            "answer": "吴承恩"
        },
        {
            "question": "电子邮件的英文缩写是什么？",
            "answer": "E-mail"
        },
        {
            "question": "牛顿三大定律是什么？",
            "answer": "惯性定律、加速度定律、作用力与反作用力定律"
        },
        {
            "question": "中国的人口大约是多少？",
            "answer": "14亿"
        },
        {
            "question": "月亮围绕什么天体旋转？",
            "answer": "地球"
        },
        {
            "question": "化学元素周期表有多少个元素？",
            "answer": "118个"
        },
        {
            "question": "《哈姆雷特》的作者是谁？",
            "answer": "莎士比亚"
        },
        {
            "question": "人工智能的英文缩写是什么？",
            "answer": "AI"
        },
        {
            "question": "第二次世界大战结束于哪一年？",
            "answer": "1945年"
        },
        {
            "question": "光合作用的产物是什么？",
            "answer": "氧气和葡萄糖"
        },
        {
            "question": "中国的国花是什么？",
            "answer": "牡丹"
        },
        {
            "question": "计算机网络中，IP地址的作用是什么？",
            "answer": "标识网络中的设备"
        },
        {
            "question": "什么是区块链技术？",
            "answer": "一种分布式账本技术，具有去中心化、不可篡改等特点"
        }
    ]
    
    # 事实性问答（扩充到20个）
    factual_questions = [
        {
            "question": "珠穆朗玛峰位于哪个山脉？",
            "correct_answer": "喜马拉雅山脉",
            "distractors": ["阿尔卑斯山脉", "安第斯山脉", "落基山脉"]
        },
        {
            "question": "第二次世界大战中，轴心国包括哪些国家？",
            "correct_answer": "德国、意大利、日本",
            "distractors": ["美国、英国、法国", "苏联、中国、波兰", "加拿大、澳大利亚、新西兰"]
        },
        {
            "question": "光在真空中的传播速度是多少？",
            "correct_answer": "299792458米/秒",
            "distractors": ["300000000米/秒", "299000000米/秒", "300792458米/秒"]
        },
        {
            "question": "《物种起源》的作者是谁？",
            "correct_answer": "达尔文",
            "distractors": ["牛顿", "爱因斯坦", "伽利略"]
        },
        {
            "question": "元素周期表中，原子序数为1的元素是什么？",
            "correct_answer": "氢",
            "distractors": ["氦", "锂", "铍"]
        },
        {
            "question": "中国古代四大发明不包括下列哪项？",
            "correct_answer": "蒸汽机",
            "distractors": ["造纸术", "火药", "指南针"]
        },
        {
            "question": "美国独立宣言签署于哪一年？",
            "correct_answer": "1776年",
            "distractors": ["1789年", "1775年", "1783年"]
        },
        {
            "question": "DNA分子的结构是什么形状？",
            "correct_answer": "双螺旋结构",
            "distractors": ["单螺旋结构", "三螺旋结构", "直线结构"]
        },
        {
            "question": "《蒙娜丽莎》是哪位画家的作品？",
            "correct_answer": "达·芬奇",
            "distractors": ["米开朗基罗", "拉斐尔", "梵高"]
        },
        {
            "question": "计算机操作系统中，Windows是哪个公司开发的？",
            "correct_answer": "微软公司",
            "distractors": ["苹果公司", "谷歌公司", "IBM公司"]
        },
        {
            "question": "太阳系中，体积最大的行星是什么？",
            "correct_answer": "木星",
            "distractors": ["土星", "天王星", "海王星"]
        },
        {
            "question": "中国历史上第一个封建王朝是什么？",
            "correct_answer": "秦朝",
            "distractors": ["夏朝", "商朝", "周朝"]
        },
        {
            "question": "相对论的提出者是谁？",
            "correct_answer": "爱因斯坦",
            "distractors": ["牛顿", "麦克斯韦", "玻尔"]
        },
        {
            "question": "世界上最长的河流是什么？",
            "correct_answer": "尼罗河",
            "distractors": ["亚马逊河", "长江", "黄河"]
        },
        {
            "question": "元素周期表中，符号为O的元素是什么？",
            "correct_answer": "氧",
            "distractors": ["氮", "碳", "氢"]
        },
        {
            "question": "《资本论》的作者是谁？",
            "correct_answer": "马克思",
            "distractors": ["恩格斯", "列宁", "斯大林"]
        },
        {
            "question": "互联网起源于哪个国家？",
            "correct_answer": "美国",
            "distractors": ["英国", "中国", "德国"]
        },
        {
            "question": "人体最主要的消化器官是什么？",
            "correct_answer": "小肠",
            "distractors": ["胃", "大肠", "肝脏"]
        },
        {
            "question": "法国大革命开始于哪一年？",
            "correct_answer": "1789年",
            "distractors": ["1776年", "1799年", "1804年"]
        },
        {
            "question": "集成电路的发明者是谁？",
            "correct_answer": "杰克·基尔比和罗伯特·诺伊斯",
            "distractors": ["爱迪生", "特斯拉", "贝尔"]
        }
    ]
    
    # 推理问答（扩充到15个）
    reasoning_questions = [
        {
            "question": "如果所有猫都会爬树，汤姆是一只猫，那么汤姆会爬树吗？",
            "correct_answer": "会",
            "expected_reasoning": "因为汤姆是猫，而所有猫都会爬树"
        },
        {
            "question": "如果明天下雨，比赛会取消。今天下雨了，比赛会取消吗？",
            "correct_answer": "不会",
            "expected_reasoning": "条件是说如果明天下雨比赛取消，但今天下雨不影响"
        },
        {
            "question": "一个篮子里有5个苹果，你拿走2个，还剩几个？",
            "correct_answer": "3个",
            "expected_reasoning": "5减去2等于3"
        },
        {
            "question": "A比B高，B比C高，那么谁最高？",
            "correct_answer": "A",
            "expected_reasoning": "根据传递性，A比B高，B比C高，所以A最高"
        },
        {
            "question": "如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？",
            "correct_answer": "不会",
            "expected_reasoning": "企鹅是鸟，但它是特例，不会飞"
        },
        {
            "question": "一个时钟敲3下需要2秒，敲6下需要几秒？",
            "correct_answer": "5秒",
            "expected_reasoning": "敲3下有2个间隔，每个间隔1秒；敲6下有5个间隔，需要5秒"
        },
        {
            "question": "如果x + 5 = 12，那么x的值是多少？",
            "correct_answer": "7",
            "expected_reasoning": "两边同时减去5，得到x = 12 - 5 = 7"
        },
        {
            "question": "所有的金属都能导电，铜是金属，所以铜能导电吗？",
            "correct_answer": "能",
            "expected_reasoning": "铜是金属，而所有金属都能导电，所以铜能导电"
        },
        {
            "question": "如果今天是星期一，那么7天后是星期几？",
            "correct_answer": "星期一",
            "expected_reasoning": "一周有7天，7天后是同一星期几"
        },
        {
            "question": "一个长方形的长是5米，宽是3米，它的面积是多少？",
            "correct_answer": "15平方米",
            "expected_reasoning": "长方形面积 = 长 × 宽 = 5 × 3 = 15"
        },
        {
            "question": "如果A是B的父亲，B是C的父亲，那么A是C的什么人？",
            "correct_answer": "祖父",
            "expected_reasoning": "A是B的父亲，B是C的父亲，所以A是C的祖父"
        },
        {
            "question": "如果温度升高，水会沸腾吗？",
            "correct_answer": "不一定",
            "expected_reasoning": "水沸腾需要达到沸点，标准大气压下是100°C，只说温度升高不一定达到沸点"
        },
        {
            "question": "一个袋子里有3个红球和2个蓝球，随机取出一个，是红球的概率是多少？",
            "correct_answer": "3/5",
            "expected_reasoning": "总共有5个球，红球有3个，所以概率是3/5"
        },
        {
            "question": "如果所有的哺乳动物都是胎生的，鸭嘴兽是哺乳动物，那么鸭嘴兽是胎生的吗？",
            "correct_answer": "不是",
            "expected_reasoning": "鸭嘴兽是哺乳动物，但它是卵生的，是特例"
        },
        {
            "question": "如果火车每小时行驶60公里，那么行驶180公里需要多少时间？",
            "correct_answer": "3小时",
            "expected_reasoning": "时间 = 距离 ÷ 速度 = 180 ÷ 60 = 3小时"
        }
    ]
    
    # 多轮对话（扩充到8个）
    multi_turn_conversations = [
        {
            "turns": [
                {"question": "你好，你叫什么名字？"},
                {"question": "你能做什么？"},
                {"question": "告诉我关于人工智能的一些知识"},
                {"question": "人工智能有哪些应用领域？"},
                {"question": "未来人工智能会取代人类吗？"}
            ]
        },
        {
            "turns": [
                {"question": "今天天气怎么样？"},
                {"question": "适合出去散步吗？"},
                {"question": "需要带伞吗？"},
                {"question": "明天天气会变好吗？"},
                {"question": "周末天气如何？"}
            ]
        },
        {
            "turns": [
                {"question": "我想学习编程，应该从什么语言开始？"},
                {"question": "Python适合初学者吗？"},
                {"question": "学习Python需要什么基础？"},
                {"question": "有什么好的Python学习资源推荐？"},
                {"question": "学会Python后可以做什么项目？"}
            ]
        },
        {
            "turns": [
                {"question": "什么是区块链？"},
                {"question": "区块链有什么特点？"},
                {"question": "区块链和比特币有什么关系？"},
                {"question": "区块链有哪些应用场景？"},
                {"question": "区块链技术安全吗？"}
            ]
        },
        {
            "turns": [
                {"question": "我想减肥，有什么建议吗？"},
                {"question": "如何健康地减肥？"},
                {"question": "减肥期间应该吃什么？"},
                {"question": "运动减肥的最佳时间是什么时候？"},
                {"question": "如何保持减肥成果？"}
            ]
        },
        {
            "turns": [
                {"question": "什么是机器学习？"},
                {"question": "机器学习和人工智能有什么区别？"},
                {"question": "机器学习有哪些算法？"},
                {"question": "学习机器学习需要什么数学基础？"},
                {"question": "机器学习有哪些实际应用？"}
            ]
        },
        {
            "turns": [
                {"question": "推荐一本书给我吧？"},
                {"question": "你喜欢什么类型的书？"},
                {"question": "最近有什么畅销书？"},
                {"question": "科幻小说有什么推荐？"},
                {"question": "如何培养阅读习惯？"}
            ]
        },
        {
            "turns": [
                {"question": "什么是元宇宙？"},
                {"question": "元宇宙和虚拟现实有什么区别？"},
                {"question": "元宇宙有什么应用场景？"},
                {"question": "元宇宙的发展前景如何？"},
                {"question": "进入元宇宙需要什么设备？"}
            ]
        }
    ]
    
    return {
        "basic_qa": basic_qa_pairs,
        "factual_qa": factual_questions,
        "reasoning_qa": reasoning_questions,
        "multi_turn_qa": multi_turn_conversations
    }

def test_generation_capability(evaluator, test_cases=None):
    """测试模型生成能力"""
    if test_cases is None:
        test_cases = [
            {
                "prompt": "今天天气真好，我打算",
                "description": "日常对话续写"
            },
            {
                "prompt": "人工智能技术最近的发展方向包括",
                "description": "技术领域知识"
            },
            {
                "prompt": "根据物理学知识，相对论的核心观点是",
                "description": "科学知识理解"
            },
            {
                "prompt": "中国的传统文化博大精深，其中最具代表性的是",
                "description": "文化知识"
            },
            {
                "prompt": "在机器学习中，过拟合是指",
                "description": "专业术语解释"
            },
            {
                "prompt": "写一首关于春天的诗：",
                "description": "诗歌创作"
            },
            {
                "prompt": "如何学习编程？我的建议是",
                "description": "建议生成"
            }
        ]
    
    results = {}
    logger.info("开始生成能力测试...")
    
    for i, case in enumerate(test_cases, 1):
        prompt = case["prompt"]
        description = case["description"]
        
        try:
            generated = evaluator.generate_text(prompt, max_length=100, temperature=0.7)
            results[prompt] = {
                "description": description,
                "generated": generated
            }
            
            print(f"\n【测试 {i} - {description}】")
            print(f"输入: {prompt}")
            print(f"输出: {generated}")
            print("-" * 80)
            
        except Exception as e:
            logger.error(f"生成测试失败: {e}")
            results[prompt] = {
                "description": description,
                "error": str(e)
            }
    
    return results

def main():
    # 配置路径 - 根据你的实际情况修改
    config = {
        "base_model_path": "./Qwen2.5-0.5B",
        "lora_model_path": "/root/shared-nvme/kmr/NLP/qwen2.5-0.5b-io-lora-r16", 
        "merged_model_path": "/root/shared-nvme/kmr/NLP/qwen2.5-0.5b-io-r16",
        "cmmlu_data_path": "./cmmlu/data",
        "output_dir": "/root/shared-nvme/kmr/NLP/qwen2.5-0.5b-io-evaluation-r16"
    }
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["merged_model_path"], exist_ok=True)
    
    # 1. 合并模型
    try:
        logger.info("=" * 60)
        logger.info("步骤1: 合并LoRA模型")
        logger.info("=" * 60)
        
        merger = ModelMerger(
            config["base_model_path"],
            config["lora_model_path"],
            config["merged_model_path"]
        )
        merged_model, tokenizer = merger.merge_and_save()
    except Exception as e:
        logger.error(f"模型合并失败: {e}")
        return
    
    # 2. 初始化评估器
    evaluator = ChineseEvaluator(merged_model, tokenizer)
    qa_evaluator = QAEvaluator(merged_model, tokenizer)
    cmmlu_evaluator = CMMLUEvaluator(merged_model, tokenizer, config["cmmlu_data_path"])
    
    # 3. 基础生成能力测试
    logger.info("=" * 60)
    logger.info("步骤2: 基础生成能力测试")
    logger.info("=" * 60)
    generation_results = test_generation_capability(evaluator)
    
    # 4. 问答能力全面评估
    logger.info("=" * 60)
    logger.info("步骤3: 问答能力全面评估")
    logger.info("=" * 60)
    
    # 创建测试数据集
    qa_test_dataset = create_qa_test_dataset()
    
    # 执行各项问答评估
    qa_accuracy_results = qa_evaluator.evaluate_qa_accuracy(qa_test_dataset["basic_qa"])
    factual_qa_results = qa_evaluator.evaluate_factual_qa(qa_test_dataset["factual_qa"])
    reasoning_qa_results = qa_evaluator.evaluate_reasoning_qa(qa_test_dataset["reasoning_qa"])
    multi_turn_qa_results = qa_evaluator.evaluate_multi_turn_qa(qa_test_dataset["multi_turn_qa"])
    
    # 合并问答评估结果
    qa_results = {
        "basic_qa": qa_accuracy_results,
        "factual_qa": factual_qa_results,
        "reasoning_qa": reasoning_qa_results,
        "multi_turn_qa": multi_turn_qa_results
    }
    
    # 5. 计算困惑度（使用CMMLU数据）
    logger.info("=" * 60)
    logger.info("步骤4: 计算困惑度")
    logger.info("=" * 60)
    sample_texts = []
    
    # 从几个科目中抽取问题作为样本
    sample_subjects = ["chinese_history", "chinese_literature", "computer_science", "philosophy", "economics"]
    for subject in sample_subjects:
        df = cmmlu_evaluator.load_cmmlu_data(subject)
        if df is not None and 'Question' in df.columns:
            sample_texts.extend(df['Question'].head(10).tolist())
    
    if sample_texts:
        try:
            perplexity = evaluator.calculate_perplexity(sample_texts)
            logger.info(f"平均困惑度: {perplexity:.4f}")
        except Exception as e:
            perplexity = None
            logger.error(f"困惑度计算失败: {e}")
    else:
        perplexity = None
        logger.warning("无法计算困惑度：没有找到样本数据")
    
    # 6. CMMLU综合评估 - 测试所有科目
    logger.info("=" * 60)
    logger.info("步骤5: CMMLU全面评估")
    logger.info("=" * 60)
    cmmlu_results, avg_cmmlu = cmmlu_evaluator.comprehensive_evaluation(num_samples=10)
    
    # 7. ROUGE和BLEU评估
    logger.info("=" * 60)
    logger.info("步骤6: 文本质量指标评估")
    logger.info("=" * 60)
    
    # 使用一些参考文本进行评估
    reference_texts = [
        "人工智能是计算机科学的一个分支，旨在创造能够执行智能任务的机器。",
        "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。",
        "深度学习使用神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个重要应用领域。",
        "计算机视觉让机器能够理解和解释视觉信息。"
    ]
    
    generated_texts = []
    for ref in reference_texts:
        # 使用前半部分作为提示
        prompt = ref[:len(ref)//2]
        try:
            generated = evaluator.generate_text(prompt, max_length=50, temperature=0.7)
            generated_texts.append(generated)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            generated_texts.append("")
    
    # 计算指标
    rouge_scores = evaluator.calculate_rouge(reference_texts, generated_texts)
    bleu_score = evaluator.calculate_bleu(reference_texts, generated_texts)
    
    # 8. 保存评估结果
    results = {
        "model_info": {
            "model_name": "Qwen2.5-0.5B 1GB微调版",
            "base_model": config["base_model_path"],
            "lora_model": config["lora_model_path"],
            "merged_model": config["merged_model_path"],
            "evaluation_date": pd.Timestamp.now().isoformat()
        },
        "perplexity": perplexity,
        "cmmlu_results": cmmlu_results,
        "cmmlu_average": avg_cmmlu,
        "rouge_scores": rouge_scores,
        "bleu_score": bleu_score,
        "qa_evaluation": qa_results,
        "generation_examples": generation_results,
        "test_config": {
            "cmmlu_samples_per_subject": 30,
            "cmmlu_total_subjects": len(cmmlu_results),
            "generation_temperature": 0.7,
        }
    }
    
    # 保存结果到文件
    results_file = os.path.join(config["output_dir"], "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 9. 打印摘要报告
    logger.info("=" * 60)
    logger.info("1GB微调模型评估结果摘要:")
    logger.info("=" * 60)
    
    logger.info(f"模型: Qwen2.5-0.5B")
    logger.info(f"合并模型保存到: {config['merged_model_path']}")
    logger.info(f"评估结果保存到: {results_file}")
    
    if perplexity:
        logger.info(f"平均困惑度: {perplexity:.4f}")
    else:
        logger.info("平均困惑度: 计算失败")
    
    logger.info(f"CMMLU平均准确率: {avg_cmmlu:.4f}")
    logger.info(f"评估科目数量: {len(cmmlu_results)}")
    
    # 问答能力摘要
    logger.info("\n问答能力评估:")
    logger.info(f"  - 基础问答准确率: {qa_results['basic_qa']['exact_match_rate']:.4f}")
    logger.info(f"  - 事实性问答准确率: {qa_results['factual_qa']['factual_accuracy']:.4f}")
    logger.info(f"  - 推理问答准确率: {qa_results['reasoning_qa']['reasoning_accuracy']:.4f}")
    logger.info(f"  - 多轮对话连贯性: {qa_results['multi_turn_qa']['average_coherence']:.4f}")
    logger.info(f"  - 回答相关性: {qa_results['basic_qa']['average_relevance']:.4f}")
    logger.info(f"  - 回答完整性: {qa_results['basic_qa']['average_completeness']:.4f}")
    
    # 显示表现最好和最差的5个科目
    if cmmlu_results:
        top_5 = list(cmmlu_results.items())[:5]
        bottom_5 = list(cmmlu_results.items())[-5:]
        
        logger.info("CMMLU表现最佳的5个科目:")
        for subject, accuracy in top_5:
            logger.info(f"   - {subject}: {accuracy:.4f}")
        
        logger.info("CMMLU表现最差的5个科目:")
        for subject, accuracy in bottom_5:
            logger.info(f"   - {subject}: {accuracy:.4f}")
    
    logger.info(f"BLEU分数: {bleu_score:.4f}")
    
    if rouge_scores and 'rouge-1' in rouge_scores:
        logger.info(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
        logger.info(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
        logger.info(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    
    # 生成示例预览
    logger.info("\n生成示例预览:")
    for i, (prompt, result) in enumerate(list(generation_results.items())[:3], 1):
        if 'generated' in result:
            logger.info(f"示例 {i}:")
            logger.info(f"   输入: {prompt}")
            logger.info(f"   输出: {result['generated'][:100]}...")
    
    logger.info("=" * 60)
    logger.info("评估完成！")

if __name__ == "__main__":
    main()