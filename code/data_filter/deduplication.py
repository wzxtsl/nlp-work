# import os
# import json
# import hashlib
# import time
# import re  
# import logging
# from tqdm import tqdm
# from datasketch import MinHash, MinHashLSH
# from config import (
#     INTERMEDIATE_PATH, BATCH_SIZE_PREPROCESS, MINHASH_NUM_PERM, LSH_THRESHOLD,
#     MIN_CHAR_LEN, MAX_CHAR_LEN, ACADEMIC_REQUIRE
# )
# from utils import stats, load_jsonl_files_with_sampling, is_colloquial, is_sensitive, has_academic_features

# def preprocess_and_md5_deduplicate():
#     """预处理 + MD5精确去重 + 基础筛选（长度、敏感、口语、学术特征）"""
#     start_time = time.time()
#     logging.info("🚀 开始预处理 + MD5精确去重 + 基础筛选（含敏感话题过滤）")
    
#     md5_set = set()
#     output_file = os.path.join(INTERMEDIATE_PATH, "preprocessed_md5_dedup_with_sensitive_filter.jsonl")
#     batch_buffer = []
    
#     with open(output_file, "w", encoding="utf-8") as f_out:
#         for item in tqdm(load_jsonl_files_with_sampling(), desc="预处理+筛选", total=stats["sampled_count"] if stats["sampled_count"] > 0 else None):
#             text = item["text"]
#             original_data = item["original_data"]
#             source_file = item["source_file"]
            
#             # 1. 基础长度过滤
#             if len(text) < MIN_CHAR_LEN or len(text) > MAX_CHAR_LEN:
#                 stats["preprocess_filtered"] += 1
#                 continue
            
#             # 2. 文本清洗（去除零宽空格和多余空格）
#             text = re.sub(r"[\u200b\s]+", " ", text).strip()
#             if not text:
#                 stats["preprocess_filtered"] += 1
#                 continue
            
#             # 3. 敏感话题过滤（优先过滤违规数据）
#             if is_sensitive(text):
#                 stats["sensitive_filtered"] += 1
#                 continue
            
#             # 4. 口语化筛选
#             if is_colloquial(text):
#                 stats["colloquial_filtered"] += 1
#                 continue
            
#             # 5. 学术特征筛选（可选）
#             if ACADEMIC_REQUIRE and not has_academic_features(text):
#                 stats["non_academic_filtered"] += 1
#                 continue
            
#             # 6. MD5精确去重
#             md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
#             if md5_hash in md5_set:
#                 stats["md5_duplicated"] += 1
#                 continue
#             md5_set.add(md5_hash)
            
#             # 批量缓存写入
#             batch_buffer.append({
#                 "text": text,
#                 "original_data": original_data,
#                 "source_file": source_file,
#                 "md5": md5_hash,
#                 "has_academic_features": has_academic_features(text)
#             })
            
#             if len(batch_buffer) >= BATCH_SIZE_PREPROCESS:
#                 for data in batch_buffer:
#                     f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
#                 batch_buffer = []
        
#         # 写入剩余数据
#         if batch_buffer:
#             for data in batch_buffer:
#                 f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    
#     # 统计结果
#     stats["stage_time"]["preprocess_md5"] = round(time.time() - start_time, 2)
#     remaining = stats["sampled_count"] - stats["preprocess_filtered"] - \
#                 stats["colloquial_filtered"] - stats["non_academic_filtered"] - \
#                 stats["md5_duplicated"] - stats["sensitive_filtered"]
    
#     logging.info(
#         f"✅ 预处理+筛选完成 | 耗时：{stats['stage_time']['preprocess_md5']}秒 | "
#         f"抽样后：{stats['sampled_count']} | 长度过滤：{stats['preprocess_filtered']} | "
#         f"敏感话题过滤：{stats['sensitive_filtered']} | 口语化：{stats['colloquial_filtered']} | "
#         f"无学术特征：{stats['non_academic_filtered']} | 完全重复（MD5）：{stats['md5_duplicated']} | 剩余：{remaining}"
#     )
#     return output_file

# def create_minhash_signature(text):
#     """生成文本的MinHash签名（2-gram捕捉中文语义）"""
#     minhash = MinHash(num_perm=MINHASH_NUM_PERM)
#     if len(text) < 2:
#         grams = [text] if text else []
#     else:
#         grams = [text[i:i+2] for i in range(len(text)-1)]
#     for gram in grams:
#         token_hash = hashlib.sha256(gram.encode('utf-8')).hexdigest()
#         minhash.update(token_hash.encode('utf-8'))
#     return minhash

# def minhash_lsh_deduplicate(input_file):
#     """Minhash LSH语义去重（去除相似文本）"""
#     start_time = time.time()
#     logging.info(f"🚀 开始Minhash LSH语义去重（阈值{ LSH_THRESHOLD }）")
    
#     # 读取预处理后的数据
#     texts = []
#     data_list = []
#     with open(input_file, "r", encoding="utf-8") as f:
#         for line in tqdm(f, desc="读取预处理数据"):
#             data = json.loads(line)
#             texts.append(data["text"])
#             data_list.append(data)
    
#     if not texts:
#         logging.warning("⚠️ MD5去重后无有效数据，跳过语义去重")
#         return input_file
    
#     # 初始化LSH并执行去重
#     lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
#     keep_indices = []
#     duplicate_count = 0
    
#     for i in tqdm(range(len(texts)), desc="MinHash去重"):
#         minhash = create_minhash_signature(texts[i])
#         similar_docs = lsh.query(minhash)
#         if not similar_docs:
#             lsh.insert(str(i), minhash)
#             keep_indices.append(i)
#         else:
#             duplicate_count += 1
    
#     stats["minhash_duplicated"] = duplicate_count
    
#     # 保存去重后的数据
#     output_file = os.path.join(INTERMEDIATE_PATH, "minhash_dedup_with_sensitive_filter.jsonl")
#     with open(output_file, "w", encoding="utf-8") as f_out:
#         for idx in keep_indices:
#             f_out.write(json.dumps(data_list[idx], ensure_ascii=False) + "\n")
    
#     # 统计结果
#     stats["stage_time"]["minhash_lsh"] = round(time.time() - start_time, 2)
#     remaining = len(keep_indices)
#     logging.info(f"✅ Minhash LSH语义去重完成 | 耗时：{stats['stage_time']['minhash_lsh']}秒 | "
#                  f"语义相似重复：{stats['minhash_duplicated']} | 剩余：{remaining}")
#     return output_file

import hashlib
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import time

from config import MINHASH_NUM_PERM, LSH_THRESHOLD
from utils import load_jsonl_files_with_sampling

def md5_deduplication(data_generator, stats):
    """MD5精确去重（接收数据生成器，返回去重后的数据列表）"""
    import logging
    start_time = time.time()
    logging.info("🚀 开始MD5精确去重")
    
    md5_set = set()
    deduplicated_data = []
    
    for item in tqdm(data_generator, desc="MD5去重", total=stats["sampled_count"] if stats["sampled_count"] > 0 else None):
        text = item["text"]
        md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        
        if md5_hash in md5_set:
            stats["md5_duplicated"] += 1
            continue
        
        md5_set.add(md5_hash)
        item["md5"] = md5_hash
        deduplicated_data.append(item)
    
    logging.info(f"✅ MD5去重完成 | 完全重复：{stats['md5_duplicated']}条 | 剩余：{len(deduplicated_data)}条")
    return deduplicated_data

def create_minhash_signature(text, num_perm=MINHASH_NUM_PERM):
    """创建MinHash签名（2-gram捕捉中文语义）"""
    minhash = MinHash(num_perm=num_perm)
    if len(text) < 2:
        grams = [text] if text else []
    else:
        grams = [text[i:i+2] for i in range(len(text)-1)]
    for gram in grams:
        token_hash = hashlib.sha256(gram.encode('utf-8')).hexdigest()
        minhash.update(token_hash.encode('utf-8'))
    return minhash

def minhash_lsh_deduplication(data_list, stats):
    """MinHash LSH语义去重（接收数据列表，返回去重后的数据列表）"""
    import logging
    start_time = time.time()
    logging.info(f"🚀 开始Minhash LSH语义去重（阈值{ LSH_THRESHOLD }）")
    
    if not data_list:
        logging.warning("⚠️ 无有效数据，跳过语义去重")
        return data_list
    
    texts = [item["text"] for item in data_list]
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    keep_indices = []
    duplicate_count = 0
    
    for i in tqdm(range(len(texts)), desc="MinHash去重"):
        minhash = create_minhash_signature(texts[i])
        similar_docs = lsh.query(minhash)
        if not similar_docs:
            lsh.insert(str(i), minhash)
            keep_indices.append(i)
        else:
            duplicate_count += 1
    
    stats["minhash_duplicated"] = duplicate_count
    deduplicated_data = [data_list[idx] for idx in keep_indices]
    
    logging.info(f"✅ Minhash LSH语义去重完成 | 语义相似重复：{stats['minhash_duplicated']}条 | 剩余：{len(deduplicated_data)}条")
    return deduplicated_data