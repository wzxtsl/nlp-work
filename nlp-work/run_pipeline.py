import os
import subprocess
import time
import re

def run_script(script_name, description):
    """运行指定脚本并实时输出日志"""
    start_time = time.time()
    print(f"\n====== 开始 {description} ======")
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ 脚本不存在：{script_path}")
        exit(1)
    
    try:
        # 实时输出脚本日志
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line.strip())
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"返回码：{process.returncode}")
        
        end_time = time.time()
        print(f"====== {description}完成，耗时：{end_time - start_time:.2f}秒 ======")
    except Exception as e:
        print(f"❌ 执行{description}出错：{str(e)}")
        exit(1)

if __name__ == "__main__":
    # 第一步：运行筛选流程
    run_script(
        script_name="filter.py",
        description="文本筛选（生成高质量数据）"
    )
    
    # 检查筛选结果文件
    filtered_input = os.path.join("data/output", "clmmu_kept_data_final.jsonl")
    if not os.path.exists(filtered_input):
        print(f"❌ 筛选结果不存在：{filtered_input}")
        exit(1)
    print(f"✅ 筛选结果路径：{filtered_input}")
    
    # 第二步：修改改写配置，指定输入为筛选后的文件
    # 找到 rewrite_config.py 的位置（在 rewrite/ 子目录下）
    rewrite_config_path = os.path.join(os.path.dirname(__file__), "rewrite", "rewrite_config.py")
    if not os.path.exists(rewrite_config_path):
        print(f"❌ 改写配置文件不存在：{rewrite_config_path}")
        exit(1)
    
    # 替换 rewrite_config.py 中的 INPUT_DATA_PATH
    with open(rewrite_config_path, "r", encoding="utf-8") as f:
        config_content = f.read()
    # 用正则替换输入路径（确保匹配原配置中的格式）
    new_config = re.sub(
        r'INPUT_DATA_PATH\s*=\s*".*?"',  # 匹配 INPUT_DATA_PATH = "任意内容"
        f'INPUT_DATA_PATH = "{filtered_input}"',  # 替换为筛选后的路径
        config_content
    )
    with open(rewrite_config_path, "w", encoding="utf-8") as f:
        f.write(new_config)
    print(f"✅ 已更新改写输入路径：{filtered_input}")
    
    # 第三步：运行改写流程（入口是 rewrite/rewrite.py）
    run_script(
        script_name="rewrite/rewrite.py",  # 明确指定子目录下的脚本
        description="文本改写（优化高困惑度和冗余文本）"
    )

    # 检查改写输出文件
    rewritten_output = os.path.join("data", "rewrite_output", "rewritten_data.jsonl")
    if not os.path.exists(rewritten_output):
        print(f"❌ 改写结果不存在：{rewritten_output}")
        exit(1)
    print(f"✅ 改写结果路径：{rewritten_output}")

    # 第四步：运行QA生成流程（入口：qa/qa_generate.py）
    run_script(
        script_name="qa/qa_generate.py",
        description="问答生成（基于改写文本生成高质量问答对）"
    )

    qa_output_path = os.path.join("data", "qa_output", "qa_pairs.jsonl")
    if not os.path.exists(qa_output_path):
        print(f"⚠️ QA输出文件未找到：{qa_output_path}，请检查日志")
    else:
        print(f"✅ QA结果路径：{qa_output_path}")
    
    # 输出最终结果路径
    print("\n🎉 全流程执行完成！")
    print(f"1. 筛选结果：{filtered_input}")
    print(f"2. 改写结果：{os.path.join('rewrite', 'data', 'rewrite_output', 'rewritten_data.jsonl')}")  # 按你的输出路径修改
    print(f"3. QA结果：{os.path.join('data', 'qa_output', 'qa_pairs.jsonl')}")
