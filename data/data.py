import json
import os
from pathlib import Path

base_dir = os.path.dirname(os.path.abspath(__file__))

source_dir = os.path.join(base_dir, "testset")
dest_dir = os.path.join(base_dir, "testset_json")

for filename in os.listdir(source_dir):
    # 只处理 .jsonl 后缀的文件
    if filename.endswith(".jsonl"):
        src_path = os.path.join(source_dir, filename)
        # 生成目标文件名（替换后缀为 .json）
        dest_filename = os.path.splitext(filename)[0] + ".json"
        dest_path = os.path.join(dest_dir, dest_filename)
            
        # 读取 JSONL 内容并转换为 JSON 数组
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                json_list = [json.loads(line.strip()) for line in f if line.strip()]
                
            # 写入目标 JSON 文件
            with open(dest_path, 'w', encoding='utf-8') as f:
                json.dump(json_list, f, ensure_ascii=False, indent=2)
                
            print(f"已转换：{src_path} -> {dest_path}")
        except Exception as e:
            print(f"处理 {src_path} 时出错：{str(e)}")