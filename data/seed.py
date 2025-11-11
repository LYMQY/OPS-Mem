import json
import os
from pathlib import Path
import random

base_dir = os.path.dirname(os.path.abspath(__file__))

source_dir = os.path.join(base_dir, "testset")
seed_data_path = os.path.join(base_dir, "seed_data.jsonl")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

for filename in os.listdir(source_dir):
    # 只处理 .jsonl 后缀的文件
    all = []
    if filename.endswith(".jsonl"):
        src_path = os.path.join(source_dir, filename)
        # 读取 JSONL 内容并转换为 JSON 数组
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                json_list = [json.loads(line.strip()) for line in f if line.strip()]
            totol = len(json_list)
            sample_size = int(totol * 0.05)
            sampled_json_list = random.sample(json_list, sample_size)
            all.extend(sampled_json_list)
            print(f"Sampled {sample_size} items from {totol} total items.")
        except Exception as e:
            print(f"处理 {src_path} 时出错：{str(e)}")
    # 写入目标 JSON 文件
    with open(seed_data_path, 'a', encoding='utf-8') as f:
        for item in all:
            json_data = json.dumps(item, ensure_ascii=False)
            f.write(json_data + '\n')