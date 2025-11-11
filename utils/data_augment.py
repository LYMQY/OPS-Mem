import utils.augment as aug
import json
import random
from tqdm import tqdm
import time 


seed_data_path = f'data/seed_data.jsonl'
aug_data_path = f'data/aug_data.jsonl'

seed_datas = []
with open(seed_data_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        seed_datas.append(data['question'])


augment = aug.Augment()

# for seed_data in seed_datas:
#     random_num = random.choice(list(range(7)))
#     print(f"[SEED {random_num}]")
#     if random_num == 0:
#         another_data = random.choice(seed_datas)
#         new_data = augment(ques=seed_data, ques2=another_data, seed=random_num)
#     else:
#         new_data = augment(ques=seed_data, seed=random_num)
#     print(f"[AUGMENTED] {new_data}")
    
#     with open(aug_data_path, 'a', encoding='utf-8') as file:
#         json_data = json.dumps(new_data)
#         file.write(json_data + '\n')

# 使用tqdm创建进度条
with tqdm(total=len(seed_datas[123:]), desc="处理子文件夹", unit="folder") as pbar:
    for seed_data in seed_datas[123:]:
        random_num = random.choice(list(range(7)))
        # 更新进度条描述
        pbar.set_description(f"[SEED {random_num}]")

        if random_num == 0:
            another_data = random.choice(seed_datas)
            new_data = augment(ques=seed_data, ques2=another_data, seed=random_num)
        else:
            new_data = augment(ques=seed_data, seed=random_num)
        
        new_data = {"question": new_data}
        with open(aug_data_path, 'a', encoding='utf-8') as file:
            json_data = json.dumps(new_data)
            file.write(json_data + '\n')
        
        # 更新进度条
        pbar.update(1)
            
        # 每处理完一个子文件夹后等待5秒
        time.sleep(4)
    