import utils.augment as aug
import json
import random


seed_data_path = f'./seed_data.jsonl'
aug_data_path = f'./aug_data.jsonl'

seed_datas = []
with open(seed_data_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        seed_datas.append(data['question'])


augment = aug.Augment()

for seed_data in seed_datas:
    random_num = random.choice(list(range(7)))
    print(f"[SEED {random_num}]")
    if random_num == 0:
        another_data = random.choice(seed_datas)
        new_data = augment(ques=seed_data, ques2=another_data, seed=random_num)
    else:
        new_data = augment(ques=seed_data, seed=random_num)
    
    with open(aug_data_path, 'a', encoding='utf-8') as file:
        json_data = json.dumps(new_data)
        file.write(json_data + '\n')
    