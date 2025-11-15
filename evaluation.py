import json
import subprocess
from dotenv import load_dotenv
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

load_dotenv()

with open("results.json", "r", encoding="utf-8") as file:
    codes = json.load(file)

client = OpenAI(api_key=os.getenv("SiliconFlow_API_KEY"), base_url="https://api.siliconflow.cn/v1")

def get_response(messages):

    response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=messages
    )

    return response.choices[0].message.content

# execute the code
def test_code(code_str):
    code_path = f"./test.py"
    with open(code_path, "w") as f1:
        encoded_code = f"# -*- coding: utf-8 -*-\n{code_str}"
        f1.write(encoded_code)

    ans = subprocess.run(f"python {code_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return answer logs, error code
    return str(ans.stdout.decode('gbk', errors='ignore')), str(ans.stderr.decode('gbk', errors='ignore'))

def envaluate(problem, results):

    prompts = f"""
    For the following optimization problem, correct answer is given and possible solution answer is given. Please judge whether the solution are correct.
    ```question
    {problem['question']}
    ```
    ```correct answer
    {problem['answer']}
    ```
    ```solution
    {results}

    Please return in the following format.
    ```judgement
    The solution is [Fill in True/False here].
    ```
"""
    messages = [
        {"role": "user", "content": prompts}
    ]

    response = get_response(messages=messages)

    return response

results = []

def evaluate_task(item):

    try:
        code_str = item['generated_code']
        out_log, err_log = test_code(code_str)
        evaluation = envaluate(item, out_log)
        results.append({
            'index': item['index'],
            'evaluation': evaluation
        })
        return f"Problem #{item['index']} Evaluation: {evaluation}"
    except Exception as e:
        error_msg = str(e)
        results.append({
            'index': item['index'],
            'error': error_msg
        })
        return f"Failed to evaluate problem #{item['index']}: {error_msg}"

# 使用多线程评估代码
max_workers = 24  # 例如，如果有8核CPU，就使用32个线程

print(f"🚀 Starting to add notes using {max_workers} threads...")
start_time = time.time()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(evaluate_task, problem) for problem in codes]

    for future in as_completed(futures):
        print(future.result())

end_time = time.time()
elapsed_time = end_time - start_time



# for item in codes[0:1]:
#     code_str = item['generated_code']
#     out_log, err_log = test_code(code_str)
#     evaluation = envaluate(item, out_log)
#     print(f"Problem #{item['index']} Evaluation: {evaluation}")
#     # results.append({
#     #     'index': item['index'],
#     #     'question': item['question'],
#     #     'answer': item['answer'],
#     #     'code_ans': out_log
#     # })

with open("test_code_results.json", "w", encoding="utf-8") as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent=4)

print(f"✅ All question items evaluated in {elapsed_time:.2f} seconds.")

