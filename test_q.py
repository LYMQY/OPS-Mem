# Qwen3-8B 在线调用测试脚本 - 使用Few-Shot学习解决线性规划问题（OR-Tools）
import json
import re
import sys
import os
from openai import OpenAI, APITimeoutError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts.fewshot_prompt import Q2C
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

Deepseek_API_Key = os.getenv('SiliconFlow_API_KEY')

# 模型加载 - Qwen3-8B
# load model and tokenizer
client = OpenAI(api_key=os.getenv("SiliconFlow_API_KEY"), base_url="https://api.siliconflow.cn/v1")

# 带重试和超时的API调用函数
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((APITimeoutError, httpx.ReadTimeout))
)

def get_response(messages):

    response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=messages
    )

    return response.choices[0].message.content

results = []

def solve_problem(problem):
    
    ortools_prompt = Q2C(problem['question'])
        
    # 使用Qwen聊天模板格式化输入
    input_text = [
            {'role': 'system', 'content': 'Please follow the given examples and use python code to solve the given question.'},
            {'role': 'user', 'content': ortools_prompt}
        ]
        
    # 生成代码
    generated_text = get_response(input_text)
        
    # 提取代码块
    code_match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
    if not code_match:
        print(f"警告: 在生成的文本中未找到有效的代码块 for problem #{problem['index']}")
        results.append({
            'index': problem['index'],
            'error': '未找到有效的Python代码块'
        })
        
    code = code_match.group(1).strip()
    # print(f"\n提取的代码 for problem #{problem['index']}:")
    #print("```python")
    #print(code)
    #print("```")
        
    try:
        # 将生成的代码存储起来
        results.append({
            'index': problem['index'],
            'question': problem['question'],
            'answer': problem['answer'],
            'generated_code': code
        })
        return f"Successfully generated code for problem #{problem['index']}"
    
    except (APITimeoutError, httpx.ReadTimeout) as e:
        print(f"错误: 问题 #{problem['index']} 超时")
        results.append({
            'index': problem['index'],
            'question': problem['question'],
            'answer': problem['answer'],
            'error': f"API超时: {str(e)}"
        })
        return f"Failed: problem #{problem['index']} - API超时"

    except Exception as e:
        error_msg = str(e)
        results.append({
            'index': problem['index'],
            'question': problem['question'],
            'generated_code': code,
            'error': error_msg
        })
        return f"Failed to generate code for problem #{problem['index']}: {error_msg}"


json_file = "data/testset_json/optibench.json"


with open(json_file, "r", encoding="utf-8") as file:
    problems = json.load(file)

max_workers = 8  # 例如，如果有8核CPU，就使用32个线程

print(f"🚀 Starting to add notes using {max_workers} threads...")
start_time = time.time()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(solve_problem, problem) for problem in problems]

    for future in as_completed(futures):
        print(future.result())

end_time = time.time()
elapsed_time = end_time - start_time

with open("results.json", "w", encoding="utf-8") as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent=4)

print(f"✅ All question items added in {elapsed_time:.2f} seconds.")

