import subprocess
import prompts.generate_prompt as generate_prompt
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import time

load_dotenv()

# load model and tokenizer
client = OpenAI(api_key=os.getenv("SiliconFlow_API_KEY"), base_url="https://api.siliconflow.cn/v1")

def get_response(messages):

    response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=messages
    )

    return response.choices[0].message.content

# inference to get five elements
def infer_five_elem(question):

    messages = [
        {"role": "user", "content": generate_prompt.Q2F(question)}
    ]
    
    response = get_response(messages=messages)
    response = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")

    if "```text" in response:
        return response.split("```text")[1].split("```")[0]
    elif "```plaintext" in response:
        return response.split("```plaintext")[1].split("```")[0]
    elif "```" in response:
        return response.split("```")[1].split("```")[0]
    else:
        return None


# inference to get pyomo python code
def infer_code(five_elem):
    messages = [
        {"role": "user", "content": generate_prompt.F2C(five_elem)}
    ]

    response = get_response(messages=messages)
    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')

# inference to get gurobi python code
def infer_codeG(five_elem):

    messages = [
        {"role": "user", "content": generate_prompt.F2G(five_elem)}
    ]

    response = get_response(messages=messages)
    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')

# inference to get copt python code
def infer_codeC(five_elem):
    messages = [
        {"role": "user", "content": generate_prompt.F2CO(five_elem)}
    ]
    response = get_response(messages=messages)
    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')

# inference to get ortools python code
def infer_codeO(five_elem):

    messages = [
        {"role": "user", "content": generate_prompt.F2O(five_elem)}
    ]

    response = get_response(messages=messages)
    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')

# execute the code
def test_code(code_str):
    code_path = f"./test.py"
    with open(code_path, "w") as f1:
        f1.write(code_str)

    ans = subprocess.run(f"python {code_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return answer logs, error code
    return str(ans.stdout.decode('gbk', errors='ignore')), str(ans.stderr.decode('gbk', errors='ignore'))

questions = []
with open("data/aug_data.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line.strip())
        questions.append(data['question'])
question = questions[0]
five_elem = infer_five_elem(question)
time.sleep(4)
code_str = infer_code(five_elem)
time.sleep(10)
code_gurobi = infer_codeG(five_elem)
time.sleep(10)
code_copt = infer_codeC(five_elem)
time.sleep(10)
code_ortools = infer_codeO(five_elem)
#out_log, err_log = test_code(code_str)
print("----- OUTPUT -----")
print(five_elem)
print("----- PYOMO CODE -----")
print(code_str)
print("----- GUROBI CODE -----")
print(code_gurobi)
print("----- COPT CODE -----")
print(code_copt)
print("----- OR-TOOLS CODE -----")
print(code_ortools)
#print(out_log)
