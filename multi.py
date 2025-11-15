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
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# 全局配置
CHUNK_SIZE = 160  # 每个进程处理的问题数量
PROCESS_MAX_WORKERS = min(multiprocessing.cpu_count() // 2, 4)  # 进程数（CPU核心数的一半，最大4）
THREAD_MAX_WORKERS = 2  # 每个进程内的线程数（控制API并发）

# 带重试和超时的API调用函数（线程内使用）
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((APITimeoutError, httpx.ReadTimeout))
)
def get_response(client, messages):
    """每个线程使用独立进程的client调用API"""
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=messages,
        timeout=30  # 30秒超时
    )
    return response.choices[0].message.content

def process_single_problem(problem, client):
    """单个问题处理（线程安全）：生成代码并返回结果字典"""
    problem_index = problem['index']
    question = problem['question']
    answer = problem.get('answer')
    
    try:
        # 生成提示词
        ortools_prompt = Q2C(question)
        
        # 构造输入消息
        input_text = [
            {'role': 'system', 'content': 'Please follow the given examples and use python code to solve the given question.'},
            {'role': 'user', 'content': ortools_prompt}
        ]
        
        # 调用模型生成代码
        generated_text = get_response(client, input_text)
        
        # 提取代码块
        code_match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
        if not code_match:
            print(f"警告: 问题 #{problem_index} 未找到有效代码块")
            return {
                'index': problem_index,
                'question': question,
                'answer': answer,
                'error': '未找到有效的Python代码块'
            }
        
        code = code_match.group(1).strip()
        return {
            'index': problem_index,
            'question': question,
            'answer': answer,
            'generated_code': code
        }
    
    except (APITimeoutError, httpx.ReadTimeout) as e:
        error_msg = f"API超时: {str(e)}"
        print(f"错误: 问题 #{problem_index} - {error_msg}")
        return {
            'index': problem_index,
            'question': question,
            'answer': answer,
            'error': error_msg
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"错误: 问题 #{problem_index} - {error_msg}")
        return {
            'index': problem_index,
            'question': question,
            'answer': answer,
            'error': error_msg
        }

def process_chunk(chunk, results):
    """单个进程的核心逻辑：处理100条问题的chunk（进程内多线程）"""
    chunk_first_idx = chunk[0]['index'] if chunk else "N/A"
    chunk_last_idx = chunk[-1]['index'] if chunk else "N/A"
    chunk_len = len(chunk)
    print(f"📌 进程启动：处理问题 {chunk_first_idx} - {chunk_last_idx}（共 {len(chunk)} 条）")
    
    # 每个进程初始化独立的OpenAI客户端
    client = OpenAI(
        api_key=os.getenv("SiliconFlow_API_KEY1"),
        base_url="https://api.siliconflow.cn/v1"
    )
    
    # 进程内启动多线程处理chunk
    with ThreadPoolExecutor(max_workers=THREAD_MAX_WORKERS) as thread_executor:
        # 提交线程任务
        futures = [
            thread_executor.submit(process_single_problem, problem, client)
            for problem in chunk
        ]
        
        # 用tqdm追踪进度：total为任务总数，desc为进度条描述
        with tqdm(total=chunk_len, desc=f"进程[{chunk_first_idx}-{chunk_last_idx}]", leave=True) as pbar:
            # 收集线程结果
            for future in as_completed(futures):
                result = future.result()
                results.append(result)  # Manager.list线程/进程安全
                pbar.update(1)  # 每完成一个任务，进度条+1
    
    print(f"✅ 进程完成：问题 {chunk_first_idx} - {chunk_last_idx} 处理完毕")

def split_problems_into_chunks(problems, chunk_size):
    """将问题列表按chunk_size拆分"""
    chunks = []
    for i in range(0, len(problems), chunk_size):
        chunk = problems[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    # 1. 读取问题数据集
    json_file = "data/testset_json/optibench.json"
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            problems = json.load(file)
        total_problems = len(problems)
        print(f"✅ 成功读取 {total_problems} 个问题")
    except Exception as e:
        print(f"❌ 读取数据集失败: {str(e)}")
        sys.exit(1)
    
    # 2. 拆分问题为chunk（每块100条）
    chunks = split_problems_into_chunks(problems, CHUNK_SIZE)
    print(f"📊 拆分后共 {len(chunks)} 个chunk，每个chunk最多 {CHUNK_SIZE} 条问题")
    
    # 3. 初始化多进程共享结果列表
    with Manager() as manager:
        results = manager.list()
        start_time = time.time()
        
        # 4. 启动多进程处理各chunk
        print(f"🚀 启动 {PROCESS_MAX_WORKERS} 个进程，每个进程内 {THREAD_MAX_WORKERS} 个线程...")
        with multiprocessing.Pool(processes=PROCESS_MAX_WORKERS) as process_pool:
            # 提交进程任务：每个进程处理一个chunk
            process_futures = [
                process_pool.apply_async(process_chunk, args=(chunk, results))
                for chunk in chunks
            ]
            
            # 等待所有进程完成
            for future in process_futures:
                future.get()  # 阻塞等待进程完成，捕获进程内异常
        
        # 5. 结果写入JSON文件
        results_list = list(results)
        with open("results.json", "w", encoding="utf-8") as outfile:
            json.dump(results_list, outfile, ensure_ascii=False, indent=4)
        
        # 6. 输出统计信息
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n🎉 所有任务完成！")
        print(f"总处理时间：{elapsed_time:.2f} 秒")
        print(f"总问题数：{total_problems}")
        print(f"成功生成代码数：{len([r for r in results_list if 'generated_code' in r])}")
        print(f"失败数：{len([r for r in results_list if 'error' in r])}")
        print(f"结果已保存到 results.json")