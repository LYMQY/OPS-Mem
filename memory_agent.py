from dualcluster_memory.memory_system import DualClusterMemorySystem
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import json

load_dotenv()

Deepseek_API_Key = os.getenv('SiliconFlow_API_KEY')

memory_system = DualClusterMemorySystem(
    model_name="model/all-MiniLM-L6-v2",
    llm_backend="deepseek",
    llm_model="deepseek-ai/DeepSeek-V3",
    evo_threshold=3,  # 降低阈值，方便测试簇整合
    similarity_threshold=0.6,
    api_key=Deepseek_API_Key
)

memorys = []

with open("data/aug_data_q2f2c.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line.strip())
        memorys.append(data)

memorys = memorys[0:20]

#print(memorys[0])

print(f"✅ Loaded {len(memorys)} memory items.")

# 添加记忆节点(单线程)
# for memory in memorys:
#     memory_system.add_note(problem_description=memory['question'], modeling_logic=memory['five_elem'], full_code=memory['code_ortools'])
# print("✅ All memory items added.")

# --- 多线程修改部分开始 ---

def add_note_task(memory):
    """
    定义一个任务函数，用于在单个线程中执行 add_note 操作。
    每个线程将处理一个 memory 条目。
    """
    try:
        memory_system.add_note(
            problem_description=memory['question'],
            modeling_logic=memory['five_elem'],
            full_code=memory['code_ortools']
        )
        # 返回成功信息
        return f"Successfully added memory: {memory.get('question', 'No question')[:50]}..."
    except Exception as e:
        # 返回失败信息和错误详情
        return f"Failed to add memory: {memory.get('question', 'No question')[:50]}... Error: {e}"

# 定义要使用的线程数量。
# 一个经验法则是使用 CPU 核心数的 2-4 倍。
# 对于 I/O 密集型任务（如网络请求），可以设置得更高。
# os.cpu_count() 会返回 CPU 的核心数。
max_workers = 8  # 例如，如果有8核CPU，就使用32个线程

print(f"🚀 Starting to add notes using {max_workers} threads...")
start_time = time.time()

# 使用 ThreadPoolExecutor 上下文管理器来管理线程池
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # executor.submit() 将任务提交给线程池，并返回一个 Future 对象列表
    # 我们遍历 memorys 列表，为每个 memory 提交一个 add_note_task 任务
    futures = [executor.submit(add_note_task, memory) for memory in memorys]

    # as_completed() 会在每个任务完成时返回其 Future 对象
    # 这样我们可以实时看到任务的完成情况
    for future in as_completed(futures):
        # future.result() 会获取任务的返回值
        print(future.result())

end_time = time.time()
elapsed_time = end_time - start_time

print(f"✅ All memory items added in {elapsed_time:.2f} seconds.")

model = memory_system.get_clusters(cluster_type="model")
print("Modeling Clusters:")
print(len(model["modeling"]))
print(model["modeling"].keys())
implementation = memory_system.get_clusters(cluster_type="implementation")
print("Implementation Clusters:")
print(len(implementation["implementation"]))
print(implementation["implementation"].keys())

models = memory_system.search("vehicle routing problem", k=10, cluster_type="model")
print("Model Search Results for 'vehicle routing problem':")
print(models)
codes = memory_system.search("vehicle routing problem", k=10, cluster_type="implementation")
print("Code Search Results for 'vehicle routing problem':")
print(codes)

# questions = "A man on a strict diet only drinks meal replacement drinks from two brands, alpha and omega. The alpha brand drink contains 30 grams of protein, 20 grams of sugar, and 350 calories per bottle. The omega brand drink contains 20 grams of protein, 15 grams of sugar, and 300 calories per bottle. The man wants to get at least 100 grams of protein and 2000 calories. In addition, because the omega brand drink contains tiny amounts of caffeine, at most 35% of the drink should be omega brand. How many bottles of each should he drink to minimize his sugar intake?"

# model = memory_system.search(questions, k=2, cluster_type="model")
# print("Model Search Results:")
# print(model)

# code = memory_system.search(questions, k=2, cluster_type="implementation")
# print("Code Search Results:")
# print(code)

# policy1 = model[2]["modeling_clusters"][0]["pattern_summary"]
# policy2 = model[2]["modeling_clusters"][1]["pattern_summary"]
# pol1 = code[2]["implementation_clusters"][0]["pattern_summary"]
# pol2 = code[2]["implementation_clusters"][1]["pattern_summary"]

# prompts = f"""
#     You are an expert in the field of operations and optimization. You need to help to solve the following optimization problem:
#     {questions}
# """

# model_query = f"""
#     There are two optimization problem modeling policy related to the above problem.
#     [Policy 1]
#     {policy1}
#     [Policy 2]
#     {policy2}
#     Thera are two optimization problem code implementation policy related to the above problem.
#     [Policy 1]
#     {pol1}
#     [Policy 2]
#     {pol2}
#     Please combine the above modeling policy and code implementation policy and provide four detailed solutions and its scores for the above optimization problem.

#     Return the result in the following JSON format:
#     {{
#         "solution_1": {{
#             "detailed_solution": "...",
#             "scores": "..."
#         }},
#         "solution_2": {{
#             "detailed_solution": "...",
#             "scores": "..."
#         }},
#         "solution_3": {{    
#             "detailed_solution": "...",
#             "scores": "..."
#         }},
#         "solution_4": {{
#             "detailed_solution": "...",
#             "scores": "..."
#         }}
#     }}
# """

# # load model and tokenizer
# client = OpenAI(api_key=os.getenv("SiliconFlow_API_KEY"), base_url="https://api.siliconflow.cn/v1")

# def get_response(messages):

#     response = client.chat.completions.create(
#             model="deepseek-ai/DeepSeek-V3",
#             messages=messages
#     )

#     return response.choices[0].message.content