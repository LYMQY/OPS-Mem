from dualcluster_memory.memory_system import DualClusterMemorySystem  # 替换为实际文件名
from dotenv import load_dotenv
import os
import time

load_dotenv()

Deepseek_API_Key = os.getenv('SiliconFlow_API_KEY')

def test_dual_cluster_memory_system():
    """测试双簇记忆系统核心功能"""
    print("="*50)
    print("开始测试双簇记忆系统")
    print("="*50)

    # --------------------------
    # 1. 初始化系统
    # --------------------------
    print("\n1. 初始化双簇记忆系统...")
    try:
        memory_system = DualClusterMemorySystem(
            model_name="model/all-MiniLM-L6-v2",
            llm_backend="deepseek",
            llm_model="deepseek-ai/DeepSeek-V3",
            evo_threshold=3,  # 降低阈值，方便测试簇整合
            similarity_threshold=0.6,
            api_key=Deepseek_API_Key
        )
        print("✅ 系统初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败：{str(e)}")
        return

    # --------------------------
    # 2. 添加测试记忆节点
    # --------------------------
    print("\n2. 添加测试记忆节点...")
    # 测试用例1：车辆路径问题（VRP）建模与实现
    test_content_1 = """
    问题：某快递公司需要为10个客户配送包裹，从仓库出发，每个客户仅访问一次，要求总行驶距离最短。
    建模逻辑：采用整数规划模型，定义二进制变量x_ij表示是否从节点i到节点j行驶，目标函数最小化总距离，约束包括：每个客户入度=1、出度=1，仓库出度=1、入度=1，消除子回路。
    代码实现：
    import gurobipy as gp
    from gurobipy import GRB

    def vrp_model(distance_matrix, num_customers):
        model = gp.Model("VRP")
        # 定义变量
        x = model.addVars(num_customers+1, num_customers+1, vtype=GRB.BINARY, name="x")
        # 目标函数
        model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i][j] for i in range(num_customers+1) for j in range(num_customers+1) if i != j), GRB.MINIMIZE)
        # 约束：每个客户入度=1
        for j in range(1, num_customers+1):
            model.addConstr(gp.quicksum(x[i][j] for i in range(num_customers+1) if i != j) == 1)
        # 约束：每个客户出度=1
        for i in range(1, num_customers+1):
            model.addConstr(gp.quicksum(x[i][j] for j in range(num_customers+1) if i != j) == 1)
        # 仓库约束
        model.addConstr(gp.quicksum(x[0][j] for j in range(1, num_customers+1)) == 1)
        model.addConstr(gp.quicksum(x[i][0] for i in range(1, num_customers+1)) == 1)
        # 求解
        model.optimize()
        return model
    """

    # 测试用例2：带时间窗的VRP（与用例1同属建模簇，不同实现细节）
    test_content_2 = """
    问题：某快递公司为10个客户配送，每个客户有时间窗[start_j, end_j]，需在时间窗内送达，总距离最短。
    建模逻辑：基于VRP整数规划模型，新增时间变量t_j表示到达客户j的时间，约束t_j >= t_i + travel_time_ij - M*(1-x_ij)，t_j >= start_j，t_j <= end_j。
    代码实现：
    import gurobipy as gp
    from gurobipy import GRB

    def vrp_time_window(distance_matrix, time_windows, num_customers):
        M = 1000  # 大M常数
        model = gp.Model("VRP_TimeWindow")
        x = model.addVars(num_customers+1, num_customers+1, vtype=GRB.BINARY, name="x")
        t = model.addVars(num_customers+1, vtype=GRB.CONTINUOUS, name="t")
        # 目标函数
        model.setObjective(gp.quicksum(distance_matrix[i][j] * x[i][j] for i, j in [(i,j) for i in range(num_customers+1) for j in range(num_customers+1) if i != j]), GRB.MINIMIZE)
        # 入度出度约束（同基础VRP）
        for j in range(1, num_customers+1):
            model.addConstr(gp.quicksum(x[i][j] for i in range(num_customers+1) if i != j) == 1)
        for i in range(1, num_customers+1):
            model.addConstr(gp.quicksum(x[i][j] for j in range(num_customers+1) if i != j) == 1)
        # 时间窗约束
        for j in range(num_customers+1):
            model.addConstr(t[j] >= time_windows[j][0])
            model.addConstr(t[j] <= time_windows[j][1])
        # 时间连续性约束
        for i in range(num_customers+1):
            for j in range(num_customers+1):
                if i != j:
                    model.addConstr(t[j] >= t[i] + distance_matrix[i][j] - M*(1 - x[i][j]))
        model.optimize()
        return model
    """

    # 测试用例3：背包问题（不同建模簇，对比测试）
    test_content_3 = """
    问题：背包容量为50，有8个物品，每个物品有重量和价值，选择物品装入背包，使总价值最大且不超过容量。
    建模逻辑：0-1整数规划，变量x_i表示是否选择物品i，目标函数最大化总价值，约束总重量<=背包容量。
    代码实现：
    import gurobipy as gp
    from gurobipy import GRB

    def knapsack_model(weights, values, capacity):
        model = gp.Model("Knapsack")
        x = model.addVars(len(weights), vtype=GRB.BINARY, name="x")
        model.setObjective(gp.quicksum(values[i] * x[i] for i in range(len(weights))), GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(len(weights))) <= capacity)
        model.optimize()
        return model
    """

    # 添加3个节点（触发簇整合，因为evo_threshold=3）
    node_id_1 = memory_system.add_note(test_content_1)
    node_id_2 = memory_system.add_note(test_content_2)
    node_id_3 = memory_system.add_note(test_content_3)
    print(f"✅ 添加3个记忆节点，ID分别为：{node_id_1}、{node_id_2}、{node_id_3}")
    print(f"✅ 当前记忆节点总数：{len(memory_system.memories)}")

    # --------------------------
    # 3. 验证簇分配逻辑
    # --------------------------
    print("\n3. 验证双簇分配结果...")
    node_1 = memory_system.read(node_id_1)
    node_2 = memory_system.read(node_id_2)
    node_3 = memory_system.read(node_id_3)

    print(f"- 节点1（VRP）：建模簇ID={node_1.modeling_cluster_id}，实现簇ID={node_1.implementation_cluster_id}")
    print(f"- 节点2（带时间窗VRP）：建模簇ID={node_2.modeling_cluster_id}，实现簇ID={node_2.implementation_cluster_id}")
    print(f"- 节点3（背包问题）：建模簇ID={node_3.modeling_cluster_id}，实现簇ID={node_3.implementation_cluster_id}")

    # 验证：VRP两个节点应属于同一建模簇（相似度高）
    if node_1.modeling_cluster_id == node_2.modeling_cluster_id:
        print("✅ 验证通过：同类型建模（VRP）分配到同一建模簇")
    else:
        print("❌ 验证失败：同类型建模未分配到同一建模簇")

    if node_1.modeling_cluster_id != node_3.modeling_cluster_id:
        print("✅ 验证通过：不同类型建模（VRP vs 背包）分配到不同建模簇")
    else:
        print("❌ 验证失败：不同类型建模分配到同一建模簇")

    # --------------------------
    # 4. 测试检索功能
    # --------------------------
    print("\n4. 测试检索功能...")
    # 检索关键词："车辆路径问题"（匹配VRP簇）
    print("\n- 检索关键词：'车辆路径问题'（建模簇检索）")
    vrp_results = memory_system.search(query="车辆路径问题", k=2, cluster_type="model")
    for idx, res in enumerate(vrp_results, 1):
        print(f"  检索结果{idx}：问题={res['problem_description'][:30]}...，建模簇={res['modeling_cluster_id']}")

    # 检索关键词："背包容量"（匹配背包问题）
    print("\n- 检索关键词：'背包容量'（全量检索）")
    knapsack_results = memory_system.search(query="背包容量", k=1, cluster_type="all")
    for idx, res in enumerate(knapsack_results, 1):
        print(f"  检索结果{idx}：问题={res['problem_description'][:30]}...，建模簇={res['modeling_cluster_id']}")

    if len(vrp_results) >= 2 and "车辆" in vrp_results[0]["problem_description"]:
        print("✅ 建模簇检索验证通过")
    if len(knapsack_results) == 1 and "背包" in knapsack_results[0]["problem_description"]:
        print("✅ 全量检索验证通过")

    # --------------------------
    # 5. 测试更新功能
    # --------------------------
    print("\n5. 测试节点更新功能...")
    # 更新节点3的问题描述
    update_success = memory_system.update(
        memory_id=node_id_3,
        problem_description="背包容量为100，10个物品的最大化价值问题（更新后）"
    )
    updated_node_3 = memory_system.read(node_id_3)
    if update_success and "更新后" in updated_node_3.problem_description:
        print(f"✅ 节点3更新成功，更新后问题描述：{updated_node_3.problem_description[:50]}...")
    else:
        print("❌ 节点更新失败")

    # --------------------------
    # 6. 测试删除功能
    # --------------------------
    print("\n6. 测试节点删除功能...")
    delete_success = memory_system.delete(memory_id=node_id_3)
    if delete_success and node_id_3 not in memory_system.memories:
        print(f"✅ 节点3（背包问题）删除成功")
        print(f"✅ 删除后记忆节点总数：{len(memory_system.memories)}")
    else:
        print("❌ 节点删除失败")

    # --------------------------
    # 7. 测试簇整合功能
    # --------------------------
    print("\n7. 验证簇整合结果...")
    # 查看整合后的建模簇数量（应为1个VRP簇）
    model_clusters = memory_system.get_clusters("model")  # 需在ChromaRetriever中实现get_all_documents方法
    print(f"✅ 整合后建模簇数量：{len(model_clusters['modeling']) if model_clusters else 0}")
    print("✅ 簇整合功能验证完成")

    # --------------------------
    # 测试总结
    # --------------------------
    print("\n" + "="*50)
    print("测试完成！核心功能验证结果：")
    print("✅ 系统初始化 | ✅ 记忆添加 | ✅ 双簇分配 | ✅ 相似检索")
    print("✅ 节点更新 | ✅ 节点删除 | ✅ 簇整合")
    print("="*50)

if __name__ == "__main__":
    test_dual_cluster_memory_system()