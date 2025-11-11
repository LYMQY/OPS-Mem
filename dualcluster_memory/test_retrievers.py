import json
from typing import Dict
from retrievers import ChromaRetriever
import os  

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def test_chroma_retriever_full_workflow():
    """测试ChromaRetriever完整工作流程：初始化→添加簇→搜索→删除→验证删除结果"""
    # 1. 初始化测试参数
    test_collection_name = "test_clusters"  # 测试用集合名（避免干扰正式数据）
    test_model_name = "all-MiniLM-L6-v2"    # 与ChromaRetriever默认模型一致
    # 测试用簇数据（模拟建模簇/实现簇的信息）
    test_clusters: Dict[str, Dict] = {
        "cluster_001": {
            "content": "车辆路径问题-网络流建模簇",  # 簇核心文本
            "metadata": {
                "problem_description": "多车辆配送路径优化，最小化总距离",  # 非默认值，会被加入增强内容
                "modeling_logic": ["弧变量x[i,j]∈{0,1}", "流守恒约束", "车辆数量限制"],  # 列表类型元数据
                "tags": ["VRP", "网络流", "整数规划"],  # 列表类型元数据
                "typical_cases": ["case_101", "case_102"],  # 关联案例ID
                "create_time": "2025-11-03",
                "cluster_type": "modeling"  # 建模簇标识
            }
        },
        "cluster_002": {
            "content": "Gurobi字典风格实现簇",  # 簇核心文本
            "metadata": {
                "problem_description": "General",  # 默认值，不会被加入增强内容
                "modeling_logic": [],  # 空列表，不会被加入增强内容
                "tags": ["Python", "Gurobi", "字典变量"],
                "tech_stack": "Python + Gurobi 10.0",
                "applicable_scale": "中小规模（<5000变量）",
                "cluster_type": "implementation"  # 实现簇标识
            }
        }
    }
    # 测试查询（分别匹配建模簇和实现簇）
    test_queries = [
        "多车辆配送的网络流建模方法",  # 应优先匹配cluster_001（建模簇）
        "用字典存储变量的Gurobi实现"   # 应优先匹配cluster_002（实现簇）
    ]


    # 2. 初始化ChromaRetriever（测试集合隔离）
    print("="*50)
    print("1. 初始化ChromaRetriever测试实例")
    print("="*50)
    try:
        retriever = ChromaRetriever(
            collection_name=test_collection_name,
            model_name=test_model_name
        )
        # 重置集合（确保测试环境干净，避免历史数据干扰）
        retriever.client.reset()
        # 重新创建干净的测试集合
        retriever = ChromaRetriever(
            collection_name=test_collection_name,
            model_name=test_model_name
        )
        print(f"✅ 初始化成功，集合名：{test_collection_name}，模型：{test_model_name}")
    except Exception as e:
        print(f"❌ 初始化失败：{str(e)}")
        return


    # 3. 测试添加簇（add_cluster）
    print("\n" + "="*50)
    print("2. 测试添加簇（add_cluster）")
    print("="*50)
    added_cluster_ids = []
    for cluster_id, cluster_data in test_clusters.items():
        try:
            retriever.add_cluster(
                cluster=cluster_data["content"],
                metadata=cluster_data["metadata"],
                cluster_id=cluster_id
            )
            added_cluster_ids.append(cluster_id)
            print(f"✅ 成功添加簇：{cluster_id}")
            # 验证增强内容（打印前100字符）
            enhanced_content = cluster_data["content"]
            if cluster_data["metadata"]["problem_description"] != "General":
                enhanced_content += f" problem_description: {cluster_data['metadata']['problem_description']}"
            if cluster_data["metadata"]["modeling_logic"]:
                enhanced_content += f" modeling_logic: {', '.join(cluster_data['metadata']['modeling_logic'])}"
            print(f"   增强嵌入内容：{enhanced_content[:100]}..." if len(enhanced_content) > 100 else f"   增强嵌入内容：{enhanced_content}")
        except Exception as e:
            print(f"❌ 添加簇 {cluster_id} 失败：{str(e)}")


    # 4. 测试搜索（search）与元数据反序列化
    print("\n" + "="*50)
    print("3. 测试搜索（search）与元数据反序列化")
    print("="*50)
    for query in test_queries:
        print(f"\n📝 测试查询：{query}")
        try:
            results = retriever.search(query=query, k=2)
            # 验证搜索结果结构
            required_keys = ["ids", "metadatas", "documents", "distances"]
            if all(key in results for key in required_keys):
                print(f"✅ 搜索结果结构完整，返回 {len(results['ids'][0])} 条结果")
                # 遍历结果，验证元数据反序列化（列表/数字类型是否恢复）
                for idx, (doc_id, metadata, doc, distance) in enumerate(zip(
                    results["ids"][0], results["metadatas"][0], results["documents"][0], results["distances"][0]
                )):
                    print(f"\n   结果{idx+1}：")
                    print(f"   - 簇ID：{doc_id}")
                    print(f"   - 相似度距离：{distance:.4f}（值越小越相似）")
                    print(f"   - 存储的文档内容：{doc[:80]}..." if len(doc) > 80 else f"   - 存储的文档内容：{doc}")
                    # 验证元数据反序列化（列表类型是否恢复）
                    if "tags" in metadata and isinstance(metadata["tags"], list):
                        print(f"   - 反序列化验证：tags（列表）={metadata['tags']}")
                    if "modeling_logic" in metadata and isinstance(metadata["modeling_logic"], list):
                        print(f"   - 反序列化验证：modeling_logic（列表）={metadata['modeling_logic']}")
            else:
                print(f"❌ 搜索结果结构不完整，缺少关键键：{[k for k in required_keys if k not in results]}")
        except Exception as e:
            print(f"❌ 搜索失败：{str(e)}")


    # 5. 测试删除文档（delete_document）
    print("\n" + "="*50)
    print("4. 测试删除文档（delete_document）")
    print("="*50)
    for cluster_id in added_cluster_ids:
        try:
            # 先查询删除前是否存在
            pre_delete_results = retriever.search(query=f"簇ID:{cluster_id}", k=1)
            if cluster_id in pre_delete_results["ids"][0]:
                # 执行删除
                retriever.delete_document(cluster_id=cluster_id)
                # 验证删除结果（删除后查询应无此ID）
                post_delete_results = retriever.search(query=f"簇ID:{cluster_id}", k=1)
                if cluster_id not in post_delete_results["ids"][0]:
                    print(f"✅ 成功删除簇：{cluster_id}（删除后查询无结果）")
                else:
                    print(f"❌ 簇 {cluster_id} 删除失败（删除后仍能查询到）")
            else:
                print(f"⚠️  簇 {cluster_id} 未找到，无需删除")
        except Exception as e:
            print(f"❌ 删除簇 {cluster_id} 失败：{str(e)}")


    # 6. 清理测试环境（删除测试集合）
    print("\n" + "="*50)
    print("5. 清理测试环境")
    print("="*50)
    try:
        retriever.client.delete_collection(name=test_collection_name)
        print(f"✅ 成功删除测试集合：{test_collection_name}")
        print("\n🎉 所有测试流程执行完毕！")
    except Exception as e:
        print(f"❌ 清理测试集合失败：{str(e)}")


# 执行测试
if __name__ == "__main__":
    test_chroma_retriever_full_workflow()