import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever
import json
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
import time

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DualClusterMemorySystem")

class MemoryNode:
    """Case Layer（L1）- 具体问题的完整求解记录"""
    def __init__(self,
                 problem_description: Optional[str] = None,
                 modeling_logic: Optional[str] = None,
                 key_constraint_snippets: Optional[str] = None,
                 full_code: Optional[str] = None,
                 modeling_cluster_id: Optional[str] = None,
                 implementation_cluster_id: Optional[str] = None,
                 id: Optional[str] = None,
                 timestamp: Optional[str] = None):
        """初始化记忆节点，存储问题-模型-代码全流程信息"""
        # 基础标识
        self.id = id or str(uuid.uuid4())  # 唯一ID
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 创建时间
        self.status = "pending"  # 状态：pending(待整合)/integrated(已整合)

        # 核心内容（问题-模型-代码）
        self.problem_description = problem_description or "General Problem"  # 问题描述
        self.modeling_logic = modeling_logic or "General Modeling Logic"  # 建模逻辑（自然语言）
        self.key_constraint_snippets = key_constraint_snippets or "General Key Constraint Snippets"  # 关键约束代码片段
        self.full_code = full_code or "General Full Code"  # 完整实现代码

        # 双簇关联
        self.modeling_cluster_id = modeling_cluster_id  # 所属建模簇ID
        self.implementation_cluster_id = implementation_cluster_id  # 所属实现簇ID
        
        # 嵌入向量（延迟生成，添加时计算）
        self.modeling_embedding: Optional[np.ndarray] = None  # 建模逻辑嵌入
        self.implementation_embedding: Optional[np.ndarray] = None  # 实现代码嵌入

        # 检索与演化相关字段
        self.retrieval_count = 0  # 检索次数（用于权重调整）
        self.links = []  # 关联的其他记忆节点ID
        self.evolution_history = []  # 演化记录


class DualClusterMemorySystem:
    """双簇记忆系统：基于MemoryNode实现建模/实现双簇管理"""
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 10,  # 簇更新阈值：累计10个pending节点触发整合
                 similarity_threshold: float = 0.3,  # 簇归属阈值：相似度<0.3归为已有簇
                 api_key: Optional[str] = None):  
        """初始化系统：创建双簇检索器、LLM控制器、演化参数"""
        # 本地记忆存储（key: MemoryNode.id, value: MemoryNode实例）
        self.memories: Dict[str, MemoryNode] = {}
        self.model_name = model_name

        # 1. 初始化ChromaDB双簇检索器（建模簇+实现簇）+ 全量记忆检索器
        try:
            # 重置旧集合，确保数据干净
            for coll_name in ["memories", "model", "implementation"]:
                temp_retriever = ChromaRetriever(collection_name=coll_name, model_name=model_name)
                temp_retriever.client.reset()
            logger.info("ChromaDB旧集合重置完成")
        except Exception as e:
            logger.warning(f"ChromaDB重置失败，使用新集合：{str(e)}")
        
        # 创建新检索器实例
        self.full_retriever = ChromaRetriever(collection_name="memories", model_name=model_name)  # 全量记忆
        self.model_retriever = ChromaRetriever(collection_name="model", model_name=model_name)    # 建模簇
        self.implementation_retriever = ChromaRetriever(collection_name="implementation", model_name=model_name)  # 实现簇

        # 2. 初始化LLM控制器（用于内容分析、演化决策）
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)

        # 3. 演化与检索参数
        self.evo_cnt = 0  # 待整合节点计数器
        self.evo_threshold = evo_threshold  # 演化触发阈值
        self.similarity_threshold = similarity_threshold  # 簇归属相似度阈值
        self.embedding_model = SentenceTransformer(model_name)  # 用于生成嵌入向量

        # 4. 演化决策Prompt（固定模板，引导LLM输出结构化结果）
        self._evolution_system_prompt = """
        你是记忆演化决策助手，需要判断新记忆节点是否需要与已有节点整合，并输出具体动作。
        新记忆信息：
        - 内容：{content}
        - 上下文：{context}
        - 关键词：{keywords}
        
        相似邻居记忆：
        {nearest_neighbors_memories}
        
        请按以下规则决策：
        1. 若新节点与邻居相似度高（主题/方法一致），should_evolve设为true，否则false；
        2. 动作（actions）可选：strengthen（强化新节点关联）、update_neighbor（更新邻居元数据）；
        3. 输出JSON格式，严格遵循schema，不额外添加内容。
        """

    def analyze_content(self, content: str) -> Dict:            
        """用LLM分析输入文本，提取MemoryNode所需的核心字段（问题/模型/代码）"""
        prompt = """
        分析以下文本，提取4个核心字段，输出JSON格式：
        1. description：问题描述（1-2句话总结待解决的问题）；
        2. modeling：建模逻辑（包含假设、参数、目标函数、约束条件）；
        3. implementation：关键约束代码片段（与业务约束相关的代码，如if/for逻辑）；
        4. code：完整实现代码（可运行的完整代码，包含导入、函数定义）。
        
        文本内容：{content}
        
        JSON格式要求：
        {{
            "description": "问题描述",
            "modeling": "建模逻辑",
            "implementation": "关键约束代码",
            "code": "完整代码"
        }}
        """.format(content=content)

        try:
            # 调用LLM并指定JSON输出格式
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "memory_extraction",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "modeling": {"type": "string"},
                                "implementation": {"type": "string"},
                                "code": {"type": "string"}
                            },
                            "required": ["description", "modeling", "implementation", "code"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"LLM内容分析失败：{str(e)}")
            # 失败时返回默认值，避免流程中断
            return {
                "description": "General Problem",
                "modeling": "General Modeling Logic",
                "implementation": "# No Key Constraint Code",
                "code": "# No Full Implementation Code"
            }

    def _generate_embeddings(self, text: str) -> np.ndarray:
        """生成文本的嵌入向量（用SentenceTransformer模型）"""
        return self.embedding_model.encode(text, convert_to_tensor=False)

    def _assign_cluster(self, node: MemoryNode) -> Tuple[Optional[str], Optional[str]]:
        """为新节点分配簇：基于相似度匹配已有簇，否则创建新簇（含簇中心和模式总结）"""
        modeling_cluster_id = None
        implementation_cluster_id = None

        # 1. 建模簇分配：基于建模逻辑的抽象总结匹配
        if node.modeling_logic.strip():
            # 生成原始建模逻辑嵌入（用于节点自身）
            node.modeling_embedding = self._generate_embeddings(node.modeling_logic)
            
            # 检索相似建模簇（k=1，取最相似）
            model_results = self.model_retriever.search(query=node.modeling_logic, k=1)
            #logger.info(f"{model_results}")
            
            if model_results["distances"] and len(model_results["distances"][0]) > 0:
                similarity = 1 / (1 + model_results["distances"][0][0])  # 距离转相似度
                if similarity <= self.similarity_threshold:
                    # 相似度达标：分配已有簇ID，并更新典型案例
                    modeling_cluster_id = model_results["ids"][0][0]
                    # 获取簇元数据并更新典型案例列表
                    cluster_metadata = model_results["metadatas"][0][0]
                    updated_node_ids = cluster_metadata["典型案例"] + [node.id]
                    # 重新添加簇（更新元数据）
                    self.model_retriever.delete_document(modeling_cluster_id)
                    self.model_retriever.add_cluster(
                        cluster=cluster_metadata["簇中心"],  # 保持原簇中心总结
                        metadata={
                            "type": "modeling",
                            "簇中心": cluster_metadata["簇中心"],
                            "模式详细总结": cluster_metadata["模式详细总结"],
                            "典型案例": updated_node_ids
                        },
                        cluster_id=modeling_cluster_id
                    )
            elif modeling_cluster_id == None:
                    # 相似度不达标：创建新簇（生成簇中心和模式总结）
                    modeling_cluster_id = str(uuid.uuid4())
                    # 调用LLM生成簇中心总结和模式详细描述
                    cluster_summary = self._generate_modeling_cluster_summary(node.modeling_logic)
                    # 生成簇中心的抽象嵌入（用于后续簇间匹配）
                    cluster_embedding = self._generate_embeddings(cluster_summary["簇中心"])
                    # 存入新簇
                    self.model_retriever.add_cluster(
                        cluster=cluster_summary["簇中心"],  # 用簇中心总结作为检索文本
                        metadata={
                            "type": "modeling",
                            "簇中心": cluster_summary["簇中心"],
                            "模式详细总结": cluster_summary["模式详细总结"],
                            "典型案例": [node.id]  # 初始典型案例为当前节点
                        },
                        cluster_id=modeling_cluster_id
                    )
                    # 存储簇中心嵌入（用于节点与簇的精确匹配）
                    node.modeling_cluster_embedding = cluster_embedding

        # 2. 实现簇分配：基于代码实现的抽象总结匹配
        if node.full_code.strip():
            # 生成原始代码嵌入（用于节点自身）
            node.implementation_embedding = self._generate_embeddings(node.full_code)
                
            # 检索相似实现簇（k=1，取最相似）
            impl_results = self.implementation_retriever.search(query=node.full_code, k=1)
                
            if impl_results["distances"] and len(impl_results["distances"][0]) > 0:
                similarity = 1 / (1 + impl_results["distances"][0][0])  # 距离转相似度
                if similarity <= self.similarity_threshold:
                    # 相似度达标：分配已有簇ID，更新典型案例
                    implementation_cluster_id = impl_results["ids"][0][0]
                    cluster_metadata = impl_results["metadatas"][0][0]
                    updated_node_ids = cluster_metadata["典型案例"] + [node.id]
                    # 重新添加簇（更新元数据）
                    self.implementation_retriever.delete_document(implementation_cluster_id)
                    self.implementation_retriever.add_cluster(
                        cluster=cluster_metadata["簇中心"],
                        metadata={
                            "type": "implementation",
                            "簇中心": cluster_metadata["簇中心"],
                            "模式详细总结": cluster_metadata["模式详细总结"],
                            "典型案例": updated_node_ids
                        },
                        cluster_id=implementation_cluster_id
                    )
            elif implementation_cluster_id == None:
                    # 相似度不达标：创建新簇（生成簇中心和模式总结）
                    implementation_cluster_id = str(uuid.uuid4()) 
                    # 调用LLM生成代码实现的簇总结
                    cluster_summary = self._generate_implementation_cluster_summary(node.full_code)
                    # 生成簇中心的抽象嵌入
                    cluster_embedding = self._generate_embeddings(cluster_summary["簇中心"])
                    # 存入新簇
                    self.implementation_retriever.add_cluster(
                        cluster=cluster_summary["簇中心"],
                        metadata={
                            "type": "implementation",
                            "簇中心": cluster_summary["簇中心"],
                            "模式详细总结": cluster_summary["模式详细总结"],
                            "典型案例": [node.id]
                        },
                        cluster_id=implementation_cluster_id
                    )
                    # 存储簇中心嵌入
                    node.implementation_cluster_embedding = cluster_embedding

        return modeling_cluster_id, implementation_cluster_id

    # 新增：生成建模簇的中心总结和模式描述
    def _generate_modeling_cluster_summary(self, modeling_logic: str) -> Dict[str, str]:
        """用LLM抽象建模逻辑，生成簇中心和模式详细总结"""
        prompt = f"""
        请分析以下建模逻辑，生成结构化的簇特征：
        1. 簇中心：1句话总结核心建模思路（如“车辆路径问题的网络流模型”）；
        2. 模式详细总结：分点描述适用场景、核心变量、必须约束、可选扩展。
        
        建模逻辑：{modeling_logic}
        
        输出格式：
        {{
            "簇中心": "核心建模思路总结",
            "模式详细总结": "适用：...\\n核心：...\\n必须包含：...\\n可选扩展：..."
        }}
        """
        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format={"type": "json_object"}
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"建模簇总结生成失败：{e}")
            return {
                "簇中心": "未分类建模逻辑",
                "模式详细总结": f"适用：未知场景\\n核心：{modeling_logic[:50]}...\\n必须包含：无\\n可选扩展：无"
            }

    # 新增：生成实现簇的中心总结和模式描述
    def _generate_implementation_cluster_summary(self, full_code: str) -> Dict[str, str]:
        """用LLM抽象代码实现，生成簇中心和模式详细总结"""
        prompt = f"""
        请分析以下代码实现，生成结构化的簇特征：
        1. 簇中心：1句话总结代码风格和技术栈（如“Python+Gurobi的字典式变量管理”）；
        2. 模式详细总结：分点描述技术栈、代码风格、适用规模、性能特点。
        
        代码实现：{full_code}
        
        输出格式：
        {{
            "簇中心": "代码实现风格总结",
            "模式详细总结": "技术栈：...\\n代码风格：...\\n适用场景：...\\n性能：..."
        }}
        """
        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format={"type": "json_object"}
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"实现簇总结生成失败：{e}")
            return {
                "簇中心": "未分类代码实现",
                "模式详细总结": f"技术栈：未知\\n代码风格：{full_code[:50]}...\\n适用场景：未知\\n性能：未知"
            }

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note：LLM补全字段→分配双簇→存入检索器→计数演化"""
        # Create MemoryNote without llm_controller
        if time is not None:
            kwargs['timestamp'] = time
        node = MemoryNode(**kwargs)

        # 🔧 LLM Analysis Enhancement: Auto-generate attributes using LLM if they are empty or default values
        needs_analysis = (
            node.problem_description == "General Problem" or  # problem_description is empty
            node.modeling_logic == "General Modeling Logic" or  # modeling_logic is default value
            node.key_constraint_snippets == "General Key Constraint Snippets" or  # key_constraint_snippets is default value
            node.full_code == "General Full Code"  # full_code is default value
        )
        
        if needs_analysis:
            
            try:
                # 用LLM分析内容，提取MemoryNode核心字段
                analysis = self.analyze_content(content)

                # Only update attributes that are not provided or have default values
                if node.problem_description == "General Problem":
                    node.problem_description = analysis["description"]
                if node.modeling_logic == "General Modeling Logic":
                    node.modeling_logic = analysis["modeling"]
                if node.key_constraint_snippets == "General Key Constraint Snippets":
                    node.key_constraint_snippets = analysis["implementation"]
                if node.full_code == "General Full Code":
                    node.full_code = analysis["code"]
            
            except Exception as e:
                print(f"Warning: LLM analysis failed, using default values: {e}")

        #logger.info(f"建模逻辑：{node.modeling_logic}")
        #logger.info(f"完整代码：{node.full_code}")

        # Step 3：为节点分配双簇（建模簇+实现簇）
        node.modeling_cluster_id, node.implementation_cluster_id = self._assign_cluster(node)
        logger.info(f"新节点{node.id}分配簇：建模簇{node.modeling_cluster_id}，实现簇{node.implementation_cluster_id}")

        # Step 4：将节点存入本地存储和ChromaDB
        self.memories[node.id] = node
        
        # 4.1 存入全量记忆检索器（用于全量相似检索）
        full_metadata = {
            "id": node.id,
            "problem_description": node.problem_description,
            "modeling_cluster_id": node.modeling_cluster_id,
            "implementation_cluster_id": node.implementation_cluster_id,
            "timestamp": node.timestamp,
            "retrieval_count": node.retrieval_count,
            "status": node.status
        }
        self.full_retriever.add_cluster(
            cluster=f"问题：{node.problem_description}\n建模：{node.modeling_logic}\n代码：{node.full_code}",
            metadata=full_metadata,
            cluster_id=node.id
        )

        # Step 5：计数待整合节点，达到阈值触发簇整合
        self.evo_cnt += 1
        if self.evo_cnt % self.evo_threshold == 0:
            logger.info(f"待整合节点数达{self.evo_threshold}，触发簇整合")
            #self.consolidate_memories()

        return node.id
    
    def get_clusters(self, cluster_type: str = "all") -> Dict[str, Dict[str, Any]]:
        """
        列出当前系统中的簇信息。
        Args:
            cluster_type: "model" / "implementation" / "all"
        Returns:
            dict: {
                "modeling": { cluster_id: {"count": int, "node_ids": [...], "representative": str}, ... },
                "implementation": { ... }
            }
        """
        model_clusters: Dict[str, Dict[str, Any]] = {}
        impl_clusters: Dict[str, Dict[str, Any]] = {}

        # 按现有 memory 聚合簇信息（不依赖外部检索器接口，以保证可靠性）
        for node in self.memories.values():
            # 建模簇聚合
            if node.modeling_cluster_id:
                cid = node.modeling_cluster_id
                if cid not in model_clusters:
                    model_clusters[cid] = {
                        "count": 0,
                        "node_ids": [],
                        "representative": node.modeling_logic[:300] if node.modeling_logic else ""
                    }
                model_clusters[cid]["count"] += 1
                model_clusters[cid]["node_ids"].append(node.id)

            # 实现簇聚合
            if node.implementation_cluster_id:
                cid = node.implementation_cluster_id
                if cid not in impl_clusters:
                    impl_clusters[cid] = {
                        "count": 0,
                        "node_ids": [],
                        "representative": node.full_code[:300] if node.full_code else ""
                    }
                impl_clusters[cid]["count"] += 1
                impl_clusters[cid]["node_ids"].append(node.id)

        result: Dict[str, Dict[str, Any]] = {}
        if cluster_type in ("all", "model"):
            result["modeling"] = model_clusters
        if cluster_type in ("all", "implementation"):
            result["implementation"] = impl_clusters

        return result

    def consolidate_memories(self):
        """簇整合：更新双簇检索器的簇信息（合并相似簇、更新节点关联）"""
        # 1. 重置建模簇检索器，重新整合所有建模逻辑
        self.model_retriever.client.reset()
        model_clusters: Dict[str, List[str]] = {}  # key: 簇ID，value: 关联节点ID列表
        
        # 2. 重新聚合建模簇（按已有簇ID分组）
        for node in self.memories.values():
            if not node.modeling_cluster_id:
                continue
            if node.modeling_cluster_id not in model_clusters:
                model_clusters[node.modeling_cluster_id] = []
            model_clusters[node.modeling_cluster_id].append(node.id)
        
        # 3. 重新存入建模簇（更新关联节点列表）
        for cluster_id, node_ids in model_clusters.items():
            # 找到簇的代表性建模逻辑（取第一个节点的建模逻辑）
            representative_node = self.memories[node_ids[0]]
            self.model_retriever.add_cluster(
                cluster=representative_node.modeling_logic,
                metadata={"type": "modeling", "related_node_ids": node_ids},
                cluster_id=cluster_id
            )

        # 4. 实现簇整合（逻辑同建模簇）
        self.implementation_retriever.client.reset()
        impl_clusters: Dict[str, List[str]] = {}
        for node in self.memories.values():
            if not node.implementation_cluster_id:
                continue
            if node.implementation_cluster_id not in impl_clusters:
                impl_clusters[node.implementation_cluster_id] = []
            impl_clusters[node.implementation_cluster_id].append(node.id)
        
        for cluster_id, node_ids in impl_clusters.items():
            representative_node = self.memories[node_ids[0]]
            self.implementation_retriever.add_cluster(
                cluster=representative_node.full_code,
                metadata={"type": "implementation", "related_node_ids": node_ids},
                cluster_id=cluster_id
            )

        # 5. 更新所有节点状态为“已整合”
        for node in self.memories.values():
            node.status = "integrated"
            node.evolution_history.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 完成簇整合")

        # 重置待整合计数器
        self.evo_cnt = 0
        logger.info("双簇整合完成，所有节点状态更新为integrated")
    
    def read(self, memory_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve

        Returns:
            MemoryNode if found, None otherwise
        """
        return self.memories.get(memory_id)

    def find_related_memories(self, query: str, k: int = 5, cluster_type: str = "all") -> Tuple[str, List[str]]:
        """检索相似记忆：支持全量/建模簇/实现簇检索"""
        if not self.memories:
            return "", []

        try:
            # 选择检索器（all: 全量，model: 建模簇，implementation: 实现簇）
            if cluster_type == "model":
                results = self.model_retriever.search(query=query, k=k)
                # 建模簇返回的是簇ID，需映射到关联的节点ID
                node_ids = []
                for i, cluster_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    node_ids.extend(metadata.get("related_node_ids", []))
                results["ids"][0] = node_ids[:k]  # 取前k个节点ID
            elif cluster_type == "implementation":
                results = self.implementation_retriever.search(query=query, k=k)
                # 实现簇同理，映射到节点ID
                node_ids = []
                for i, cluster_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    node_ids.extend(metadata.get("related_node_ids", []))
                results["ids"][0] = node_ids[:k]
            else:
                results = self.full_retriever.search(query=query, k=k)

            # 格式化检索结果
            memory_str = ""
            related_ids = []
            for i, node_id in enumerate(results["ids"][0]):
                node = self.memories.get(node_id)
                if not node:
                    continue
                # 更新检索次数
                node.retrieval_count += 1
                self.memories[node_id] = node  # 保存更新后的检索次数
                
                # 格式化输出信息
                memory_str += (
                    f"记忆ID: {node.id}\n"
                    f"问题描述: {node.problem_description[:100]}...\n"
                    f"建模逻辑: {node.modeling_logic[:100]}...\n"
                    f"所属建模簇: {node.modeling_cluster_id}\n"
                    f"所属实现簇: {node.implementation_cluster_id}\n"
                    f"检索次数: {node.retrieval_count}\n"
                    f"-------------------------\n"
                )
                related_ids.append(node_id)
            
            return memory_str, related_ids
        except Exception as e:
            logger.error(f"检索相似记忆失败: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """原始格式检索结果（包含关联节点信息）"""
        if not self.memories:
            return ""
            
        # 先获取基础检索结果
        _, related_ids = self.find_related_memories(query, k=k)
        memory_str = ""
        
        for node_id in related_ids[:k]:
            node = self.memories.get(node_id)
            if not node:
                continue
                
            # 添加主节点信息
            memory_str += (
                f"时间: {node.timestamp}\n"
                f"问题: {node.problem_description}\n"
                f"建模: {node.modeling_logic[:150]}\n"
                f"代码片段: {node.key_constraint_snippets[:100]}\n"
                f"-------------------------\n"
            )
            
            # 添加关联节点（通过links字段）
            for link_id in node.links[:2]:  # 每个节点最多显示2个关联节点
                link_node = self.memories.get(link_id)
                if link_node:
                    memory_str += (
                        f"关联记忆ID: {link_id}\n"
                        f"关联问题: {link_node.problem_description[:100]}\n"
                        f"-------------------------\n"
                    )
        
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNode]:
        """通过ID读取记忆节点"""
        node = self.memories.get(memory_id)
        if node:
            # 读取时更新最后访问时间（扩展字段，需在MemoryNode中添加）
            node.last_accessed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.memories[memory_id] = node
        return node

    def update(self, memory_id: str, **kwargs) -> bool:
        """更新记忆节点字段（支持问题描述、建模逻辑等核心字段）"""
        if memory_id not in self.memories:
            logger.warning(f"更新失败：记忆节点{memory_id}不存在")
            return False
            
        node = self.memories[memory_id]
        # 记录原始簇ID（用于后续判断是否需要重新分配簇）
        old_model_cluster_id = node.modeling_cluster_id
        old_impl_cluster_id = node.implementation_cluster_id
        
        # 更新字段（仅更新MemoryNode中存在的属性）
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
                logger.info(f"节点{memory_id}更新字段: {key}")
        
        # 若建模逻辑或代码变更，需重新生成嵌入并分配簇
        if "modeling_logic" in kwargs or "full_code" in kwargs:
            new_model_cluster_id, new_impl_cluster_id = self._assign_cluster(node)
            node.modeling_cluster_id = new_model_cluster_id
            node.implementation_cluster_id = new_impl_cluster_id
            logger.info(f"节点{memory_id}簇信息更新：建模簇{new_model_cluster_id}，实现簇{new_impl_cluster_id}")
        
        # 更新ChromaDB中的全量记忆
        full_metadata = {
            "id": node.id,
            "problem_description": node.problem_description,
            "modeling_cluster_id": node.modeling_cluster_id,
            "implementation_cluster_id": node.implementation_cluster_id,
            "timestamp": node.timestamp,
            "retrieval_count": node.retrieval_count,
            "status": node.status
        }
        self.full_retriever.delete_document(memory_id)
        self.full_retriever.add_cluster(
            cluster=f"问题：{node.problem_description}\n建模：{node.modeling_logic}\n代码：{node.full_code}",
            metadata=full_metadata,
            cluster_id=memory_id
        )
        
        # 记录演化历史
        node.evolution_history.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 手动更新节点")
        self.memories[memory_id] = node
        return True

    def delete(self, memory_id: str) -> bool:
        """删除记忆节点（同步删除双簇关联）"""
        if memory_id not in self.memories:
            return False
            
        node = self.memories[memory_id]
        # 1. 从双簇中移除节点关联（简化处理：标记簇需重新整合）
        if node.modeling_cluster_id:
            logger.info(f"节点{memory_id}从建模簇{node.modeling_cluster_id}移除")
        if node.implementation_cluster_id:
            logger.info(f"节点{memory_id}从实现簇{node.implementation_cluster_id}移除")
        
        # 2. 从ChromaDB删除
        self.full_retriever.delete_document(memory_id)
        
        # 3. 从本地存储删除
        del self.memories[memory_id]
        logger.info(f"节点{memory_id}已完全删除")
        return True

    def process_memory(self, node: MemoryNode) -> Tuple[bool, MemoryNode]:
        """记忆演化处理：通过LLM判断是否与相似节点整合"""
        if not self.memories or len(self.memories) == 1:  # 只有当前节点时无需演化
            return False, node
            
        try:
            # 1. 获取最相似的5个节点作为邻居
            neighbors_text, neighbor_ids = self.find_related_memories(
                query=node.problem_description, 
                k=5, 
                cluster_type="all"
            )
            if not neighbors_text or not neighbor_ids:
                return False, node
                
            # 2. 构造LLM演化决策提示
            prompt = self._evolution_system_prompt.format(
                content=node.problem_description,
                context=node.modeling_logic,
                keywords=node.key_constraint_snippets[:50],  # 用关键约束作为关键词
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(neighbor_ids)
            )
            
            # 3. 调用LLM获取演化决策
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evolution_decision",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "suggested_connections": {"type": "array", "items": {"type": "string"}},
                                "new_context_neighborhood": {"type": "array", "items": {"type": "string"}},
                                "tags_to_update": {"type": "array", "items": {"type": "string"}},
                                "new_tags_neighborhood": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
                            },
                            "required": ["should_evolve", "actions", "suggested_connections"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            response_json = json.loads(response)
            should_evolve = response_json["should_evolve"]
            
            if not should_evolve:
                return False, node
                
            # 4. 执行演化动作
            actions = response_json["actions"]
            for action in actions:
                if action == "strengthen":
                    # 强化关联：添加建议的节点链接
                    node.links.extend([
                        link_id for link_id in response_json["suggested_connections"] 
                        if link_id in self.memories and link_id not in node.links
                    ])
                    node.evolution_history.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 强化关联：{node.links}")
                    
                elif action == "update_neighbor":
                    # 更新邻居节点的上下文和标签（此处简化为更新建模逻辑）
                    new_contexts = response_json.get("new_context_neighborhood", [])
                    for i, neighbor_id in enumerate(neighbor_ids[:len(new_contexts)]):
                        neighbor_node = self.memories.get(neighbor_id)
                        if neighbor_node:
                            neighbor_node.modeling_logic = new_contexts[i]  # 更新建模逻辑
                            neighbor_node.evolution_history.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 被节点{node.id}更新建模逻辑")
                            self.memories[neighbor_id] = neighbor_node  # 保存更新
            
            return True, node
            
        except Exception as e:
            logger.error(f"记忆演化处理失败: {str(e)}")
            return False, node

    def search(self, query: str, k: int = 5, cluster_type: str = "all") -> List[Dict[str, Any]]:
        """结构化检索接口：返回包含节点详情的字典列表"""
        _, related_ids = self.find_related_memories(query, k=k, cluster_type=cluster_type)
        results = []
        
        for node_id in related_ids[:k]:
            node = self.memories.get(node_id)
            if not node:
                continue
                
            results.append({
                "id": node.id,
                "problem_description": node.problem_description,
                "modeling_logic": node.modeling_logic,
                "full_code": node.full_code[:200] + "...",  # 截断长代码
                "modeling_cluster_id": node.modeling_cluster_id,
                "implementation_cluster_id": node.implementation_cluster_id,
                "retrieval_count": node.retrieval_count,
                "timestamp": node.timestamp,
                "status": node.status
            })
        
        return results