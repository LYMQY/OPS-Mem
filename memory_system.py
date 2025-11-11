import uuid
from datetime import datetime
import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch
import time
import os
import tempfile
import subprocess
import sys
from typing import List, Dict, Optional, Any, Tuple, Union

# 基础配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DualClusterMemorySystem")

# ------------------------------
# 1. 核心数据结构定义
# ------------------------------
class MemoryNode:
    """Case Layer（L1）- 具体问题的完整求解记录"""
    def __init__(self,
                 problem_description: str,
                 modeling_logic: str,
                 key_constraint_snippets: str,
                 full_code: str,
                 modeling_cluster_id: Optional[str] = None,
                 implementation_cluster_id: Optional[str] = None,
                 id: Optional[str] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            problem_description (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            modeling_logic (str): The modeling logic in natural language
            key_constraint_snippets (str): Key constraint code snippets
            full_code (str): The full implementation code
            modeling_cluster_id (Optional[str]): Associated modeling cluster ID
            implementation_cluster_id (Optional[str]): Associated implementation cluster ID
        """
        # 基础标识
        self.id = id or str(uuid.uuid4())  # 唯一ID
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 创建时间
        self.status = "pending"  # 状态：pending(待整合)/integrated(已整合)
        
        # 核心内容（问题-建模-实现）
        self.problem_description = problem_description  # 问题描述
        self.modeling_logic = modeling_logic  # 建模逻辑（自然语言）
        self.key_constraint_snippets = key_constraint_snippets  # 关键约束代码片段
        self.full_code = full_code  # 完整实现代码
        
        # 双簇关联
        self.modeling_cluster_id = modeling_cluster_id  # 所属建模簇ID
        self.implementation_cluster_id = implementation_cluster_id  # 所属实现簇ID
        
        # 嵌入向量（延迟生成，添加时计算）
        self.modeling_embedding: Optional[np.ndarray] = None  # 建模逻辑嵌入
        self.implementation_embedding: Optional[np.ndarray] = None  # 实现代码嵌入

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于存储和传输"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "status": self.status,
            "problem_description": self.problem_description,
            "modeling_logic": self.modeling_logic,
            "key_constraint_snippets": self.key_constraint_snippets,
            "full_code": self.full_code,
            "modeling_cluster_id": self.modeling_cluster_id,
            "implementation_cluster_id": self.implementation_cluster_id,
            "modeling_embedding": self.modeling_embedding.tolist() if self.modeling_embedding is not None else None,
            "implementation_embedding": self.implementation_embedding.tolist() if self.implementation_embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """从字典重建MemoryNode"""
        node = cls(
            problem_description=data["problem_description"],
            modeling_logic=data["modeling_logic"],
            key_constraint_snippets=data["key_constraint_snippets"],
            full_code=data["full_code"],
            modeling_cluster_id=data["modeling_cluster_id"],
            implementation_cluster_id=data["implementation_cluster_id"],
            id=data["id"]
        )
        node.timestamp = data["timestamp"]
        node.status = data["status"]
        node.modeling_embedding = np.array(data["modeling_embedding"]) if data["modeling_embedding"] else None
        node.implementation_embedding = np.array(data["implementation_embedding"]) if data["implementation_embedding"] else None
        return node


class ModelingCluster:
    """Abstraction Layer（L2）- Modeling Cluster：存储同类建模方法的抽象模式"""
    def __init__(self,
                 pattern_summary: str,
                 id: Optional[str] = None):
        self.id = id or f"B{str(uuid.uuid4())[:6]}"  # 建模簇ID，前缀B（Modeling）
        self.pattern_summary = pattern_summary  # 模式总结（适用场景、核心逻辑、扩展点）
        self.cluster_center: Optional[np.ndarray] = None  # 簇中心（所有节点建模嵌入的平均）
        self.typical_cases: List[str] = []  # 典型案例ID列表
        self.create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_time = self.create_time

    def update_cluster_center(self, modeling_embeddings: List[np.ndarray]) -> None:
        """更新簇中心：基于所有典型案例的建模嵌入计算平均"""
        if not modeling_embeddings:
            return
        self.cluster_center = np.mean(modeling_embeddings, axis=0)
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_typical_case(self, case_id: str) -> None:
        """添加典型案例（去重）"""
        if case_id not in self.typical_cases:
            self.typical_cases.append(case_id)
            self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_summary": self.pattern_summary,
            "cluster_center": self.cluster_center.tolist() if self.cluster_center is not None else None,
            "typical_cases": self.typical_cases,
            "create_time": self.create_time,
            "update_time": self.update_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelingCluster":
        cluster = cls(
            pattern_summary=data["pattern_summary"],
            id=data["id"]
        )
        cluster.cluster_center = np.array(data["cluster_center"]) if data["cluster_center"] else None
        cluster.typical_cases = data["typical_cases"]
        cluster.create_time = data["create_time"]
        cluster.update_time = data["update_time"]
        return cluster


class ImplementationCluster:
    """Abstraction Layer（L2）- Implementation Cluster：存储同类代码实现的抽象模式"""
    def __init__(self,
                 pattern_summary: str,
                 id: Optional[str] = None):
        self.id = id or f"I{str(uuid.uuid4())[:6]}"  # 实现簇ID，前缀I（Implementation）
        self.pattern_summary = pattern_summary  # 模式总结（技术栈、代码风格、适用规模、性能）
        self.cluster_center: Optional[np.ndarray] = None  # 簇中心（所有节点实现嵌入的平均）
        self.typical_cases: List[str] = []  # 典型案例ID列表
        self.create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_time = self.create_time

    def update_cluster_center(self, implementation_embeddings: List[np.ndarray]) -> None:
        """更新簇中心：基于所有典型案例的实现嵌入计算平均"""
        if not implementation_embeddings:
            return
        self.cluster_center = np.mean(implementation_embeddings, axis=0)
        self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_typical_case(self, case_id: str) -> None:
        """添加典型案例（去重）"""
        if case_id not in self.typical_cases:
            self.typical_cases.append(case_id)
            self.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_summary": self.pattern_summary,
            "cluster_center": self.cluster_center.tolist() if self.cluster_center is not None else None,
            "typical_cases": self.typical_cases,
            "create_time": self.create_time,
            "update_time": self.update_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImplementationCluster":
        cluster = cls(
            pattern_summary=data["pattern_summary"],
            id=data["id"]
        )
        cluster.cluster_center = np.array(data["cluster_center"]) if data["cluster_center"] else None
        cluster.typical_cases = data["typical_cases"]
        cluster.create_time = data["create_time"]
        cluster.update_time = data["update_time"]
        return cluster


# ------------------------------
# 2. 工具函数（嵌入生成、相似度计算、验证）
# ------------------------------
def generate_embedding(model_name: str, text: str) -> np.ndarray:
    """生成文本嵌入向量（基于预训练模型）"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    # 取[CLS] token的嵌入作为文本表示
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    # 归一化（提升相似度计算稳定性）
    return embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) != 0 else embedding


def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """计算两个嵌入向量的余弦相似度（0-1，值越大越相似）"""
    if emb1 is None or emb2 is None:
        return 0.0
    # 确保向量维度一致
    if emb1.shape != emb2.shape:
        return 0.0
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]


class CodeValidator:
    """代码验证器：负责语法检查、运行验证和性能评估"""
    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, str]:
        """语法检查：判断代码是否存在语法错误"""
        try:
            compile(code, filename="<string>", mode="exec")
            return True, "语法检查通过"
        except SyntaxError as e:
            return False, f"语法错误: {str(e)}"

    @staticmethod
    def run_validation(code: str, test_case: str = None, timeout: int = 30) -> Tuple[bool, str, float]:
        """运行验证：执行代码并检查是否能正常运行（支持小规模测试用例）"""
        # 构造完整代码（如果有测试用例，追加到代码末尾）
        full_test_code = code
        if test_case:
            full_test_code += "\n\n" + test_case

        # 使用临时文件执行代码（避免环境污染）
        start_time = time.time()
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
                temp_file.write(full_test_code)
                temp_file_path = temp_file.name

            # 执行代码并捕获输出
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # 计算运行时间
            run_time = time.time() - start_time

            # 检查执行结果
            if result.returncode == 0:
                return True, f"运行成功，输出: {result.stdout[:200]}..." if result.stdout else "运行成功，无输出", run_time
            else:
                return False, f"运行失败，错误: {result.stderr[:200]}...", run_time

        except subprocess.TimeoutExpired:
            return False, f"运行超时（超过{timeout}秒）", timeout
        except Exception as e:
            return False, f"验证异常: {str(e)}", time.time() - start_time
        finally:
            # 清理临时文件
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)


# ------------------------------
# 3. 核心系统：双簇记忆网络
# ------------------------------
class DualClusterMemorySystem:
    def __init__(self,
                 llm_controller: Any,  # LLM控制器（需实现get_completion方法，如对接GPT/ollama）
                 model_name: str = "all-MiniLM-L6-v2",
                 evo_threshold: int = 10,  # 簇更新阈值：累计pending节点数达到时触发整合
                 similarity_threshold: float = 0.7  # 簇归属阈值：相似度高于此值归为已有簇
                 ):
        # 依赖组件
        self.llm_controller = llm_controller  # LLM用于推理树构建和簇判断
        self.similarity_threshold = similarity_threshold  # 簇归属相似度阈值

        # 记忆存储（案例层+抽象层）
        self.memory_nodes: Dict[str, MemoryNode] = {}  # 案例节点：key=节点ID
        self.modeling_clusters: Dict[str, ModelingCluster] = {}  # 建模簇：key=簇ID
        self.implementation_clusters: Dict[str, ImplementationCluster] = {}  # 实现簇：key=簇ID

        # 交叉索引：(建模簇ID, 实现簇ID) → 案例节点ID列表（快速查询双簇交叉案例）
        self.cross_index: Dict[Tuple[str, str], List[str]] = {}

        # 演化控制
        self.pending_node_count = 0  # 待整合节点计数
        self.evo_threshold = evo_threshold  # 触发簇整合的阈值

    # ------------------------------
    # 3.1 记忆添加：新案例节点的插入与簇归属判断
    # ------------------------------
    def add_memory_node(self, node: MemoryNode) -> Tuple[bool, str, MemoryNode]:
        """添加新记忆节点，自动计算嵌入、判断簇归属"""
        try:
            # 1. 生成嵌入向量
            node.modeling_embedding = generate_embedding(node.modeling_logic)
            node.implementation_embedding = generate_embedding(node.full_code)

            # 2. 判断建模簇归属（已有簇匹配/新建簇）
            modeling_cluster_id, is_new_modeling_cluster = self._match_or_create_modeling_cluster(node)
            node.modeling_cluster_id = modeling_cluster_id

            # 3. 判断实现簇归属（已有簇匹配/新建簇）
            implementation_cluster_id, is_new_implementation_cluster = self._match_or_create_implementation_cluster(node)
            node.implementation_cluster_id = implementation_cluster_id

            # 4. 存储节点
            self.memory_nodes[node.id] = node
            self.pending_node_count += 1  # 增加待整合计数
            logger.info(f"添加记忆节点成功，ID: {node.id}，建模簇: {modeling_cluster_id}，实现簇: {implementation_cluster_id}")

            # 5. 更新交叉索引
            cross_key = (modeling_cluster_id, implementation_cluster_id)
            if cross_key not in self.cross_index:
                self.cross_index[cross_key] = []
            self.cross_index[cross_key].append(node.id)

            # 6. 检查是否触发簇整合（待整合节点达到阈值）
            if self.pending_node_count >= self.evo_threshold:
                self._consolidate_clusters()  # 整合所有簇（更新簇中心、模式总结）
                self.pending_node_count = 0  # 重置待整合计数

            return True, "添加成功", node
        except Exception as e:
            logger.error(f"添加记忆节点失败: {str(e)}")
            return False, f"添加失败: {str(e)}", node

    def _match_or_create_modeling_cluster(self, node: MemoryNode) -> Tuple[str, bool]:
        """匹配已有建模簇，若无则新建（基于嵌入相似度）"""
        if not self.modeling_clusters:  # 无任何建模簇时，新建第一个
            new_cluster = ModelingCluster(
                pattern_summary=self._generate_cluster_summary("modeling", [node])
            )
            new_cluster.add_typical_case(node.id)
            new_cluster.update_cluster_center([node.modeling_embedding])
            self.modeling_clusters[new_cluster.id] = new_cluster
            return new_cluster.id, True

        # 计算与所有建模簇中心的相似度
        cluster_similarities = []
        for cluster_id, cluster in self.modeling_clusters.items():
            if cluster.cluster_center is None:
                continue
            sim = calculate_similarity(node.modeling_embedding, cluster.cluster_center)
            cluster_similarities.append((cluster_id, sim))

        # 按相似度排序，取最高值
        cluster_similarities.sort(key=lambda x: x[1], reverse=True)
        top_cluster_id, top_sim = cluster_similarities[0] if cluster_similarities else (None, 0.0)

        # 若最高相似度高于阈值，归为该簇；否则新建簇
        if top_sim >= self.similarity_threshold and top_cluster_id:
            top_cluster = self.modeling_clusters[top_cluster_id]
            top_cluster.add_typical_case(node.id)
            return top_cluster_id, False
        else:
            # 新建建模簇，生成模式总结
            new_cluster = ModelingCluster(
                pattern_summary=self._generate_cluster_summary("modeling", [node])
            )
            new_cluster.add_typical_case(node.id)
            new_cluster.update_cluster_center([node.modeling_embedding])
            self.modeling_clusters[new_cluster.id] = new_cluster
            return new_cluster.id, True

    def _match_or_create_implementation_cluster(self, node: MemoryNode) -> Tuple[str, bool]:
        """匹配已有实现簇，若无则新建（基于嵌入相似度）"""
        if not self.implementation_clusters:  # 无任何实现簇时，新建第一个
            new_cluster = ImplementationCluster(
                pattern_summary=self._generate_cluster_summary("implementation", [node])
            )
            new_cluster.add_typical_case(node.id)
            new_cluster.update_cluster_center([node.implementation_embedding])
            self.implementation_clusters[new_cluster.id] = new_cluster
            return new_cluster.id, True

        # 计算与所有实现簇中心的相似度
        cluster_similarities = []
        for cluster_id, cluster in self.implementation_clusters.items():
            if cluster.cluster_center is None:
                continue
            sim = calculate_similarity(node.implementation_embedding, cluster.cluster_center)
            cluster_similarities.append((cluster_id, sim))

        # 按相似度排序，取最高值
        cluster_similarities.sort(key=lambda x: x[1], reverse=True)
        top_cluster_id, top_sim = cluster_similarities[0] if cluster_similarities else (None, 0.0)

        # 若最高相似度高于阈值，归为该簇；否则新建簇
        if top_sim >= self.similarity_threshold and top_cluster_id:
            top_cluster = self.implementation_clusters[top_cluster_id]
            top_cluster.add_typical_case(node.id)
            return top_cluster_id, False
        else:
            # 新建实现簇，生成模式总结
            new_cluster = ImplementationCluster(
                pattern_summary=self._generate_cluster_summary("implementation", [node])
            )
            new_cluster.add_typical_case(node.id)
            new_cluster.update_cluster_center([node.implementation_embedding])
            self.implementation_clusters[new_cluster.id] = new_cluster
            return new_cluster.id, True

    def _generate_cluster_summary(self, cluster_type: str, nodes: List[MemoryNode]) -> str:
        """生成簇的模式总结（调用LLM）"""
        if cluster_type == "modeling":
            # 提取建模相关信息
            modeling_infos = [
                f"问题：{node.problem_description[:50]}...\n建模逻辑：{node.modeling_logic}"
                for node in nodes
            ]
            prompt = f"""
            基于以下建模案例，生成建模簇的模式总结，包含3部分：
            1. 适用场景：该建模方法适合解决哪类问题（如"多车辆配送路径优化"）
            2. 核心逻辑：关键变量定义、核心约束（如"弧变量x[i,j]表示路径，流守恒约束确保路径完整"）
            3. 扩展点：可选添加的约束或优化方向（如"支持时间窗、客户优先级扩展"）
            
            案例列表：
            {chr(10).join(modeling_infos)}
            
            要求：简洁明了，控制在150字以内，避免具体案例细节。
            """
        else:  # implementation
            # 提取实现相关信息
            implementation_infos = [
                f"代码技术栈：{self._extract_tech_stack(node.full_code)}\n代码风格：{self._extract_code_style(node.full_code)}"
                for node in nodes
            ]
            prompt = f"""
            基于以下实现案例，生成实现簇的模式总结，包含3部分：
            1. 技术栈：使用的语言、库（如"Python + Gurobi"）
            2. 代码风格：变量/约束的定义方式（如"字典存储变量，列表推导式构造约束"）
            3. 适用规模：适合的问题复杂度（如"中小规模问题，变量数<5000"）
            
            案例列表：
            {chr(10).join(implementation_infos)}
            
            要求：简洁明了，控制在150字以内，避免具体案例细节。
            """

        # 调用LLM生成总结
        try:
            response = self.llm_controller.get_completion(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM生成簇总结失败，使用默认总结: {str(e)}")
            return f"默认{cluster_type}簇总结（{len(nodes)}个案例）"

    def _extract_tech_stack(self, code: str) -> str:
        """提取代码技术栈（简单规则匹配）"""
        tech_stack = []
        if "import gurobi" in code or "gurobipy" in code:
            tech_stack.append("Gurobi")
        if "import ortools" in code:
            tech_stack.append("OR-Tools")
        if "import pulp" in code:
            tech_stack.append("PuLP")
        return "Python + " + " + ".join(tech_stack) if tech_stack else "Python"

    def _extract_code_style(self, code: str) -> str:
        """提取代码风格（简单规则匹配）"""
        if "dict(" in code or "{}" in code and "x[" in code:
            return "字典存储变量"
        elif "class" in code and "def __init__" in code:
            return "面向对象风格"
        elif "for " in code and "in " in code and "[" in code:
            return "列表推导式构造约束"
        else:
            return "基础线性风格"

    # ------------------------------
    # 3.2 簇整合：更新簇中心与模式总结
    # ------------------------------
    def _consolidate_clusters(self) -> None:
        """整合所有簇：更新簇中心（基于所有典型案例）、重新生成模式总结"""
        logger.info("开始整合所有簇...")
        # 1. 整合建模簇
        for cluster_id, cluster in self.modeling_clusters.items():
            # 获取簇下所有典型案例的建模嵌入
            case_embeddings = []
            valid_cases = []
            for case_id in cluster.typical_cases:
                case = self.memory_nodes.get(case_id)
                if case and case.modeling_embedding is not None:
                    case_embeddings.append(case.modeling_embedding)
                    valid_cases.append(case)
            # 更新簇中心
            cluster.update_cluster_center(case_embeddings)
            # 重新生成模式总结（基于所有有效案例）
            if valid_cases:
                cluster.pattern_summary = self._generate_cluster_summary("modeling", valid_cases)
            # 标记案例为"已整合"
            for case_id in valid_cases:
                self.memory_nodes[case_id].status = "integrated"

        # 2. 整合实现簇
        for cluster_id, cluster in self.implementation_clusters.items():
            # 获取簇下所有典型案例的实现嵌入
            case_embeddings = []
            valid_cases = []
            for case_id in cluster.typical_cases:
                case = self.memory_nodes.get(case_id)
                if case and case.implementation_embedding is not None:
                    case_embeddings.append(case.implementation_embedding)
                    valid_cases.append(case)
            # 更新簇中心
            cluster.update_cluster_center(case_embeddings)
            # 重新生成模式总结（基于所有有效案例）
            if valid_cases:
                cluster.pattern_summary = self._generate_cluster_summary("implementation", valid_cases)
            # 标记案例为"已整合"
            for case_id in valid_cases:
                self.memory_nodes[case_id].status = "integrated"

        logger.info("簇整合完成")

    # ------------------------------
    # 3.3 推理引擎：双推理树构建与候选生成
    # ------------------------------
    def build_modeling_inference_tree(self, problem_desc: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """构建建模推理树：生成Top-K适合的建模簇及置信度（调用LLM）"""
        if not self.modeling_clusters:
            return []

        # 整理现有建模簇信息
        modeling_clusters_info = [
            f"建模簇ID：{cid}\n模式总结：{cluster.pattern_summary}"
            for cid, cluster in self.modeling_clusters.items()
        ]

        prompt = f"""
        任务：分析问题适合的建模方法，基于现有建模簇生成推理结果。
        输入：
        1. 问题描述：{problem_desc}
        2. 现有建模簇列表：
        {chr(10).join(modeling_clusters_info)}
        
        输出要求（严格JSON格式）：
        [
            {{
                "modeling_cluster_id": "建模簇ID",
                "reasoning": "为什么适合（结合问题与簇模式总结，1-2句话）",
                "variables": "关键变量定义（如'弧变量x[i,j]∈{0,1}'）",
                "constraints": ["核心约束1", "核心约束2"],
                "confidence": 0-1的浮点数（置信度，越高越适合）
            }}
        ]
        
        要求：
        1. 只从现有建模簇中选择，不新增簇
        2. 按置信度降序返回Top-{top_k}
        3. 置信度基于问题与簇的匹配度、簇的案例数量综合判断
        """

        try:
            response = self.llm_controller.get_completion(prompt, response_format={"type": "json"})
            inference_result = json.loads(response)
            # 截取Top-K并过滤无效簇ID
            valid_result = [
                item for item in inference_result
                if item["modeling_cluster_id"] in self.modeling_clusters
            ]
            return valid_result[:top_k]
        except Exception as e:
            logger.error(f"构建建模推理树失败: {str(e)}")
            return []

    def build_implementation_inference_tree(self, problem_desc: str, modeling_inference: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
        """构建实现推理树：结合问题与建模结果，生成Top-K适合的实现簇及置信度（调用LLM）"""
        if not self.implementation_clusters:
            return []

        # 整理现有实现簇信息
        implementation_clusters_info = [
            f"实现簇ID：{cid}\n模式总结：{cluster.pattern_summary}"
            for cid, cluster in self.implementation_clusters.items()
        ]

        # 整理建模推理结果（作为参考）
        modeling_info = [
            f"建模簇ID：{item['modeling_cluster_id']}\n核心约束：{item['constraints']}"
            for item in modeling_inference
        ]

        prompt = f"""
        任务：分析问题适合的代码实现方式，基于现有实现簇生成推理结果。
        输入：
        1. 问题描述：{problem_desc}
        2. 已选建模方法：
        {chr(10).join(modeling_info)}
        3. 现有实现簇列表：
        {chr(10).join(implementation_clusters_info)}
        
        输出要求（严格JSON格式）：
        [
            {{
                "implementation_cluster_id": "实现簇ID",
                "reasoning": "为什么适合（结合问题、建模方法与簇模式总结，1-2句话）",
                "tech_stack": "技术栈（如'Python + Gurobi'）",
                "applicable_scale": "适用规模（如'中小规模问题'）",
                "confidence": 0-1的浮点数（置信度，越高越适合）
            }}
        ]
        
        要求：
        1. 只从现有实现簇中选择，不新增簇
        2. 按置信度降序返回Top-{top_k}
        3. 置信度基于问题规模、建模方法与实现簇的匹配度综合判断
        """

        try:
            response = self.llm_controller.get_completion(prompt, response_format={"type": "json"})
            inference_result = json.loads(response)
            # 截取Top-K并过滤无效簇ID
            valid_result = [
                item for item in inference_result
                if item["implementation_cluster_id"] in self.implementation_clusters
            ]
            return valid_result[:top_k]
        except Exception as e:
            logger.error(f"构建实现推理树失败: {str(e)}")
            return []

    def generate_candidate_solutions(self, modeling_inference: List[Dict[str, Any]], implementation_inference: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成候选方案矩阵：建模推理Top-K × 实现推理Top-K，每个交叉点取最优案例"""
        candidates = []
        for modeling_item in modeling_inference:
            modeling_cid = modeling_item["modeling_cluster_id"]
            modeling_conf = modeling_item["confidence"]

            for implementation_item in implementation_inference:
                implementation_cid = implementation_item["implementation_cluster_id"]
                implementation_conf = implementation_item["confidence"]

                # 从交叉索引中获取该（建模簇，实现簇）对应的所有案例
                cross_key = (modeling_cid, implementation_cid)
                case_ids = self.cross_index.get(cross_key, [])
                if not case_ids:
                    continue

                # 选择最优案例（按创建时间倒序，取最新的1个）
                case_ids_sorted = sorted(case_ids, key=lambda x: self.memory_nodes[x].timestamp, reverse=True)
                best_case_id = case_ids_sorted[0]
                best_case = self.memory_nodes[best_case_id]

                # 计算综合置信度（建模置信度 × 实现置信度）
                combined_conf = round(modeling_conf * implementation_conf, 3)

                # 构造候选方案
                candidates.append({
                    "candidate_id": f"CAN-{str(uuid.uuid4())[:8]}",
                    "modeling_cluster": {
                        "id": modeling_cid,
                        "confidence": modeling_conf,
                        "summary": self.modeling_clusters[modeling_cid].pattern_summary
                    },
                    "implementation_cluster": {
                        "id": implementation_cid,
                        "confidence": implementation_conf,
                        "summary": self.implementation_clusters[implementation_cid].pattern_summary
                    },
                    "best_case": {
                        "id": best_case_id,
                        "problem": best_case.problem_description[:50] + "...",
                        "full_code": best_case.full_code
                    },
                    "combined_confidence": combined_conf
                })

        # 按综合置信度降序排序
        candidates.sort(key=lambda x: x["combined_confidence"], reverse=True)
        return candidates

    # ------------------------------
    # 3.4 候选验证与最优方案选择
    # ------------------------------
    def validate_candidates(self, candidates: List[Dict[str, Any]], test_case: str, timeout: int = 30) -> List[Dict[str, Any]]:
        """验证候选方案：运行测试用例，筛选可行方案"""
        validated_candidates = []
        for candidate in candidates:
            code = candidate["best_case"]["full_code"]
            # 1. 语法检查
            syntax_valid, syntax_msg = CodeValidator.check_syntax(code)
            if not syntax_valid:
                validated_candidates.append({
                    **candidate,
                    "validation_result": "failed",
                    "validation_msg": syntax_msg,
                    "run_time": 0.0
                })
                continue

            # 2. 运行验证
            run_valid, run_msg, run_time = CodeValidator.run_validation(code, test_case, timeout)
            validation_result = "success" if run_valid else "failed"
            validated_candidates.append({
                **candidate,
                "validation_result": validation_result,
                "validation_msg": run_msg,
                "run_time": round(run_time, 2)
            })

        return validated_candidates

    def select_best_solution(self, validated_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """选择最优方案：优先选验证通过的，再按综合置信度降序取第1个"""
        # 筛选验证通过的候选
        success_candidates = [
            cand for cand in validated_candidates
            if cand["validation_result"] == "success"
        ]
        if not success_candidates:
            logger.warning("所有候选方案均验证失败")
            return None

        # 按综合置信度降序，取第1个
        best_candidate = success_candidates[0]
        logger.info(f"选择最优方案，候选ID: {best_candidate['candidate_id']}，综合置信度: {best_candidate['combined_confidence']}")
        return best_candidate

    # ------------------------------
    # 3.5 完整工作流程：问题→推理→候选→验证→返回
    # ------------------------------
    def solve_problem(self, problem_desc: str, test_case: str, top_k_modeling: int = 2, top_k_implementation: int = 2, validation_timeout: int = 30) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """完整问题求解流程：推理树构建→候选生成→验证→最优选择"""
        logger.info(f"开始处理问题：{problem_desc[:50]}...")
        start_time = time.time()

        # 步骤1：构建建模推理树（Top-K建模方法）
        logger.info("步骤1/5：构建建模推理树")
        modeling_inference = self.build_modeling_inference_tree(problem_desc, top_k_modeling)
        if not modeling_inference:
            logger.error("未生成有效的建模推理结果，流程终止")
            return None, []
        logger.info(f"建模推理结果（Top-{top_k_modeling}）：{[item['modeling_cluster_id'] for item in modeling_inference]}")

        # 步骤2：构建实现推理树（Top-K实现方式）
        logger.info("步骤2/5：构建实现推理树")
        implementation_inference = self.build_implementation_inference_tree(problem_desc, modeling_inference, top_k_implementation)
        if not implementation_inference:
            logger.error("未生成有效的实现推理结果，流程终止")
            return None, []
        logger.info(f"实现推理结果（Top-{top_k_implementation}）：{[item['implementation_cluster_id'] for item in implementation_inference]}")

        # 步骤3：生成候选方案矩阵（建模×实现）
        logger.info("步骤3/5：生成候选方案矩阵")
        candidates = self.generate_candidate_solutions(modeling_inference, implementation_inference)
        if not candidates:
            logger.error("未生成任何候选方案，流程终止")
            return None, []
        logger.info(f"生成候选方案数量：{len(candidates)}")

        # 步骤4：并行验证候选方案
        logger.info("步骤4/5：验证候选方案")
        validated_candidates = self.validate_candidates(candidates, test_case, validation_timeout)
        # 统计验证结果
        success_count = sum(1 for cand in validated_candidates if cand["validation_result"] == "success")
        logger.info(f"候选方案验证完成，成功{success_count}个，失败{len(validated_candidates)-success_count}个")

        # 步骤5：选择最优方案
        logger.info("步骤5/5：选择最优方案")
        best_solution = self.select_best_solution(validated_candidates)

        # 输出总耗时
        total_time = round(time.time() - start_time, 2)
        logger.info(f"问题处理完成，总耗时：{total_time}秒")

        return best_solution, validated_candidates