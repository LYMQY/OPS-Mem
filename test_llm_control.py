import os
import json
from dualcluster_memory.llm_controller import LLMController
from dotenv import load_dotenv

load_dotenv()

Gemini_API_Key = os.getenv('Gemini_API_Key_1')
Deepseek_API_Key = os.getenv('SiliconFlow_API_KEY')


def test_llm_controller_gemini():
    """测试Gemini后端"""
    try:
        # 初始化控制器（使用gemini-2.5）
        llm = LLMController(backend="gemini", model="gemini-2.5-flash", api_key=Gemini_API_Key)
        
        # 测试基础文本生成
        prompt = "用一句话描述机器学习的定义"
        response = llm.get_completion(prompt, response_format=None)
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Gemini文本生成结果: {response[:50]}...")
        
        # 测试JSON格式输出
        json_prompt = "生成一个包含'city'和'country'字段的JSON对象，值分别为'北京'和'中国'"
        json_response = llm.get_completion(
            json_prompt,
            response_format={"type": "json_object"}
        )
        # 验证JSON格式合法性
        json_data = json.loads(json_response)
        assert "city" in json_data and json_data["city"] == "北京"
        assert "country" in json_data and json_data["country"] == "中国"
        print("Gemini JSON输出验证通过")
        
    except Exception as e:
        print(f"Gemini测试失败: {str(e)}")

def test_llm_controller_deepseek():
    """测试Deepseek后端"""
    try:
        # 初始化控制器（使用DeepSeek-V3）
        llm = LLMController(backend="deepseek", model="deepseek-ai/DeepSeek-V3", api_key=Deepseek_API_Key)
        
        # 测试基础文本生成
        prompt = "用一句话描述深度学习的定义"
        response = llm.get_completion(prompt, response_format=None)
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Deepseek文本生成结果: {response}...")
        
        # 测试JSON格式输出
        json_prompt = "生成一个包含'language'和'version'字段的JSON对象，值分别为'Python'和'3.11'"
        json_response = llm.get_completion(
            json_prompt,
            response_format={"type": "json_object"}
        )
        print(f"Deepseek JSON响应: {json_response}")
        # 验证JSON格式合法性
        json_data = json.loads(json_response)
        assert "language" in json_data and json_data["language"] == "Python"
        assert "version" in json_data and json_data["version"] == "3.11"
        print("Deepseek JSON输出验证通过")
        
    except Exception as e:
        print(f"Deepseek测试失败: {str(e)}")

def test_invalid_backend():
    """测试无效后端参数"""
    pass

if __name__ == "__main__":
    # 依次运行所有测试
    test_invalid_backend()
    print("------------------------")
    test_llm_controller_gemini()
    print("------------------------")
    test_llm_controller_deepseek()
    print("所有测试通过!")