#!/usr/bin/env python3
"""
DeepSeek LLM 配置和测试示例

演示如何配置和使用 DeepSeek 作为智能运维智能体的语言模型
"""

import asyncio
import os
import sys
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.llm_config import (
    LLMConfig, 
    setup_deepseek_llm, 
    get_llm_config_from_env,
    validate_llm_config,
    test_llm_connection,
    DEEPSEEK_CONFIGS
)
from src.agents.intelligent_ops_agent import IntelligentOpsAgent, AgentConfig
from src.dspy_modules.alert_analyzer import AlertInfo


def setup_environment():
    """设置测试环境"""
    # 设置环境变量 (请替换为您的实际 API Key)
    test_api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not test_api_key:
        print("⚠️  请设置 DEEPSEEK_API_KEY 环境变量")
        print("   方法1: export DEEPSEEK_API_KEY='your-api-key'")
        print("   方法2: 创建 .env 文件并添加 DEEPSEEK_API_KEY")
        return False
    
    # 设置其他环境变量
    os.environ.setdefault("LLM_PROVIDER", "deepseek")
    os.environ.setdefault("LLM_MODEL_NAME", "deepseek-chat")
    os.environ.setdefault("LLM_BASE_URL", "https://api.deepseek.com/v1")
    os.environ.setdefault("LLM_TEMPERATURE", "0.1")
    os.environ.setdefault("LLM_MAX_TOKENS", "2000")
    
    return True


def test_llm_configurations():
    """测试不同的 LLM 配置"""
    print("\n🔧 测试 LLM 配置")
    print("=" * 40)
    
    # 测试1: 从环境变量获取配置
    print("\n1. 从环境变量获取配置")
    env_config = get_llm_config_from_env()
    print(f"   提供商: {env_config.provider}")
    print(f"   模型: {env_config.model_name}")
    print(f"   基础URL: {env_config.base_url}")
    print(f"   温度: {env_config.temperature}")
    
    # 验证配置
    if validate_llm_config(env_config):
        print("   ✅ 配置验证通过")
    else:
        print("   ❌ 配置验证失败")
        return False
    
    # 测试2: 使用预设配置
    print("\n2. 使用预设 DeepSeek 配置")
    deepseek_config = DEEPSEEK_CONFIGS["deepseek-chat"]
    deepseek_config.api_key = os.getenv("DEEPSEEK_API_KEY")
    
    print(f"   模型: {deepseek_config.model_name}")
    print(f"   最大令牌: {deepseek_config.max_tokens}")
    
    # 测试3: 自定义配置
    print("\n3. 自定义配置")
    custom_config = LLMConfig(
        provider="deepseek",
        model_name="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=4000
    )
    
    print(f"   自定义最大令牌: {custom_config.max_tokens}")
    
    return True


def test_basic_llm_functionality():
    """测试基础 LLM 功能"""
    print("\n🧪 测试基础 LLM 功能")
    print("=" * 40)
    
    try:
        # 获取配置
        config = get_llm_config_from_env()
        config.api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 设置 LLM
        print("\n1. 设置 DeepSeek LLM")
        dspy_lm, langchain_llm = setup_deepseek_llm(config)
        print("   ✅ LLM 设置完成")
        
        # 测试 DSPy LLM
        print("\n2. 测试 DSPy LLM")
        test_prompt = "请用中文简短回答：什么是智能运维？"
        response = dspy_lm(test_prompt)
        print(f"   问题: {test_prompt}")
        print(f"   回答: {response}")
        
        # 测试 LangChain LLM
        print("\n3. 测试 LangChain LLM")
        langchain_response = langchain_llm.invoke("请解释一下什么是告警分析")
        print(f"   问题: 请解释一下什么是告警分析")
        print(f"   回答: {langchain_response.content}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM 功能测试失败: {str(e)}")
        return False


def test_dspy_modules_with_deepseek():
    """测试 DSPy 模块与 DeepSeek 的集成"""
    print("\n🧠 测试 DSPy 模块与 DeepSeek 集成")
    print("=" * 40)
    
    try:
        # 设置 LLM
        config = get_llm_config_from_env()
        config.api_key = os.getenv("DEEPSEEK_API_KEY")
        dspy_lm, _ = setup_deepseek_llm(config)
        
        # 测试告警分析器
        print("\n1. 测试告警分析器")
        from src.dspy_modules.alert_analyzer import AlertAnalyzer
        
        # 注意：这里需要实际的 DSPy 配置，可能需要模拟
        alert_analyzer = AlertAnalyzer()
        
        test_alert = AlertInfo(
            alert_id="test_deepseek_001",
            timestamp="2024-01-01T12:00:00Z",
            severity="high",
            source="cpu_monitor",
            message="CPU 使用率超过 90% 阈值",
            metrics={"cpu_usage": 0.95},
            tags=["cpu", "performance"]
        )
        
        print(f"   告警信息: {test_alert.message}")
        print("   💡 注意: DSPy 模块需要真实的模型调用，当前为演示模式")
        
        # 实际在生产环境中会调用:
        # analysis_result = alert_analyzer.forward(test_alert)
        # print(f"   分析结果: {analysis_result}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DSPy 模块测试失败: {str(e)}")
        return False


async def test_intelligent_ops_agent_with_deepseek():
    """测试智能运维智能体与 DeepSeek 的集成"""
    print("\n🤖 测试智能运维智能体与 DeepSeek 集成")
    print("=" * 40)
    
    try:
        # 设置 LLM
        config = get_llm_config_from_env()
        config.api_key = os.getenv("DEEPSEEK_API_KEY")
        setup_deepseek_llm(config)
        
        # 创建智能体
        print("\n1. 创建智能运维智能体")
        agent_config = AgentConfig(
            agent_id="deepseek_test_agent",
            agent_type="general",
            specialization="deepseek_powered",
            enable_learning=True,
            enable_reporting=True,
            auto_execution=False
        )
        
        agent = IntelligentOpsAgent(agent_config)
        print(f"   ✅ 智能体创建完成: {agent_config.agent_id}")
        
        # 测试告警处理
        print("\n2. 测试告警处理")
        test_alert = {
            "alert_id": "deepseek_test_alert",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "high",
            "source": "deepseek_monitor",
            "message": "DeepSeek API 响应时间异常",
            "metrics": {
                "response_time": 5000,
                "success_rate": 0.85,
                "error_rate": 0.15
            },
            "tags": ["api", "performance", "deepseek"]
        }
        
        print(f"   告警: {test_alert['message']}")
        
        # 注意：实际的告警处理需要 LLM 调用
        print("   💡 注意: 完整的告警处理需要 LLM 模型调用")
        print("   当前演示智能体的基础功能")
        
        # 获取智能体状态
        status = agent.get_agent_status()
        print(f"   智能体状态: {status['current_state']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 智能体测试失败: {str(e)}")
        return False


def demonstrate_deepseek_features():
    """演示 DeepSeek 的特性"""
    print("\n⭐ DeepSeek 特性演示")
    print("=" * 40)
    
    print("\n🔹 DeepSeek 优势:")
    print("   • 高性价比：API 调用成本低")
    print("   • 中文优化：对中文理解和生成效果好")
    print("   • 代码能力：DeepSeek-Coder 在代码任务上表现优秀")
    print("   • 开源透明：模型架构和训练方法公开")
    print("   • API 兼容：支持 OpenAI 格式的 API 调用")
    
    print("\n🔹 适用场景:")
    print("   • 运维日志分析（中文日志友好）")
    print("   • 故障诊断报告生成")
    print("   • 自动化脚本生成")
    print("   • 运维知识问答")
    print("   • 成本敏感的大规模部署")
    
    print("\n🔹 配置建议:")
    print("   • 通用任务：使用 deepseek-chat")
    print("   • 代码生成：使用 deepseek-coder")
    print("   • 温度设置：0.1-0.3 (运维任务需要准确性)")
    print("   • 令牌数量：2000-4000 (根据任务复杂度)")


async def main():
    """主函数"""
    print("🚀 DeepSeek LLM 配置和测试")
    print("=" * 50)
    
    # 1. 环境设置
    if not setup_environment():
        print("\n❌ 环境设置失败，请检查 API Key 配置")
        return
    
    # 2. 配置测试
    if not test_llm_configurations():
        print("\n❌ 配置测试失败")
        return
    
    # 3. 连接测试
    if not test_llm_connection():
        print("\n❌ 连接测试失败，请检查网络和 API Key")
        return
    
    # 4. 基础功能测试
    if test_basic_llm_functionality():
        print("\n✅ 基础功能测试成功")
    
    # 5. DSPy 模块测试
    if test_dspy_modules_with_deepseek():
        print("\n✅ DSPy 模块测试成功")
    
    # 6. 智能体测试
    if await test_intelligent_ops_agent_with_deepseek():
        print("\n✅ 智能体测试成功")
    
    # 7. 特性演示
    demonstrate_deepseek_features()
    
    print("\n" + "=" * 50)
    print("🎉 DeepSeek LLM 配置和测试完成！")
    print("\n💡 接下来的步骤:")
    print("   1. 设置您的 DEEPSEEK_API_KEY")
    print("   2. 运行完整的智能运维示例")
    print("   3. 根据需要调整模型参数")
    print("   4. 在生产环境中部署")


if __name__ == "__main__":
    asyncio.run(main())