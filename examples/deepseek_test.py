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
        
        # 创建告警分析器实例
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
        print("   🔄 正在调用 DeepSeek 模型进行告警分析...")
        
        # 实际调用 DSPy 模块
        analysis_result = alert_analyzer.forward(
            alert_info=test_alert,
            historical_alerts=[]
        )
        
        print("   ✅ 告警分析完成！")
        print(f"   - 分类: {analysis_result.category}")
        print(f"   - 优先级: {analysis_result.priority}")
        print(f"   - 紧急程度: {analysis_result.urgency_score}")
        print(f"   - 根因提示: {analysis_result.root_cause_hints[:100]}...")
        print(f"   - 建议操作: {analysis_result.recommended_actions[:100]}...")
        
        # 测试诊断智能体
        print("\n2. 测试诊断智能体")
        from src.dspy_modules.diagnostic_agent import DiagnosticAgent, DiagnosticContext
        
        diagnostic_agent = DiagnosticAgent()
        diagnostic_context = DiagnosticContext(
            alert_analysis=analysis_result,
            system_metrics={"cpu_usage": 0.95, "memory_usage": 0.65},
            log_entries=["2024-01-01 12:00:00 ERROR: High CPU usage detected"],
            historical_incidents=[],
            topology_info={"services": ["web", "api", "db"]}
        )
        
        print("   🔄 正在进行根因诊断...")
        diagnostic_result = diagnostic_agent.forward(diagnostic_context)
        
        print("   ✅ 诊断完成！")
        print(f"   - 根因: {diagnostic_result.root_cause}")
        print(f"   - 置信度: {diagnostic_result.confidence_score}")
        print(f"   - 影响评估: {diagnostic_result.impact_assessment}")
        print(f"   - 恢复时间估计: {diagnostic_result.recovery_time_estimate}")
        
        # 测试行动规划器
        print("\n3. 测试行动规划器")
        from src.dspy_modules.action_planner import ActionPlanner
        
        action_planner = ActionPlanner()
        
        print("   🔄 正在生成行动计划...")
        action_plan = action_planner.forward(
            diagnostic_result=diagnostic_result,
            system_context={"environment": "production", "cluster": "web-cluster"}
        )
        
        print("   ✅ 行动计划生成完成！")
        print(f"   - 计划ID: {action_plan.plan_id}")
        print(f"   - 优先级: {action_plan.priority}")
        print(f"   - 步骤数量: {len(action_plan.steps)}")
        print(f"   - 预估时间: {action_plan.estimated_duration}分钟")
        
        if action_plan.steps:
            print("   - 主要步骤:")
            for i, step in enumerate(action_plan.steps[:3], 1):
                print(f"     {i}. {step.description}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DSPy 模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
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
        
        # 测试完整的工作流
        print("\n1. 测试 LangGraph 工作流")
        from src.langgraph_workflow.ops_workflow import OpsWorkflow
        from src.langgraph_workflow.state_manager import StateManager
        
        # 创建工作流实例
        workflow = OpsWorkflow()
        state_manager = StateManager()
        
        # 创建初始状态
        initial_state = state_manager.initialize_state("deepseek_test_workflow")
        
        # 添加测试告警
        test_alert = AlertInfo(
            alert_id="deepseek_workflow_test",
            timestamp="2024-01-01T12:00:00Z",
            severity="critical",
            source="deepseek_monitor",
            message="DeepSeek API 响应时间异常，影响生产环境",
            metrics={
                "response_time": 8000,
                "success_rate": 0.75,
                "error_rate": 0.25,
                "cpu_usage": 0.90
            },
            tags=["api", "performance", "deepseek", "critical"]
        )
        
        # 更新状态，添加告警
        initial_state = state_manager.update_state(initial_state, {
            "current_alert": test_alert
        })
        
        print(f"   告警: {test_alert.message}")
        print("   🔄 正在运行完整的智能运维工作流...")
        
        # 运行工作流
        step_count = 0
        results = {}
        
        async for step in workflow.stream_run(initial_state, max_iterations=10):
            step_count += 1
            step_name = list(step.keys())[0] if step else "unknown"
            print(f"   📊 步骤 {step_count}: {step_name}")
            
            # 保存结果
            if step:
                results.update(step)
            
            # 限制步骤数量
            if step_count >= 8:
                break
        
        print("   ✅ 工作流执行完成！")
        
        # 显示关键结果
        final_state = list(results.values())[-1] if results else initial_state
        
        if final_state.get("alert_analysis"):
            analysis = final_state["alert_analysis"]
            print(f"   - 告警分析: {analysis.category} (优先级: {analysis.priority})")
        
        if final_state.get("diagnostic_result"):
            diagnosis = final_state["diagnostic_result"]
            print(f"   - 诊断结果: {diagnosis.root_cause}")
            print(f"   - 置信度: {diagnosis.confidence_score}")
        
        if final_state.get("action_plan"):
            plan = final_state["action_plan"]
            print(f"   - 行动计划: {len(plan.steps)}个步骤")
            print(f"   - 预估时间: {plan.estimated_duration}分钟")
        
        if final_state.get("execution_result"):
            exec_result = final_state["execution_result"]
            print(f"   - 执行结果: {exec_result.status}")
        
        if final_state.get("incident_report"):
            report = final_state["incident_report"]
            print(f"   - 事件报告: {report.incident_id}")
        
        # 测试高级智能体功能
        print("\n2. 测试高级智能体功能")
        agent_config = AgentConfig(
            agent_id="deepseek_advanced_agent",
            agent_type="general",
            specialization="deepseek_powered",
            enable_learning=True,
            enable_reporting=True,
            auto_execution=False
        )
        
        agent = IntelligentOpsAgent(agent_config)
        print(f"   ✅ 高级智能体创建完成: {agent_config.agent_id}")
        
        # 测试告警处理
        alert_dict = {
            "alert_id": "deepseek_advanced_test",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "high",
            "source": "advanced_monitor",
            "message": "内存使用率持续上升，可能存在内存泄漏",
            "metrics": {
                "memory_usage": 0.92,
                "memory_growth_rate": 0.05,
                "gc_frequency": 500
            },
            "tags": ["memory", "leak", "performance"]
        }
        
        print(f"   告警: {alert_dict['message']}")
        print("   🔄 正在处理告警...")
        
        # 使用智能体处理告警
        result = await agent.process_alert(alert_dict)
        print(f"   ✅ 告警处理: {result['status']}")
        
        # 获取性能指标
        metrics = agent.get_performance_metrics()
        print(f"   📊 处理事件数: {metrics['incidents_processed']}")
        print(f"   📊 成功率: {metrics['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 智能体测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
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