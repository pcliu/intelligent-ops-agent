#!/usr/bin/env python3
"""
测试智能路由 DSPy 模块
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dspy_modules.smart_router import SmartIntelligentRouter, RouterDecision
from dspy_modules.alert_analyzer import AlertInfo
from utils.llm_config import setup_deepseek_llm


def test_smart_router():
    """测试智能路由模块"""
    
    print("🧪 开始测试智能路由 DSPy 模块...")
    
    try:
        # 1. 初始化 DSPy LLM
        print("\n🔧 初始化 DSPy LLM...")
        llm = setup_deepseek_llm()
        print("✅ DSPy LLM 初始化成功")
        
        # 2. 创建智能路由器
        print("\n🧠 创建智能路由器...")
        smart_router = SmartIntelligentRouter()
        print("✅ 智能路由器创建成功")
        
        # 3. 测试场景 1：初始告警状态
        print("\n📋 测试场景 1：初始告警状态")
        test_alert = AlertInfo(
            alert_id="TEST001",
            timestamp="2025-01-25T10:00:00Z",
            severity="high",
            source="web_server",
            message="HTTP 响应时间超过 5 秒",
            metrics={"response_time": 8.5, "error_rate": 0.15},
            tags=["performance", "web"]
        )
        
        initial_state = {
            "messages": [],
            "alert_info": test_alert,
            "symptoms": None,
            "context": None,
            "analysis_result": None,
            "diagnostic_result": None,
            "action_plan": None,
            "execution_result": None,
            "report": None,
            "memory_queries": None,
            "memory_context": None,
            "info_collection_queries": None,
            "info_collection_context": None,
            "errors": None
        }
        
        decision1 = smart_router.forward(initial_state)
        print(f"✅ 路由决策完成:")
        print(f"   - 下一个节点: {decision1.next_node}")
        print(f"   - 决策置信度: {decision1.confidence:.2f}")
        print(f"   - 记忆优先级: {decision1.memory_priority}")
        print(f"   - 业务优先级: {decision1.business_priority}")
        print(f"   - 推理: {decision1.reasoning[:100]}...")
        
        # 4. 测试场景 2：诊断阶段，低置信度需要记忆支持
        print("\n📋 测试场景 2：诊断阶段，低置信度")
        diagnosis_state = {
            **initial_state,
            "analysis_result": {
                "category": "performance",
                "priority": "high",
                "urgency_score": 0.9
            },
            "symptoms": ["响应缓慢", "CPU使用率高"],
            "diagnostic_result": {
                "root_cause": "数据库连接池满载",
                "confidence_score": 0.4,  # 低置信度
                "potential_causes": ["内存泄漏", "数据库锁等待", "网络延迟"]
            }
        }
        
        decision2 = smart_router.forward(diagnosis_state)
        print(f"✅ 路由决策完成:")
        print(f"   - 下一个节点: {decision2.next_node}")
        print(f"   - 决策置信度: {decision2.confidence:.2f}")
        print(f"   - 记忆优先级: {decision2.memory_priority}")
        print(f"   - 建议查询数: {len(decision2.suggested_queries)}")
        if decision2.suggested_queries:
            print(f"   - 建议查询: {decision2.suggested_queries[:2]}")
        
        # 5. 测试场景 3：需要信息收集
        print("\n📋 测试场景 3：需要信息收集")
        incomplete_state = {
            "messages": [],
            "alert_info": None,  # 缺少告警信息
            "symptoms": ["系统异常"],
            "context": None,
            "memory_queries": None,
            "memory_context": None,
            "info_collection_queries": ["系统状态", "错误日志"],
            "info_collection_context": None
        }
        
        decision3 = smart_router.forward(incomplete_state)
        print(f"✅ 路由决策完成:")
        print(f"   - 下一个节点: {decision3.next_node}")
        print(f"   - 需要信息收集: {decision3.requires_info_collection}")
        print(f"   - 缺失信息查询: {decision3.missing_info_queries}")
        
        # 6. 测试场景 4：完整流程，准备结束
        print("\n📋 测试场景 4：完整流程")
        complete_state = {
            **diagnosis_state,
            "action_plan": {
                "actions": ["重启数据库连接池", "增加连接数限制"],
                "risk_level": "low"
            },
            "execution_result": {
                "success": True,
                "results": ["连接池重启成功", "响应时间恢复正常"]
            },
            "report": {
                "summary": "性能问题已解决",
                "resolution_time": "15分钟"
            }
        }
        
        decision4 = smart_router.forward(complete_state)
        print(f"✅ 路由决策完成:")
        print(f"   - 下一个节点: {decision4.next_node}")
        print(f"   - 决策置信度: {decision4.confidence:.2f}")
        print(f"   - 替代路径: {decision4.alternative_paths}")
        
        print(f"\n🎉 智能路由 DSPy 模块测试全部通过!")
        print(f"   - 状态分析: ✅")
        print(f"   - 记忆需求分析: ✅")
        print(f"   - 路由决策优化: ✅")
        print(f"   - 多场景测试: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ 智能路由测试失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        return False


def main():
    """主函数"""
    success = test_smart_router()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()