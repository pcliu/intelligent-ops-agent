#!/usr/bin/env python3
"""
测试智能路由模块基础功能（不调用 LLM）
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dspy_modules.smart_router import SmartIntelligentRouter, RouterDecision
from dspy_modules.alert_analyzer import AlertInfo


def test_router_basic():
    """测试路由器基础功能"""
    
    print("🧪 开始测试智能路由器基础功能...")
    
    try:
        # 1. 创建智能路由器
        print("\n🧠 创建智能路由器...")
        smart_router = SmartIntelligentRouter()
        print("✅ 智能路由器创建成功")
        
        # 2. 测试状态提取方法
        print("\n📊 测试状态提取方法...")
        test_alert = AlertInfo(
            alert_id="TEST001",
            timestamp="2025-01-25T10:00:00Z",
            severity="high",
            source="web_server",
            message="HTTP 响应时间超过 5 秒",
            metrics={"response_time": 8.5},
            tags=["performance"]
        )
        
        test_state = {
            "alert_info": test_alert,
            "symptoms": ["响应缓慢"],
            "analysis_result": {"category": "performance", "priority": "high"},
            "diagnostic_result": {
                "root_cause": "数据库连接池满载",
                "confidence_score": 0.4
            },
            "errors": ["网络超时"]
        }
        
        # 测试业务数据提取
        business_data = smart_router._extract_business_data(test_state)
        print(f"✅ 业务数据提取: {business_data[:100]}...")
        
        # 测试完成状态检查
        completion_status = smart_router._check_completion_status(test_state)
        print(f"✅ 完成状态检查: {completion_status}")
        
        # 测试错误状态检查
        error_status = smart_router._check_error_status(test_state)
        print(f"✅ 错误状态检查: {error_status}")
        
        # 测试状态序列化
        serialized_state = smart_router._serialize_complete_state(test_state)
        print(f"✅ 状态序列化: {serialized_state[:100]}...")
        
        # 3. 测试信息收集需求检查
        print("\n📝 测试信息收集需求检查...")
        
        # 缺少告警信息的状态
        incomplete_state = {"symptoms": ["异常"]}
        needs_info = smart_router._check_info_collection_needs(incomplete_state)
        print(f"✅ 缺少告警信息需要收集: {needs_info}")
        
        # 完整状态
        complete_state = {"alert_info": test_alert, "symptoms": ["异常"], "context": {"env": "prod"}}
        needs_info = smart_router._check_info_collection_needs(complete_state)
        print(f"✅ 完整状态需要收集: {needs_info}")
        
        # 4. 测试回退决策
        print("\n🔄 测试回退决策...")
        fallback_decision = smart_router._create_fallback_decision(test_state, "测试错误")
        print(f"✅ 回退决策:")
        print(f"   - 下一个节点: {fallback_decision.next_node}")
        print(f"   - 推理: {fallback_decision.reasoning}")
        print(f"   - 置信度: {fallback_decision.confidence}")
        
        # 测试不同状态的回退决策
        empty_state = {}
        fallback_empty = smart_router._create_fallback_decision(empty_state, "空状态")
        print(f"✅ 空状态回退: {fallback_empty.next_node}")
        
        memory_query_state = {"memory_queries": ["test query"]}
        fallback_memory = smart_router._create_fallback_decision(memory_query_state, "记忆查询")
        print(f"✅ 记忆查询状态回退: {fallback_memory.next_node}")
        
        # 5. 测试节点优先级映射
        print("\n🏆 测试节点优先级...")
        print("✅ 节点优先级映射:")
        for node, priority in smart_router.node_priority_map.items():
            print(f"   - {node}: {priority}")
        
        print(f"\n🎉 智能路由器基础功能测试全部通过!")
        print(f"   - 状态提取方法: ✅")
        print(f"   - 信息收集检查: ✅") 
        print(f"   - 回退决策: ✅")
        print(f"   - 配置验证: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ 路由器基础功能测试失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        return False


def main():
    """主函数"""
    success = test_router_basic()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()