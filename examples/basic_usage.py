#!/usr/bin/env python3
"""
智能运维智能体基础使用示例

本示例展示如何使用结合了 LangGraph 和 DSPy 的智能运维系统：
1. 初始化智能体
2. 处理监控告警
3. 执行自动化运维
4. 生成运维报告
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.langgraph_workflow.ops_workflow import OpsWorkflow
from src.dspy_modules.alert_analyzer import AlertInfo
from src.langgraph_workflow.state_manager import StateManager


async def main():
    """主要演示函数"""
    print("🚀 智能运维智能体演示开始")
    print("=" * 50)
    
    # 1. 初始化智能运维工作流
    print("\n📋 1. 初始化智能运维工作流")
    ops_workflow = OpsWorkflow()
    
    # 验证工作流配置
    validation_errors = ops_workflow.validate_workflow()
    if validation_errors:
        print(f"❌ 工作流验证失败: {validation_errors}")
        return
    
    print("✅ 工作流初始化成功")
    
    # 编译工作流图
    compiled_graph = ops_workflow.compile()
    print("✅ 工作流编译完成")
    
    # 2. 创建模拟告警
    print("\n🚨 2. 创建模拟告警")
    alert_info = AlertInfo(
        alert_id="alert_20240101_120000",
        timestamp=datetime.now().isoformat(),
        severity="high",
        source="monitoring_system",
        message="CPU usage exceeds 80% threshold on web-server-01",
        metrics={
            "cpu_usage": 0.85,
            "memory_usage": 0.72,
            "disk_usage": 0.45
        },
        tags=["cpu", "performance", "web-server-01"]
    )
    
    print(f"告警ID: {alert_info.alert_id}")
    print(f"严重程度: {alert_info.severity}")
    print(f"告警消息: {alert_info.message}")
    
    # 3. 初始化工作流状态
    print("\n🔄 3. 初始化工作流状态")
    state_manager = StateManager()
    initial_state = state_manager.initialize_state("demo_workflow_001")
    
    # 将告警添加到状态中
    initial_state = state_manager.update_state(initial_state, {
        "current_alert": alert_info,
        "system_metrics": {
            "cpu_usage": 0.85,
            "memory_usage": 0.72,
            "disk_usage": 0.45,
            "network_latency": 120
        },
        "system_context": {
            "environment": "production",
            "region": "us-east-1",
            "cluster": "web-cluster-01"
        }
    })
    
    print(f"工作流ID: {initial_state['workflow_id']}")
    print(f"初始阶段: {initial_state['workflow_stage']}")
    
    # 4. 运行智能运维工作流
    print("\n🔄 4. 运行智能运维工作流")
    print("开始自动化运维流程...")
    
    try:
        # 使用流式运行以观察进度
        async for step in ops_workflow.stream_run(initial_state, max_iterations=50):
            for node_name, node_output in step.items():
                print(f"📍 节点 '{node_name}' 执行完成")
                print(f"   阶段: {node_output.get('workflow_stage', 'N/A')}")
                print(f"   状态: {node_output.get('workflow_status', 'N/A')}")
                
                # 显示关键信息
                if node_output.get('alert_analysis'):
                    analysis = node_output['alert_analysis']
                    print(f"   📊 告警分析: 优先级={analysis.priority}, 分类={analysis.category}")
                
                if node_output.get('diagnostic_result'):
                    diagnosis = node_output['diagnostic_result']
                    print(f"   🔍 诊断结果: {diagnosis.root_cause}")
                
                if node_output.get('action_plan'):
                    plan = node_output['action_plan']
                    print(f"   📋 行动计划: {len(plan.steps)} 个步骤")
                
                if node_output.get('execution_result'):
                    execution = node_output['execution_result']
                    print(f"   ⚡ 执行结果: {execution.status}")
                
                if node_output.get('incident_report'):
                    report = node_output['incident_report']
                    print(f"   📄 事件报告: {report.title}")
                
                # 检查是否有错误
                if node_output.get('errors'):
                    print(f"   ⚠️  错误: {len(node_output['errors'])} 个")
                
                print()
        
        print("✅ 工作流执行完成")
        
    except Exception as e:
        print(f"❌ 工作流执行失败: {str(e)}")
        return
    
    # 5. 获取工作流指标
    print("\n📊 5. 工作流执行指标")
    metrics = ops_workflow.get_workflow_metrics("demo_workflow_001")
    
    print(f"执行时间: {metrics.get('execution_time_seconds', 0):.2f} 秒")
    print(f"最终状态: {metrics.get('status', 'N/A')}")
    print(f"成功率: {metrics.get('success_rate', 0):.2%}")
    print(f"重试次数: {metrics.get('retry_count', 0)}")
    print(f"错误总数: {metrics.get('total_errors', 0)}")
    
    # 6. 获取状态快照
    print("\n💾 6. 状态快照")
    snapshot = ops_workflow.get_state_snapshot("demo_workflow_001")
    if snapshot:
        print(f"工作流ID: {snapshot.get('workflow_id')}")
        print(f"当前阶段: {snapshot.get('stage')}")
        print(f"开始时间: {snapshot.get('start_time')}")
        print(f"最后更新: {snapshot.get('last_update')}")
    
    # 7. 显示工作流定义
    print("\n📋 7. 工作流定义")
    definition = ops_workflow.get_workflow_definition()
    print(f"名称: {definition['name']}")
    print(f"版本: {definition['version']}")
    print(f"节点数量: {len(definition['nodes'])}")
    print(f"阶段: {', '.join(definition['stages'])}")
    
    print("\n" + "=" * 50)
    print("🎉 智能运维智能体演示完成")


def demo_individual_components():
    """演示各个组件的独立使用"""
    print("\n🔧 组件独立使用演示")
    print("=" * 30)
    
    # DSPy 模块演示
    from src.dspy_modules.alert_analyzer import AlertAnalyzer
    from src.dspy_modules.diagnostic_agent import DiagnosticAgent, DiagnosticContext
    from src.dspy_modules.action_planner import ActionPlanner
    from src.dspy_modules.report_generator import ReportGenerator
    
    print("\n📊 DSPy 告警分析器演示")
    alert_analyzer = AlertAnalyzer()
    
    # 创建示例告警
    alert = AlertInfo(
        alert_id="demo_alert_001",
        timestamp=datetime.now().isoformat(),
        severity="high",
        source="cpu_monitor",
        message="High CPU usage detected on server-01",
        metrics={"cpu_usage": 0.90},
        tags=["cpu", "server-01"]
    )
    
    try:
        # 注意: 这里需要实际的 DSPy 模型配置才能运行
        # analysis = alert_analyzer.forward(alert)
        # print(f"分析结果: {analysis}")
        print("💡 DSPy 模块需要配置语言模型后才能运行")
    except Exception as e:
        print(f"⚠️  DSPy 模块演示跳过: {str(e)}")
    
    print("\n🏗️  LangGraph 状态管理演示")
    state_manager = StateManager()
    
    # 创建初始状态
    state = state_manager.initialize_state("demo_state_001")
    print(f"初始状态: {state_manager.get_state_summary(state)}")
    
    # 更新状态
    updated_state = state_manager.update_state(state, {
        "current_alert": alert,
        "workflow_stage": "alerting"
    })
    print(f"更新后状态: {state_manager.get_state_summary(updated_state)}")
    
    # 验证状态
    validation_errors = state_manager.validate_state(updated_state)
    if validation_errors:
        print(f"状态验证错误: {validation_errors}")
    else:
        print("✅ 状态验证通过")


if __name__ == "__main__":
    # 运行主演示
    asyncio.run(main())
    
    # 运行组件演示
    demo_individual_components()