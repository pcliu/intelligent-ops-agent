#!/usr/bin/env python3
"""
智能运维智能体完整演示

展示如何使用智能运维智能体进行完整的运维流程：
1. 创建和配置智能体
2. 处理真实运维场景
3. 多智能体协作
4. 学习和优化
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.intelligent_ops_agent import IntelligentOpsAgent, AgentConfig, AgentManager
from src.dspy_modules.alert_analyzer import AlertInfo


class OpsScenarioGenerator:
    """运维场景生成器"""
    
    @staticmethod
    def generate_cpu_spike_scenario() -> Dict[str, Any]:
        """生成CPU峰值场景"""
        return {
            "alert": {
                "alert_id": f"cpu_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": "high",
                "source": "system_monitor",
                "message": "CPU usage exceeded 90% threshold on production server",
                "metrics": {
                    "cpu_usage": 0.95,
                    "memory_usage": 0.78,
                    "disk_io": 0.82,
                    "network_io": 0.65
                },
                "tags": ["cpu", "performance", "production", "server-web-01"]
            },
            "context": {
                "environment": "production",
                "affected_services": ["web_service", "api_service"],
                "user_impact": "Response time increased by 200%",
                "business_impact": "Customer checkout process affected"
            }
        }
    
    @staticmethod
    def generate_memory_leak_scenario() -> Dict[str, Any]:
        """生成内存泄漏场景"""
        return {
            "alert": {
                "alert_id": f"memory_leak_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": "critical",
                "source": "application_monitor",
                "message": "Memory usage continuously increasing, potential memory leak detected",
                "metrics": {
                    "memory_usage": 0.92,
                    "memory_growth_rate": 0.05,  # 5% per hour
                    "gc_frequency": 500,  # per minute
                    "heap_size": 8000000000  # 8GB
                },
                "tags": ["memory", "leak", "application", "java"]
            },
            "context": {
                "environment": "production",
                "affected_services": ["order_service", "inventory_service"],
                "user_impact": "Service crashes every 2 hours",
                "business_impact": "Order processing failures"
            }
        }
    
    @staticmethod
    def generate_network_outage_scenario() -> Dict[str, Any]:
        """生成网络中断场景"""
        return {
            "alert": {
                "alert_id": f"network_outage_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": "critical",
                "source": "network_monitor",
                "message": "Network connectivity lost to external services",
                "metrics": {
                    "packet_loss": 0.80,
                    "latency": 5000,  # 5 seconds
                    "bandwidth_utilization": 0.05,
                    "connection_success_rate": 0.15
                },
                "tags": ["network", "outage", "external", "connectivity"]
            },
            "context": {
                "environment": "production",
                "affected_services": ["payment_service", "notification_service"],
                "user_impact": "Cannot complete transactions",
                "business_impact": "Revenue loss estimated at $10,000/hour"
            }
        }
    
    @staticmethod
    def generate_security_incident_scenario() -> Dict[str, Any]:
        """生成安全事件场景"""
        return {
            "alert": {
                "alert_id": f"security_incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": "critical",
                "source": "security_monitor",
                "message": "Suspicious login attempts detected from multiple IP addresses",
                "metrics": {
                    "failed_login_attempts": 150,
                    "suspicious_ips": 25,
                    "brute_force_score": 0.9,
                    "account_lockouts": 8
                },
                "tags": ["security", "authentication", "brute_force", "attack"]
            },
            "context": {
                "environment": "production",
                "affected_services": ["auth_service", "user_service"],
                "user_impact": "Legitimate users unable to login",
                "business_impact": "Potential data breach risk"
            }
        }


async def demo_single_agent_ops():
    """演示单个智能体运维流程"""
    print("\n🤖 单个智能体运维流程演示")
    print("=" * 50)
    
    # 1. 创建智能体
    print("\n📋 1. 创建智能运维智能体")
    config = AgentConfig(
        agent_id="ops_agent_001",
        agent_type="general",
        specialization="system_performance",
        max_retries=3,
        timeout=300,
        enable_learning=True,
        enable_reporting=True,
        auto_execution=False  # 演示模式，禁用自动执行
    )
    
    agent = IntelligentOpsAgent(config)
    print(f"✅ 智能体创建完成: {config.agent_id}")
    print(f"   类型: {config.agent_type}")
    print(f"   专业化: {config.specialization}")
    
    # 2. 处理 CPU 峰值场景
    print("\n🚨 2. 处理 CPU 峰值告警")
    cpu_scenario = OpsScenarioGenerator.generate_cpu_spike_scenario()
    
    print(f"告警信息: {cpu_scenario['alert']['message']}")
    print(f"CPU使用率: {cpu_scenario['alert']['metrics']['cpu_usage']:.1%}")
    print(f"业务影响: {cpu_scenario['context']['business_impact']}")
    
    # 处理告警
    result = await agent.process_alert(cpu_scenario['alert'])
    print(f"\n📊 处理结果: {result['status']}")
    
    if result['status'] == 'success':
        print("✅ 告警处理成功")
        if result.get('alert_analysis'):
            analysis = result['alert_analysis']
            print(f"   优先级: {analysis.priority}")
            print(f"   分类: {analysis.category}")
            print(f"   紧急程度: {analysis.urgency_score:.2f}")
        
        if result.get('diagnostic_result'):
            diagnosis = result['diagnostic_result']
            print(f"   根因: {diagnosis.root_cause}")
            print(f"   置信度: {diagnosis.confidence_score:.2f}")
        
        if result.get('action_plan'):
            plan = result['action_plan']
            print(f"   计划步骤: {len(plan.steps)}个")
            print(f"   预估时间: {plan.estimated_duration}分钟")
    
    # 3. 获取智能体状态
    print("\n📊 3. 智能体状态")
    status = agent.get_agent_status()
    print(f"状态: {status['current_state']}")
    print(f"工作流状态: {status['workflow_status']}")
    print(f"处理事件数: {status['incident_count']}")
    
    # 4. 获取性能指标
    print("\n📈 4. 性能指标")
    metrics = agent.get_performance_metrics()
    print(f"处理事件数: {metrics['incidents_processed']}")
    print(f"平均解决时间: {metrics['average_resolution_time']:.1f}秒")
    print(f"成功率: {metrics['success_rate']:.1%}")
    
    return agent


async def demo_multi_agent_collaboration():
    """演示多智能体协作"""
    print("\n🤝 多智能体协作演示")
    print("=" * 50)
    
    # 1. 创建智能体管理器
    print("\n📋 1. 创建智能体管理器")
    manager = AgentManager()
    
    # 2. 创建专业化智能体
    print("\n🤖 2. 创建专业化智能体")
    agents_config = [
        AgentConfig(
            agent_id="network_specialist",
            agent_type="network",
            specialization="network_diagnostics",
            enable_learning=True,
            enable_reporting=True
        ),
        AgentConfig(
            agent_id="security_specialist",
            agent_type="security",
            specialization="security_response",
            enable_learning=True,
            enable_reporting=True
        ),
        AgentConfig(
            agent_id="performance_specialist",
            agent_type="performance",
            specialization="performance_optimization",
            enable_learning=True,
            enable_reporting=True
        )
    ]
    
    agents = {}
    for config in agents_config:
        agent = manager.create_agent(config)
        agents[config.agent_id] = agent
        print(f"✅ 创建智能体: {config.agent_id} ({config.specialization})")
    
    # 3. 处理复杂场景
    print("\n🚨 3. 处理复杂安全事件")
    security_scenario = OpsScenarioGenerator.generate_security_incident_scenario()
    
    print(f"事件: {security_scenario['alert']['message']}")
    print(f"失败登录尝试: {security_scenario['alert']['metrics']['failed_login_attempts']}")
    print(f"可疑IP数量: {security_scenario['alert']['metrics']['suspicious_ips']}")
    
    # 安全智能体处理
    security_agent = agents['security_specialist']
    security_result = await security_agent.process_alert(security_scenario['alert'])
    
    print(f"\n🔒 安全智能体处理结果: {security_result['status']}")
    
    # 4. 网络故障场景
    print("\n🌐 4. 处理网络故障")
    network_scenario = OpsScenarioGenerator.generate_network_outage_scenario()
    
    print(f"事件: {network_scenario['alert']['message']}")
    print(f"丢包率: {network_scenario['alert']['metrics']['packet_loss']:.1%}")
    print(f"延迟: {network_scenario['alert']['metrics']['latency']}ms")
    
    # 网络智能体处理
    network_agent = agents['network_specialist']
    network_result = await network_agent.process_alert(network_scenario['alert'])
    
    print(f"\n🔗 网络智能体处理结果: {network_result['status']}")
    
    # 5. 性能问题场景
    print("\n⚡ 5. 处理性能问题")
    memory_scenario = OpsScenarioGenerator.generate_memory_leak_scenario()
    
    print(f"事件: {memory_scenario['alert']['message']}")
    print(f"内存使用率: {memory_scenario['alert']['metrics']['memory_usage']:.1%}")
    print(f"堆大小: {memory_scenario['alert']['metrics']['heap_size'] / 1000000000:.1f}GB")
    
    # 性能智能体处理
    performance_agent = agents['performance_specialist']
    performance_result = await performance_agent.process_alert(memory_scenario['alert'])
    
    print(f"\n🚀 性能智能体处理结果: {performance_result['status']}")
    
    # 6. 系统状态汇总
    print("\n📊 6. 系统状态汇总")
    system_status = manager.get_system_status()
    print(f"总智能体数: {system_status['total_agents']}")
    print(f"活跃智能体数: {system_status['active_agents']}")
    
    for agent_info in system_status['agent_list']:
        agent_id = agent_info['agent_id']
        status = agent_info['status']
        print(f"   {agent_id}: {status['current_state']} ({status['workflow_status']})")
    
    return manager


async def demo_learning_and_optimization():
    """演示学习和优化功能"""
    print("\n🧠 学习和优化功能演示")
    print("=" * 50)
    
    # 1. 创建支持学习的智能体
    print("\n📋 1. 创建支持学习的智能体")
    config = AgentConfig(
        agent_id="learning_agent",
        agent_type="adaptive",
        specialization="continuous_learning",
        enable_learning=True,
        enable_reporting=True
    )
    
    agent = IntelligentOpsAgent(config)
    print(f"✅ 学习智能体创建完成: {config.agent_id}")
    
    # 2. 处理多个场景以积累经验
    print("\n🔄 2. 处理多个场景以积累经验")
    scenarios = [
        OpsScenarioGenerator.generate_cpu_spike_scenario(),
        OpsScenarioGenerator.generate_memory_leak_scenario(),
        OpsScenarioGenerator.generate_network_outage_scenario()
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n   场景 {i}: {scenario['alert']['message']}")
        result = await agent.process_alert(scenario['alert'])
        print(f"   处理结果: {result['status']}")
    
    # 3. 模拟反馈数据
    print("\n📝 3. 提供反馈数据")
    feedback_data = {
        "user_satisfaction": 0.85,
        "resolution_effectiveness": 0.90,
        "false_positive_rate": 0.05,
        "automation_accuracy": 0.88,
        "response_time_satisfaction": 0.92
    }
    
    learning_result = await agent.learn_from_feedback(feedback_data)
    print(f"学习结果: {learning_result['status']}")
    
    if learning_result['status'] == 'success':
        update_info = learning_result['learning_update']
        print(f"   处理反馈点: {update_info['data_points_added']}")
        print(f"   总学习数据: {update_info['total_learning_data']}")
    
    # 4. 显示学习后的性能指标
    print("\n📊 4. 学习后的性能指标")
    metrics = agent.get_performance_metrics()
    print(f"处理事件数: {metrics['incidents_processed']}")
    print(f"学习数据点: {metrics['learning_data_points']}")
    print(f"当前性能: {json.dumps(metrics['current_performance'], indent=2)}")
    
    return agent


async def demo_advanced_diagnostics():
    """演示高级诊断功能"""
    print("\n🔍 高级诊断功能演示")
    print("=" * 50)
    
    # 1. 创建诊断专家智能体
    print("\n📋 1. 创建诊断专家智能体")
    config = AgentConfig(
        agent_id="diagnostic_expert",
        agent_type="diagnostic",
        specialization="root_cause_analysis",
        enable_learning=True,
        enable_reporting=True
    )
    
    agent = IntelligentOpsAgent(config)
    print(f"✅ 诊断专家创建完成: {config.agent_id}")
    
    # 2. 提供症状进行诊断
    print("\n🔍 2. 基于症状进行诊断")
    symptoms = [
        "API response time increased by 300%",
        "Database connection pool exhausted",
        "Memory usage gradually increasing",
        "CPU usage spikes during peak hours",
        "Error rate increased from 0.1% to 5%"
    ]
    
    context = {
        "system_metrics": {
            "cpu_usage": 0.85,
            "memory_usage": 0.90,
            "disk_io": 0.70,
            "network_latency": 250
        },
        "log_entries": [
            "2024-01-01 12:00:00 ERROR: Connection timeout",
            "2024-01-01 12:00:05 WARN: Memory pressure detected",
            "2024-01-01 12:00:10 ERROR: Database connection failed"
        ],
        "topology_info": {
            "services": ["web", "api", "database", "cache"],
            "dependencies": {
                "web": ["api"],
                "api": ["database", "cache"],
                "database": [],
                "cache": []
            }
        }
    }
    
    print("症状列表:")
    for i, symptom in enumerate(symptoms, 1):
        print(f"   {i}. {symptom}")
    
    diagnosis_result = await agent.diagnose_issue(symptoms, context)
    print(f"\n🔬 诊断结果: {diagnosis_result['status']}")
    
    if diagnosis_result['status'] == 'success':
        diagnosis = diagnosis_result['diagnosis']
        print(f"   根因: {diagnosis['root_cause']}")
        print(f"   置信度: {diagnosis['confidence_score']:.2f}")
        print(f"   影响评估: {diagnosis['impact_assessment']}")
        print(f"   受影响组件: {', '.join(diagnosis['affected_components'])}")
        print(f"   恢复时间估计: {diagnosis['recovery_estimate']}")
    
    # 3. 基于诊断结果生成行动计划
    print("\n📋 3. 生成行动计划")
    if diagnosis_result['status'] == 'success':
        plan_result = await agent.plan_actions(
            diagnosis_result['diagnosis'], 
            context
        )
        
        print(f"行动计划状态: {plan_result['status']}")
        
        if plan_result['status'] == 'success':
            plan = plan_result['action_plan']
            print(f"   计划ID: {plan['plan_id']}")
            print(f"   优先级: {plan['priority']}")
            print(f"   预估时间: {plan['estimated_duration']}分钟")
            print(f"   需要审批: {plan['approval_required']}")
            print(f"   步骤数量: {len(plan['steps'])}")
            
            # 显示前几个步骤
            print("   主要步骤:")
            for i, step in enumerate(plan['steps'][:3], 1):
                print(f"     {i}. {step['description']}")
    
    return agent


async def main():
    """主要演示函数"""
    print("🚀 智能运维智能体完整功能演示")
    print("=" * 60)
    
    # 单个智能体演示
    single_agent = await demo_single_agent_ops()
    
    # 多智能体协作演示
    multi_agent_manager = await demo_multi_agent_collaboration()
    
    # 学习和优化演示
    learning_agent = await demo_learning_and_optimization()
    
    # 高级诊断演示
    diagnostic_agent = await demo_advanced_diagnostics()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 演示总结")
    print("=" * 60)
    
    print(f"✅ 单个智能体演示: 完成")
    print(f"✅ 多智能体协作演示: 完成")
    print(f"✅ 学习和优化演示: 完成")
    print(f"✅ 高级诊断演示: 完成")
    
    print("\n💡 关键特性演示:")
    print("   🔄 LangGraph 工作流编排")
    print("   🧠 DSPy 模块化推理")
    print("   🤖 多智能体协作")
    print("   📊 智能诊断分析")
    print("   📋 自动化行动规划")
    print("   📈 持续学习优化")
    print("   📄 智能报告生成")
    
    print("\n🎉 智能运维智能体完整功能演示完成！")


if __name__ == "__main__":
    asyncio.run(main())