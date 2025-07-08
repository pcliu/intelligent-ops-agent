#!/usr/bin/env python3
"""
多智能体协作运维场景示例

本示例展示如何使用多个智能体协作处理复杂的运维场景：
1. 网络故障诊断智能体
2. 数据库性能优化智能体
3. 应用程序监控智能体
4. 安全事件响应智能体
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.langgraph_workflow.ops_workflow import OpsWorkflow, WorkflowFactory
from src.dspy_modules.alert_analyzer import AlertInfo
from src.langgraph_workflow.state_manager import StateManager


class MultiAgentOpsSystem:
    """多智能体运维系统"""
    
    def __init__(self):
        self.agents: Dict[str, OpsWorkflow] = {}
        self.state_manager = StateManager()
        self.shared_context = {}
        
    def register_agent(self, agent_name: str, workflow: OpsWorkflow):
        """注册智能体"""
        self.agents[agent_name] = workflow
        
    async def coordinate_agents(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """协调多个智能体处理事件"""
        results = {}
        
        # 根据事件类型分配给相应的智能体
        incident_type = incident_data.get('type', 'general')
        
        if incident_type == 'network':
            results['network_agent'] = await self._run_network_agent(incident_data)
        elif incident_type == 'database':
            results['database_agent'] = await self._run_database_agent(incident_data)
        elif incident_type == 'application':
            results['application_agent'] = await self._run_application_agent(incident_data)
        elif incident_type == 'security':
            results['security_agent'] = await self._run_security_agent(incident_data)
        else:
            # 复杂事件需要多个智能体协作
            results = await self._run_collaborative_agents(incident_data)
        
        return results
    
    async def _run_network_agent(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行网络诊断智能体"""
        if 'network_agent' not in self.agents:
            return {"error": "Network agent not registered"}
        
        # 创建网络相关的初始状态
        initial_state = self.state_manager.initialize_state("network_incident_001")
        
        # 添加网络特定的告警
        network_alert = AlertInfo(
            alert_id=f"network_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity=incident_data.get('severity', 'high'),
            source='network_monitor',
            message=incident_data.get('message', 'Network connectivity issue detected'),
            metrics=incident_data.get('metrics', {
                'latency': 500,
                'packet_loss': 0.05,
                'bandwidth_usage': 0.95
            }),
            tags=['network', 'connectivity', 'latency']
        )
        
        initial_state = self.state_manager.update_state(initial_state, {
            'current_alert': network_alert,
            'system_context': {
                'agent_type': 'network',
                'specialized_tools': ['ping', 'traceroute', 'netstat'],
                'expertise_domain': 'network_infrastructure'
            }
        })
        
        # 运行网络智能体
        result = await self.agents['network_agent'].run(initial_state)
        
        return {
            'agent_type': 'network',
            'incident_id': network_alert.alert_id,
            'resolution_status': result.get('workflow_status', 'unknown'),
            'recommendations': self._extract_network_recommendations(result)
        }
    
    async def _run_database_agent(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行数据库性能优化智能体"""
        if 'database_agent' not in self.agents:
            return {"error": "Database agent not registered"}
        
        initial_state = self.state_manager.initialize_state("database_incident_001")
        
        database_alert = AlertInfo(
            alert_id=f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity=incident_data.get('severity', 'high'),
            source='database_monitor',
            message=incident_data.get('message', 'Database performance degradation detected'),
            metrics=incident_data.get('metrics', {
                'query_response_time': 2500,
                'connection_pool_usage': 0.90,
                'deadlock_count': 15
            }),
            tags=['database', 'performance', 'postgresql']
        )
        
        initial_state = self.state_manager.update_state(initial_state, {
            'current_alert': database_alert,
            'system_context': {
                'agent_type': 'database',
                'specialized_tools': ['pg_stat_activity', 'explain_plan', 'index_advisor'],
                'expertise_domain': 'database_optimization'
            }
        })
        
        result = await self.agents['database_agent'].run(initial_state)
        
        return {
            'agent_type': 'database',
            'incident_id': database_alert.alert_id,
            'resolution_status': result.get('workflow_status', 'unknown'),
            'recommendations': self._extract_database_recommendations(result)
        }
    
    async def _run_application_agent(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行应用程序监控智能体"""
        if 'application_agent' not in self.agents:
            return {"error": "Application agent not registered"}
        
        initial_state = self.state_manager.initialize_state("application_incident_001")
        
        app_alert = AlertInfo(
            alert_id=f"application_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity=incident_data.get('severity', 'medium'),
            source='application_monitor',
            message=incident_data.get('message', 'Application error rate increased'),
            metrics=incident_data.get('metrics', {
                'error_rate': 0.15,
                'response_time': 3000,
                'memory_usage': 0.85
            }),
            tags=['application', 'error_rate', 'java']
        )
        
        initial_state = self.state_manager.update_state(initial_state, {
            'current_alert': app_alert,
            'system_context': {
                'agent_type': 'application',
                'specialized_tools': ['jstack', 'heap_dump', 'profiler'],
                'expertise_domain': 'application_debugging'
            }
        })
        
        result = await self.agents['application_agent'].run(initial_state)
        
        return {
            'agent_type': 'application',
            'incident_id': app_alert.alert_id,
            'resolution_status': result.get('workflow_status', 'unknown'),
            'recommendations': self._extract_application_recommendations(result)
        }
    
    async def _run_security_agent(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行安全事件响应智能体"""
        if 'security_agent' not in self.agents:
            return {"error": "Security agent not registered"}
        
        initial_state = self.state_manager.initialize_state("security_incident_001")
        
        security_alert = AlertInfo(
            alert_id=f"security_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity=incident_data.get('severity', 'critical'),
            source='security_monitor',
            message=incident_data.get('message', 'Suspicious activity detected'),
            metrics=incident_data.get('metrics', {
                'failed_login_attempts': 50,
                'suspicious_ip_count': 10,
                'data_exfiltration_risk': 0.8
            }),
            tags=['security', 'intrusion', 'anomaly']
        )
        
        initial_state = self.state_manager.update_state(initial_state, {
            'current_alert': security_alert,
            'system_context': {
                'agent_type': 'security',
                'specialized_tools': ['siem', 'threat_intel', 'forensics'],
                'expertise_domain': 'security_response'
            }
        })
        
        result = await self.agents['security_agent'].run(initial_state)
        
        return {
            'agent_type': 'security',
            'incident_id': security_alert.alert_id,
            'resolution_status': result.get('workflow_status', 'unknown'),
            'recommendations': self._extract_security_recommendations(result)
        }
    
    async def _run_collaborative_agents(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行协作智能体处理复杂事件"""
        # 同时运行多个智能体
        tasks = []
        
        # 网络智能体
        if 'network_agent' in self.agents:
            tasks.append(self._run_network_agent(incident_data))
        
        # 数据库智能体
        if 'database_agent' in self.agents:
            tasks.append(self._run_database_agent(incident_data))
        
        # 应用智能体
        if 'application_agent' in self.agents:
            tasks.append(self._run_application_agent(incident_data))
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整合结果
        integrated_result = {
            'collaboration_type': 'parallel',
            'participating_agents': len(tasks),
            'individual_results': results,
            'integrated_recommendations': self._integrate_recommendations(results)
        }
        
        return integrated_result
    
    def _extract_network_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """提取网络相关建议"""
        return [
            "检查网络设备状态",
            "优化路由配置",
            "监控带宽使用情况",
            "检查DNS解析时间"
        ]
    
    def _extract_database_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """提取数据库相关建议"""
        return [
            "优化慢查询",
            "调整连接池配置",
            "检查索引使用情况",
            "监控锁等待时间"
        ]
    
    def _extract_application_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """提取应用程序相关建议"""
        return [
            "分析异常日志",
            "检查内存泄漏",
            "优化代码性能",
            "监控JVM参数"
        ]
    
    def _extract_security_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """提取安全相关建议"""
        return [
            "封锁可疑IP地址",
            "重置受影响账户密码",
            "加强访问控制",
            "进行取证分析"
        ]
    
    def _integrate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """整合多个智能体的建议"""
        integrated = []
        
        for result in results:
            if isinstance(result, dict) and 'recommendations' in result:
                integrated.extend(result['recommendations'])
        
        # 去重和优先级排序
        unique_recommendations = list(set(integrated))
        
        return unique_recommendations


async def demo_multi_agent_network_incident():
    """演示网络故障处理场景"""
    print("\n🌐 网络故障处理场景演示")
    print("=" * 40)
    
    # 创建多智能体系统
    multi_agent_system = MultiAgentOpsSystem()
    
    # 创建并注册网络智能体
    network_agent = WorkflowFactory.create_ops_workflow({
        "agent_type": "network",
        "specialization": "network_infrastructure"
    })
    multi_agent_system.register_agent("network_agent", network_agent)
    
    # 模拟网络故障事件
    network_incident = {
        "type": "network",
        "severity": "high",
        "message": "Network latency spike detected across multiple regions",
        "metrics": {
            "latency": 800,
            "packet_loss": 0.12,
            "bandwidth_usage": 0.98
        },
        "affected_regions": ["us-east-1", "us-west-2"],
        "impact": "Service degradation affecting 15% of users"
    }
    
    print(f"📊 网络事件: {network_incident['message']}")
    print(f"📈 延迟: {network_incident['metrics']['latency']}ms")
    print(f"📉 丢包率: {network_incident['metrics']['packet_loss']:.2%}")
    
    # 运行网络智能体
    try:
        result = await multi_agent_system.coordinate_agents(network_incident)
        print(f"\n✅ 处理结果: {result}")
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


async def demo_multi_agent_database_incident():
    """演示数据库性能问题处理场景"""
    print("\n🗄️ 数据库性能问题处理场景演示")
    print("=" * 40)
    
    multi_agent_system = MultiAgentOpsSystem()
    
    # 创建并注册数据库智能体
    database_agent = WorkflowFactory.create_ops_workflow({
        "agent_type": "database",
        "specialization": "database_optimization"
    })
    multi_agent_system.register_agent("database_agent", database_agent)
    
    # 模拟数据库性能问题
    database_incident = {
        "type": "database",
        "severity": "critical",
        "message": "Database query response time exceeding SLA",
        "metrics": {
            "query_response_time": 5000,
            "connection_pool_usage": 0.95,
            "deadlock_count": 25,
            "cache_hit_ratio": 0.65
        },
        "affected_services": ["user_service", "order_service"],
        "impact": "API response time increased by 300%"
    }
    
    print(f"📊 数据库事件: {database_incident['message']}")
    print(f"⏱️ 响应时间: {database_incident['metrics']['query_response_time']}ms")
    print(f"🔒 死锁数量: {database_incident['metrics']['deadlock_count']}")
    
    try:
        result = await multi_agent_system.coordinate_agents(database_incident)
        print(f"\n✅ 处理结果: {result}")
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


async def demo_collaborative_incident():
    """演示多智能体协作处理复杂事件"""
    print("\n🤝 多智能体协作场景演示")
    print("=" * 40)
    
    multi_agent_system = MultiAgentOpsSystem()
    
    # 注册多个智能体
    agents = {
        "network_agent": {"agent_type": "network", "specialization": "network_infrastructure"},
        "database_agent": {"agent_type": "database", "specialization": "database_optimization"},
        "application_agent": {"agent_type": "application", "specialization": "application_debugging"}
    }
    
    for agent_name, config in agents.items():
        agent = WorkflowFactory.create_ops_workflow(config)
        multi_agent_system.register_agent(agent_name, agent)
    
    # 模拟复杂的系统性故障
    complex_incident = {
        "type": "complex",
        "severity": "critical",
        "message": "System-wide performance degradation affecting multiple components",
        "metrics": {
            "overall_response_time": 8000,
            "error_rate": 0.25,
            "resource_utilization": 0.95
        },
        "affected_components": ["load_balancer", "database", "cache", "application_servers"],
        "impact": "Service completely unavailable for 45% of users"
    }
    
    print(f"📊 复杂事件: {complex_incident['message']}")
    print(f"⏱️ 响应时间: {complex_incident['metrics']['overall_response_time']}ms")
    print(f"❌ 错误率: {complex_incident['metrics']['error_rate']:.2%}")
    print(f"🔧 受影响组件: {', '.join(complex_incident['affected_components'])}")
    
    try:
        result = await multi_agent_system.coordinate_agents(complex_incident)
        print(f"\n✅ 协作处理结果: {result}")
        
        # 显示集成建议
        if 'integrated_recommendations' in result:
            print("\n📋 集成建议:")
            for i, recommendation in enumerate(result['integrated_recommendations'], 1):
                print(f"   {i}. {recommendation}")
        
    except Exception as e:
        print(f"❌ 协作处理失败: {str(e)}")


async def main():
    """主要演示函数"""
    print("🤖 多智能体运维系统演示")
    print("=" * 50)
    
    # 网络故障演示
    await demo_multi_agent_network_incident()
    
    # 数据库性能问题演示
    await demo_multi_agent_database_incident()
    
    # 多智能体协作演示
    await demo_collaborative_incident()
    
    print("\n" + "=" * 50)
    print("🎉 多智能体运维系统演示完成")


if __name__ == "__main__":
    asyncio.run(main())