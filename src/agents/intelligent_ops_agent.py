"""
智能运维智能体主类

整合 LangGraph 和 DSPy 的智能运维智能体实现
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from ..langgraph_workflow.ops_workflow import OpsWorkflow, WorkflowFactory
from ..langgraph_workflow.state_manager import StateManager, OpsState
from ..dspy_modules.alert_analyzer import AlertInfo, AlertAnalyzer
from ..dspy_modules.diagnostic_agent import DiagnosticAgent
from ..dspy_modules.action_planner import ActionPlanner
from ..dspy_modules.report_generator import ReportGenerator
from ..utils.llm_config import setup_deepseek_llm, get_llm_config_from_env


@dataclass
class AgentConfig:
    """智能体配置"""
    agent_id: str
    agent_type: str = "general"
    specialization: Optional[str] = None
    max_retries: int = 3
    timeout: int = 300
    enable_learning: bool = True
    enable_reporting: bool = True
    auto_execution: bool = False


class IntelligentOpsAgent:
    """智能运维智能体
    
    整合 LangGraph 工作流编排和 DSPy 模块化推理，
    提供智能化的运维决策和自动化执行能力。
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # 初始化 LLM (DeepSeek)
        try:
            llm_config = get_llm_config_from_env()
            self.dspy_lm, self.langchain_llm = setup_deepseek_llm(llm_config)
            print(f"✅ LLM 初始化成功: {llm_config.provider} - {llm_config.model_name}")
        except Exception as e:
            print(f"⚠️  LLM 初始化失败: {str(e)}")
            print("   将使用模拟模式运行")
            self.dspy_lm = None
            self.langchain_llm = None
        
        self.workflow = WorkflowFactory.create_ops_workflow({
            "agent_type": config.agent_type,
            "specialization": config.specialization,
            "max_retries": config.max_retries,
            "timeout": config.timeout
        })
        
        # DSPy 模块
        self.alert_analyzer = AlertAnalyzer()
        self.diagnostic_agent = DiagnosticAgent()
        self.action_planner = ActionPlanner()
        self.report_generator = ReportGenerator()
        
        # 状态管理
        self.state_manager = StateManager()
        self.current_state: Optional[OpsState] = None
        
        # 历史记录
        self.incident_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # 学习数据
        self.learning_data: Dict[str, Any] = {}
        
    async def process_alert(self, alert: Union[AlertInfo, Dict[str, Any]]) -> Dict[str, Any]:
        """处理告警
        
        Args:
            alert: 告警信息
            
        Returns:
            Dict: 处理结果
        """
        try:
            # 转换告警格式
            if isinstance(alert, dict):
                alert_info = AlertInfo(**alert)
            else:
                alert_info = alert
            
            # 初始化工作流状态
            workflow_id = f"{self.config.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_state = self.state_manager.initialize_state(workflow_id)
            
            # 设置告警
            self.current_state = self.state_manager.update_state(self.current_state, {
                "current_alert": alert_info,
                "max_retries": self.config.max_retries
            })
            
            # 运行工作流
            final_state = await self.workflow.run(self.current_state)
            
            # 更新当前状态
            self.current_state = final_state
            
            # 记录历史
            if self.config.enable_learning:
                await self._update_learning_data(final_state)
            
            # 生成处理结果
            result = self._generate_process_result(final_state)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def diagnose_issue(self, symptoms: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """诊断问题
        
        Args:
            symptoms: 症状列表
            context: 上下文信息
            
        Returns:
            Dict: 诊断结果
        """
        try:
            # 创建诊断上下文
            from ..dspy_modules.diagnostic_agent import DiagnosticContext
            from ..dspy_modules.alert_analyzer import AlertAnalysisResult
            
            # 模拟告警分析结果
            alert_analysis = AlertAnalysisResult(
                alert_id="diagnostic_request",
                priority="medium",
                category="investigation",
                urgency_score=0.5,
                root_cause_hints=symptoms,
                recommended_actions=[]
            )
            
            diagnostic_context = DiagnosticContext(
                alert_analysis=alert_analysis,
                system_metrics=context.get("system_metrics", {}),
                log_entries=context.get("log_entries", []),
                historical_incidents=self.incident_history,
                topology_info=context.get("topology_info", {})
            )
            
            # 执行诊断
            diagnostic_result = self.diagnostic_agent.forward(diagnostic_context)
            
            return {
                "status": "success",
                "diagnosis": {
                    "root_cause": diagnostic_result.root_cause,
                    "confidence_score": diagnostic_result.confidence_score,
                    "impact_assessment": diagnostic_result.impact_assessment,
                    "affected_components": diagnostic_result.affected_components,
                    "business_impact": diagnostic_result.business_impact,
                    "recovery_estimate": diagnostic_result.recovery_time_estimate,
                    "similar_incidents": diagnostic_result.similar_incidents,
                    "evidence": diagnostic_result.evidence
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def plan_actions(self, diagnostic_result: Dict[str, Any], 
                          system_context: Dict[str, Any]) -> Dict[str, Any]:
        """规划行动
        
        Args:
            diagnostic_result: 诊断结果
            system_context: 系统上下文
            
        Returns:
            Dict: 行动计划
        """
        try:
            from ..dspy_modules.diagnostic_agent import DiagnosticResult
            
            # 转换诊断结果
            diag_result = DiagnosticResult(
                incident_id=diagnostic_result.get("incident_id", "plan_request"),
                root_cause=diagnostic_result.get("root_cause", "Unknown"),
                confidence_score=diagnostic_result.get("confidence_score", 0.5),
                impact_assessment=diagnostic_result.get("impact_assessment", "medium"),
                affected_components=diagnostic_result.get("affected_components", []),
                business_impact=diagnostic_result.get("business_impact", "Unknown"),
                recovery_time_estimate=diagnostic_result.get("recovery_estimate", "Unknown"),
                similar_incidents=diagnostic_result.get("similar_incidents", []),
                evidence=diagnostic_result.get("evidence", [])
            )
            
            # 生成行动计划
            action_plan = self.action_planner.forward(diag_result, system_context)
            
            return {
                "status": "success",
                "action_plan": {
                    "plan_id": action_plan.plan_id,
                    "priority": action_plan.priority,
                    "estimated_duration": action_plan.estimated_duration,
                    "risk_assessment": action_plan.risk_assessment,
                    "approval_required": action_plan.approval_required,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "action_type": step.action_type,
                            "description": step.description,
                            "command": step.command,
                            "timeout": step.timeout,
                            "risk_level": step.risk_level
                        }
                        for step in action_plan.steps
                    ],
                    "rollback_plan": [
                        {
                            "step_id": step.step_id,
                            "description": step.description,
                            "command": step.command
                        }
                        for step in action_plan.rollback_plan
                    ],
                    "pre_checks": action_plan.pre_checks,
                    "post_checks": action_plan.post_checks,
                    "notifications": action_plan.notifications
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_report(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告
        
        Args:
            incident_data: 事件数据
            
        Returns:
            Dict: 报告结果
        """
        try:
            if not self.config.enable_reporting:
                return {
                    "status": "disabled",
                    "message": "Reporting is disabled for this agent"
                }
            
            # 这里需要根据实际状态生成报告
            # 简化实现，返回基本报告信息
            report = {
                "incident_id": incident_data.get("incident_id", "unknown"),
                "title": incident_data.get("title", "System Incident"),
                "summary": incident_data.get("summary", "Automated incident response completed"),
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.config.agent_id,
                "status": "generated"
            }
            
            return {
                "status": "success",
                "report": report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_actions(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行行动计划
        
        Args:
            action_plan: 行动计划
            
        Returns:
            Dict: 执行结果
        """
        try:
            if not self.config.auto_execution:
                return {
                    "status": "manual_approval_required",
                    "message": "Automatic execution is disabled. Manual approval required.",
                    "plan_id": action_plan.get("plan_id", "unknown")
                }
            
            # 模拟执行过程
            executed_steps = []
            failed_steps = []
            
            for step in action_plan.get("steps", []):
                try:
                    # 模拟步骤执行
                    await asyncio.sleep(0.1)  # 模拟执行时间
                    executed_steps.append(step["step_id"])
                except Exception as e:
                    failed_steps.append({
                        "step_id": step["step_id"],
                        "error": str(e)
                    })
            
            execution_status = "success" if not failed_steps else "partial" if executed_steps else "failed"
            
            return {
                "status": execution_status,
                "execution_result": {
                    "plan_id": action_plan.get("plan_id", "unknown"),
                    "executed_steps": executed_steps,
                    "failed_steps": failed_steps,
                    "execution_time": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取智能体状态
        
        Returns:
            Dict: 智能体状态
        """
        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "specialization": self.config.specialization,
            "current_state": self.current_state.get("workflow_stage") if self.current_state else "idle",
            "workflow_status": self.current_state.get("workflow_status") if self.current_state else "ready",
            "incident_count": len(self.incident_history),
            "learning_enabled": self.config.enable_learning,
            "reporting_enabled": self.config.enable_reporting,
            "auto_execution_enabled": self.config.auto_execution,
            "last_update": datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标
        
        Returns:
            Dict: 性能指标
        """
        return {
            "agent_id": self.config.agent_id,
            "incidents_processed": len(self.incident_history),
            "average_resolution_time": self._calculate_avg_resolution_time(),
            "success_rate": self._calculate_success_rate(),
            "learning_data_points": len(self.learning_data),
            "current_performance": self.performance_metrics.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """从反馈中学习
        
        Args:
            feedback: 反馈数据
            
        Returns:
            Dict: 学习结果
        """
        try:
            if not self.config.enable_learning:
                return {
                    "status": "disabled",
                    "message": "Learning is disabled for this agent"
                }
            
            # 更新学习数据
            self.learning_data.update(feedback)
            
            # 这里可以实现更复杂的学习逻辑
            # 比如更新 DSPy 模块的优化器
            
            return {
                "status": "success",
                "learning_update": {
                    "feedback_processed": True,
                    "data_points_added": len(feedback),
                    "total_learning_data": len(self.learning_data)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_process_result(self, final_state: OpsState) -> Dict[str, Any]:
        """生成处理结果"""
        return {
            "status": "success" if final_state["workflow_status"] == "completed" else "failed",
            "workflow_id": final_state["workflow_id"],
            "final_stage": final_state["workflow_stage"],
            "errors": final_state.get("errors", []),
            "alert_analysis": final_state.get("alert_analysis"),
            "diagnostic_result": final_state.get("diagnostic_result"),
            "action_plan": final_state.get("action_plan"),
            "execution_result": final_state.get("execution_result"),
            "incident_report": final_state.get("incident_report"),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _update_learning_data(self, final_state: OpsState):
        """更新学习数据"""
        if final_state.get("incident_report"):
            self.incident_history.append({
                "incident_id": final_state["incident_report"].incident_id,
                "timestamp": datetime.now().isoformat(),
                "root_cause": final_state["incident_report"].root_cause_analysis,
                "resolution": final_state["incident_report"].resolution_summary,
                "lessons_learned": final_state["incident_report"].lessons_learned
            })
    
    def _calculate_avg_resolution_time(self) -> float:
        """计算平均解决时间"""
        if not self.incident_history:
            return 0.0
        
        # 简化计算
        return 300.0  # 假设平均5分钟
    
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if not self.incident_history:
            return 0.0
        
        # 简化计算
        return 0.85  # 假设85%成功率


class AgentManager:
    """智能体管理器"""
    
    def __init__(self):
        self.agents: Dict[str, IntelligentOpsAgent] = {}
    
    def create_agent(self, config: AgentConfig) -> IntelligentOpsAgent:
        """创建智能体"""
        agent = IntelligentOpsAgent(config)
        self.agents[config.agent_id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[IntelligentOpsAgent]:
        """获取智能体"""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """列出所有智能体"""
        return list(self.agents.keys())
    
    def remove_agent(self, agent_id: str) -> bool:
        """移除智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.current_state]),
            "agent_list": [
                {
                    "agent_id": agent_id,
                    "status": agent.get_agent_status()
                }
                for agent_id, agent in self.agents.items()
            ],
            "timestamp": datetime.now().isoformat()
        }