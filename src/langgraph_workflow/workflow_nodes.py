import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..dspy_modules.alert_analyzer import AlertAnalyzer, AlertInfo, AlertAnalysisResult
from ..dspy_modules.diagnostic_agent import DiagnosticAgent, DiagnosticContext
from ..dspy_modules.action_planner import ActionPlanner
from ..dspy_modules.report_generator import ReportGenerator, ExecutionResult
from .state_manager import OpsState, StateManager


class WorkflowNodes:
    """工作流节点集合"""
    
    def __init__(self):
        self.alert_analyzer = AlertAnalyzer()
        self.diagnostic_agent = DiagnosticAgent()
        self.action_planner = ActionPlanner()
        self.report_generator = ReportGenerator()
        self.state_manager = StateManager()
    
    async def monitor_collect_node(self, state: OpsState) -> OpsState:
        """监控数据采集节点"""
        try:
            # 模拟监控数据采集
            monitoring_data = await self._collect_monitoring_data()
            
            # 更新系统指标
            state = self.state_manager.update_state(state, {
                "system_metrics": monitoring_data["metrics"],
                "system_context": monitoring_data["context"]
            })
            
            # 检查是否有新告警
            if monitoring_data.get("alerts"):
                # 创建告警信息
                alert_info = AlertInfo(
                    alert_id=monitoring_data["alerts"][0]["id"],
                    timestamp=datetime.now().isoformat(),
                    severity=monitoring_data["alerts"][0]["severity"],
                    source=monitoring_data["alerts"][0]["source"],
                    message=monitoring_data["alerts"][0]["message"],
                    metrics=monitoring_data["alerts"][0].get("metrics", {}),
                    tags=monitoring_data["alerts"][0].get("tags", [])
                )
                
                state = self.state_manager.update_state(state, {
                    "current_alert": alert_info
                })
                
                # 转换到告警处理阶段
                state = self.state_manager.transition_stage(state, "alerting")
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Monitor collect error: {str(e)}")
            return state
    
    async def alert_process_node(self, state: OpsState) -> OpsState:
        """告警处理节点"""
        try:
            if not state["current_alert"]:
                raise ValueError("No current alert to process")
            
            # 分析告警
            alert_analysis = self.alert_analyzer.forward(
                alert_info=state["current_alert"],
                historical_alerts=state["historical_alerts"]
            )
            
            # 更新状态
            state = self.state_manager.update_state(state, {
                "alert_analysis": alert_analysis,
                "historical_alerts": state["historical_alerts"] + [state["current_alert"]]
            })
            
            # 转换到诊断阶段
            state = self.state_manager.transition_stage(state, "diagnosis")
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Alert process error: {str(e)}")
            return state
    
    async def diagnosis_node(self, state: OpsState) -> OpsState:
        """故障诊断节点"""
        try:
            if not state["alert_analysis"]:
                raise ValueError("No alert analysis for diagnosis")
            
            # 构建诊断上下文
            diagnostic_context = DiagnosticContext(
                alert_analysis=state["alert_analysis"],
                system_metrics=state["system_metrics"],
                log_entries=await self._collect_log_entries(state),
                historical_incidents=state["incident_history"],
                topology_info=state["system_context"].get("topology", {})
            )
            
            # 执行诊断
            diagnostic_result = self.diagnostic_agent.forward(diagnostic_context)
            
            # 更新状态
            state = self.state_manager.update_state(state, {
                "diagnostic_context": diagnostic_context.dict(),
                "diagnostic_result": diagnostic_result
            })
            
            # 转换到规划阶段
            state = self.state_manager.transition_stage(state, "planning")
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Diagnosis error: {str(e)}")
            return state
    
    async def action_plan_node(self, state: OpsState) -> OpsState:
        """行动规划节点"""
        try:
            if not state["diagnostic_result"]:
                raise ValueError("No diagnostic result for action planning")
            
            # 生成行动计划
            action_plan = self.action_planner.forward(
                diagnostic_result=state["diagnostic_result"],
                system_context=state["system_context"]
            )
            
            # 更新状态
            state = self.state_manager.update_state(state, {
                "action_plan": action_plan
            })
            
            # 转换到执行阶段
            state = self.state_manager.transition_stage(state, "execution")
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Action planning error: {str(e)}")
            return state
    
    async def action_execute_node(self, state: OpsState) -> OpsState:
        """自动化执行节点"""
        try:
            if not state["action_plan"]:
                raise ValueError("No action plan for execution")
            
            # 执行行动计划
            execution_result = await self._execute_action_plan(state["action_plan"])
            
            # 更新状态
            state = self.state_manager.update_state(state, {
                "execution_result": execution_result
            })
            
            # 转换到报告阶段
            state = self.state_manager.transition_stage(state, "reporting")
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Action execution error: {str(e)}")
            return state
    
    async def report_generate_node(self, state: OpsState) -> OpsState:
        """报告生成节点"""
        try:
            if not all([state["diagnostic_result"], state["action_plan"], state["execution_result"]]):
                raise ValueError("Missing required data for report generation")
            
            # 生成事件报告
            incident_report = self.report_generator.generate_incident_report(
                diagnostic_result=state["diagnostic_result"],
                action_plan=state["action_plan"],
                execution_result=state["execution_result"]
            )
            
            # 更新状态
            state = self.state_manager.update_state(state, {
                "incident_report": incident_report
            })
            
            # 将事件添加到历史记录
            incident_record = {
                "incident_id": incident_report.incident_id,
                "timestamp": datetime.now().isoformat(),
                "root_cause": incident_report.root_cause_analysis,
                "resolution": incident_report.resolution_summary,
                "lessons_learned": incident_report.lessons_learned
            }
            
            state = self.state_manager.update_state(state, {
                "incident_history": state["incident_history"] + [incident_record]
            })
            
            # 完成工作流
            state = self.state_manager.update_workflow_status(state, "completed")
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Report generation error: {str(e)}")
            return state
    
    async def feedback_learn_node(self, state: OpsState) -> OpsState:
        """反馈学习节点"""
        try:
            # 收集反馈数据
            feedback_data = await self._collect_feedback_data(state)
            
            # 更新学习数据
            learning_data = state["learning_data"].copy()
            learning_data.update(feedback_data)
            
            state = self.state_manager.update_state(state, {
                "learning_data": learning_data
            })
            
            # 重置工作流到监控状态
            state = self.state_manager.transition_stage(state, "monitoring")
            state = self.state_manager.reset_retry(state)
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Feedback learning error: {str(e)}")
            return state
    
    async def error_handling_node(self, state: OpsState) -> OpsState:
        """错误处理节点"""
        try:
            # 检查是否可以重试
            if self.state_manager.can_retry(state):
                state = self.state_manager.increment_retry(state)
                
                # 根据错误类型决定重试策略
                if "Alert process error" in str(state["errors"]):
                    state = self.state_manager.transition_stage(state, "alerting")
                elif "Diagnosis error" in str(state["errors"]):
                    state = self.state_manager.transition_stage(state, "diagnosis")
                elif "Action planning error" in str(state["errors"]):
                    state = self.state_manager.transition_stage(state, "planning")
                elif "Action execution error" in str(state["errors"]):
                    state = self.state_manager.transition_stage(state, "execution")
                else:
                    state = self.state_manager.transition_stage(state, "monitoring")
            else:
                # 超过重试次数，标记为失败
                state = self.state_manager.update_workflow_status(state, "failed")
            
            return state
            
        except Exception as e:
            state = self.state_manager.add_error(state, f"Error handling error: {str(e)}")
            return state
    
    async def _collect_monitoring_data(self) -> Dict[str, Any]:
        """收集监控数据"""
        # 模拟监控数据收集
        await asyncio.sleep(0.1)
        
        return {
            "metrics": {
                "cpu_usage": 0.75,
                "memory_usage": 0.60,
                "disk_usage": 0.45,
                "network_latency": 50
            },
            "context": {
                "environment": "production",
                "region": "us-east-1",
                "cluster": "ops-cluster-1",
                "topology": {
                    "services": ["web", "api", "database"],
                    "dependencies": {
                        "web": ["api"],
                        "api": ["database"]
                    }
                }
            },
            "alerts": [
                {
                    "id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "severity": "high",
                    "source": "monitoring_system",
                    "message": "CPU usage exceeds 70% threshold",
                    "metrics": {"cpu_usage": 0.75},
                    "tags": ["cpu", "performance"]
                }
            ]
        }
    
    async def _collect_log_entries(self, state: OpsState) -> List[str]:
        """收集日志条目"""
        # 模拟日志收集
        await asyncio.sleep(0.1)
        
        return [
            "2024-01-01 12:00:00 ERROR: CPU usage spike detected",
            "2024-01-01 12:00:05 WARN: Memory pressure increasing",
            "2024-01-01 12:00:10 INFO: Attempting to scale resources"
        ]
    
    async def _execute_action_plan(self, action_plan) -> ExecutionResult:
        """执行行动计划"""
        # 模拟行动计划执行
        await asyncio.sleep(0.5)
        
        start_time = datetime.now()
        executed_steps = []
        failed_steps = []
        
        for step in action_plan.steps:
            try:
                # 模拟步骤执行
                await asyncio.sleep(0.1)
                executed_steps.append(step.step_id)
            except Exception as e:
                failed_steps.append(step.step_id)
        
        end_time = datetime.now()
        
        return ExecutionResult(
            plan_id=action_plan.plan_id,
            execution_start=start_time,
            execution_end=end_time,
            status="success" if not failed_steps else "partial" if executed_steps else "failed",
            executed_steps=executed_steps,
            failed_steps=failed_steps,
            rollback_executed=False,
            final_state={"status": "resolved"}
        )
    
    async def _collect_feedback_data(self, state: OpsState) -> Dict[str, Any]:
        """收集反馈数据"""
        # 模拟反馈数据收集
        await asyncio.sleep(0.1)
        
        return {
            "resolution_success": True,
            "user_satisfaction": 0.85,
            "time_to_resolution": 300,  # 秒
            "automation_effectiveness": 0.90
        }
    
    def get_node_router(self) -> Dict[str, Any]:
        """获取节点路由配置"""
        return {
            "monitor_collect": self.monitor_collect_node,
            "alert_process": self.alert_process_node,
            "diagnosis": self.diagnosis_node,
            "action_plan": self.action_plan_node,
            "action_execute": self.action_execute_node,
            "report_generate": self.report_generate_node,
            "feedback_learn": self.feedback_learn_node,
            "error_handling": self.error_handling_node
        }
    
    def get_conditional_edges(self) -> Dict[str, Any]:
        """获取条件边配置"""
        def route_from_monitor(state: OpsState) -> str:
            if state["current_alert"]:
                return "alert_process"
            return "monitor_collect"
        
        def route_from_alert(state: OpsState) -> str:
            if state["errors"]:
                return "error_handling"
            return "diagnosis"
        
        def route_from_diagnosis(state: OpsState) -> str:
            if state["errors"]:
                return "error_handling"
            return "action_plan"
        
        def route_from_plan(state: OpsState) -> str:
            if state["errors"]:
                return "error_handling"
            return "action_execute"
        
        def route_from_execute(state: OpsState) -> str:
            if state["errors"]:
                return "error_handling"
            return "report_generate"
        
        def route_from_report(state: OpsState) -> str:
            if state["errors"]:
                return "error_handling"
            return "feedback_learn"
        
        def route_from_feedback(state: OpsState) -> str:
            return "monitor_collect"
        
        def route_from_error(state: OpsState) -> str:
            if state["workflow_status"] == "failed":
                return "END"
            return state["workflow_stage"]
        
        return {
            "monitor_collect": route_from_monitor,
            "alert_process": route_from_alert,
            "diagnosis": route_from_diagnosis,
            "action_plan": route_from_plan,
            "action_execute": route_from_execute,
            "report_generate": route_from_report,
            "feedback_learn": route_from_feedback,
            "error_handling": route_from_error
        }