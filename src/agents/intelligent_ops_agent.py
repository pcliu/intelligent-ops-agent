"""
智能运维智能体主类

基于 LangGraph 的智能运维智能体实现
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
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


class AgentState(TypedDict):
    """智能体状态"""
    # 智能体基本信息
    agent_id: str
    agent_type: str
    specialization: Optional[str]
    
    # 当前任务信息
    current_task: Optional[str]  # process_alert, diagnose_issue, plan_actions, etc.
    task_input: Optional[Dict[str, Any]]
    task_output: Optional[Dict[str, Any]]
    
    # 处理状态
    status: str  # idle, processing, completed, failed
    stage: str  # input, analyze, execute, output
    
    # 数据流
    alert_info: Optional[AlertInfo]
    symptoms: Optional[List[str]]
    context: Optional[Dict[str, Any]]
    
    # 中间结果
    analysis_result: Optional[Dict[str, Any]]
    diagnostic_result: Optional[Dict[str, Any]]
    action_plan: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    report: Optional[Dict[str, Any]]
    
    # 历史和学习
    incident_history: List[Dict[str, Any]]
    learning_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    # 错误处理
    errors: List[str]
    retry_count: int
    max_retries: int
    
    # 元数据
    start_time: datetime
    last_update: datetime
    workflow_id: str


class IntelligentOpsAgent:
    """智能运维智能体
    
    基于 LangGraph 的智能运维智能体，将智能体本身实现为一个状态图
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
        
        # DSPy 模块
        self.alert_analyzer = AlertAnalyzer()
        self.diagnostic_agent = DiagnosticAgent()
        self.action_planner = ActionPlanner()
        self.report_generator = ReportGenerator()
        
        # 构建智能体图
        self.graph = self._build_agent_graph()
        self.compiled_graph = None
        
        print(f"✅ 智能体图构建完成: {config.agent_id}")
    
    def _build_agent_graph(self) -> StateGraph:
        """构建智能体状态图"""
        # 创建状态图
        agent_graph = StateGraph(AgentState)
        
        # 添加节点
        agent_graph.add_node("initialize", self._initialize_node)
        agent_graph.add_node("route_task", self._route_task_node)
        agent_graph.add_node("process_alert", self._process_alert_node)
        agent_graph.add_node("diagnose_issue", self._diagnose_issue_node)
        agent_graph.add_node("plan_actions", self._plan_actions_node)
        agent_graph.add_node("execute_actions", self._execute_actions_node)
        agent_graph.add_node("generate_report", self._generate_report_node)
        agent_graph.add_node("learn_feedback", self._learn_feedback_node)
        agent_graph.add_node("finalize", self._finalize_node)
        agent_graph.add_node("error_handler", self._error_handler_node)
        
        # 设置入口点
        agent_graph.set_entry_point("initialize")
        
        # 添加边
        agent_graph.add_edge("initialize", "route_task")
        
        # 条件边：根据任务类型路由
        agent_graph.add_conditional_edges(
            "route_task",
            self._route_task_condition,
            {
                "process_alert": "process_alert",
                "diagnose_issue": "diagnose_issue", 
                "plan_actions": "plan_actions",
                "execute_actions": "execute_actions",
                "generate_report": "generate_report",
                "learn_feedback": "learn_feedback",
                "error": "error_handler"
            }
        )
        
        # 各个任务节点完成后的路由
        for task_node in ["process_alert", "diagnose_issue", "plan_actions", 
                         "execute_actions", "generate_report", "learn_feedback"]:
            agent_graph.add_conditional_edges(
                task_node,
                self._task_completion_condition,
                {
                    "finalize": "finalize",
                    "error": "error_handler"
                }
            )
        
        # 错误处理
        agent_graph.add_conditional_edges(
            "error_handler",
            self._error_recovery_condition,
            {
                "retry": "route_task",
                "finalize": "finalize",
                "END": END
            }
        )
        
        # 结束节点
        agent_graph.add_edge("finalize", END)
        
        return agent_graph
    
    def compile(self):
        """编译智能体图"""
        if not self.compiled_graph:
            self.compiled_graph = self.graph.compile()
        return self.compiled_graph
        
    # ==================== 节点函数 ====================
    
    async def _initialize_node(self, state: AgentState) -> AgentState:
        """初始化节点"""
        now = datetime.now()
        return {
            **state,
            "status": "processing",
            "stage": "initialize",
            "start_time": now,
            "last_update": now,
            "errors": [],
            "retry_count": 0
        }
    
    async def _route_task_node(self, state: AgentState) -> AgentState:
        """任务路由节点"""
        return {
            **state,
            "stage": "routing",
            "last_update": datetime.now()
        }
    
    async def _process_alert_node(self, state: AgentState) -> AgentState:
        """处理告警节点"""
        try:
            alert_info = state.get("alert_info")
            if not alert_info:
                raise ValueError("No alert information provided")
            
            # 分析告警
            # 获取历史告警信息，转换为 AlertInfo 格式
            historical_incidents = state.get("incident_history", [])
            historical_alerts = []
            for incident in historical_incidents:
                if isinstance(incident, dict) and "alert_info" in incident:
                    historical_alerts.append(incident["alert_info"])
            
            analysis_result = self.alert_analyzer.forward(
                alert_info=alert_info,
                historical_alerts=historical_alerts
            )
            
            return {
                **state,
                "stage": "alert_processed",
                "analysis_result": {
                    "priority": analysis_result.priority,
                    "category": analysis_result.category,
                    "urgency_score": analysis_result.urgency_score,
                    "root_cause_hints": analysis_result.root_cause_hints,
                    "recommended_actions": analysis_result.recommended_actions
                },
                "last_update": datetime.now()
            }
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Alert processing error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    async def _diagnose_issue_node(self, state: AgentState) -> AgentState:
        """诊断问题节点"""
        try:
            symptoms = state.get("symptoms", [])
            context = state.get("context", {})
            
            # 创建诊断上下文
            from ..dspy_modules.diagnostic_agent import DiagnosticContext
            from ..dspy_modules.alert_analyzer import AlertAnalysisResult
            
            # 创建模拟告警分析结果
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
                historical_incidents=state.get("incident_history", []),
                topology_info=context.get("topology_info", {})
            )
            
            # 执行诊断
            diagnostic_result = self.diagnostic_agent.forward(diagnostic_context)
            
            return {
                **state,
                "stage": "diagnosed",
                "diagnostic_result": {
                    "root_cause": diagnostic_result.root_cause,
                    "confidence_score": diagnostic_result.confidence_score,
                    "impact_assessment": diagnostic_result.impact_assessment,
                    "affected_components": diagnostic_result.affected_components,
                    "business_impact": diagnostic_result.business_impact,
                    "recovery_estimate": diagnostic_result.recovery_time_estimate,
                    "similar_incidents": diagnostic_result.similar_incidents,
                    "evidence": diagnostic_result.evidence
                },
                "last_update": datetime.now()
            }
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Diagnosis error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    async def _plan_actions_node(self, state: AgentState) -> AgentState:
        """规划行动节点"""
        try:
            diagnostic_result = state.get("diagnostic_result")
            if not diagnostic_result:
                raise ValueError("No diagnostic result available")
            
            # 转换诊断结果
            from ..dspy_modules.diagnostic_agent import DiagnosticResult
            
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
            action_plan = self.action_planner.forward(diag_result, state.get("context", {}))
            
            return {
                **state,
                "stage": "planned",
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
                "last_update": datetime.now()
            }
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Action planning error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    async def _execute_actions_node(self, state: AgentState) -> AgentState:
        """执行行动节点"""
        try:
            action_plan = state.get("action_plan")
            if not action_plan:
                raise ValueError("No action plan available")
            
            if not self.config.auto_execution:
                return {
                    **state,
                    "stage": "executed",
                    "execution_result": {
                        "status": "manual_approval_required",
                        "message": "Automatic execution is disabled. Manual approval required.",
                        "plan_id": action_plan.get("plan_id", "unknown")
                    },
                    "last_update": datetime.now()
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
                **state,
                "stage": "executed",
                "execution_result": {
                    "status": execution_status,
                    "plan_id": action_plan.get("plan_id", "unknown"),
                    "executed_steps": executed_steps,
                    "failed_steps": failed_steps,
                    "execution_time": datetime.now().isoformat()
                },
                "last_update": datetime.now()
            }
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Execution error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    async def _generate_report_node(self, state: AgentState) -> AgentState:
        """生成报告节点"""
        try:
            if not self.config.enable_reporting:
                return {
                    **state,
                    "stage": "reported",
                    "report": {
                        "status": "disabled",
                        "message": "Reporting is disabled for this agent"
                    },
                    "last_update": datetime.now()
                }
            
            # 生成报告
            report = {
                "incident_id": state.get("workflow_id", "unknown"),
                "title": f"Agent {state['agent_id']} Task Report",
                "summary": f"Completed task: {state.get('current_task', 'unknown')}",
                "timestamp": datetime.now().isoformat(),
                "agent_id": state["agent_id"],
                "status": "generated",
                "results": {
                    "analysis": state.get("analysis_result"),
                    "diagnosis": state.get("diagnostic_result"),
                    "action_plan": state.get("action_plan"),
                    "execution": state.get("execution_result")
                }
            }
            
            return {
                **state,
                "stage": "reported",
                "report": report,
                "last_update": datetime.now()
            }
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Report generation error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    async def _learn_feedback_node(self, state: AgentState) -> AgentState:
        """学习反馈节点"""
        try:
            if not self.config.enable_learning:
                return {
                    **state,
                    "stage": "learned",
                    "last_update": datetime.now()
                }
            
            # 从任务输入中获取反馈数据
            feedback = state.get("task_input", {})
            
            # 更新学习数据
            updated_learning_data = {
                **state.get("learning_data", {}),
                **feedback,
                "last_feedback_time": datetime.now().isoformat()
            }
            
            # 更新历史记录
            updated_history = state.get("incident_history", [])
            if state.get("report"):
                updated_history.append({
                    "incident_id": state["workflow_id"],
                    "timestamp": datetime.now().isoformat(),
                    "task_type": state.get("current_task"),
                    "results": state.get("task_output", {})
                })
            
            return {
                **state,
                "stage": "learned",
                "learning_data": updated_learning_data,
                "incident_history": updated_history,
                "last_update": datetime.now()
            }
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Learning error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    async def _finalize_node(self, state: AgentState) -> AgentState:
        """完成节点"""
        return {
            **state,
            "status": "completed",
            "stage": "finalized",
            "task_output": {
                "status": "success" if not state.get("errors") else "completed_with_errors",
                "task_type": state.get("current_task"),
                "results": {
                    "analysis": state.get("analysis_result"),
                    "diagnosis": state.get("diagnostic_result"),
                    "action_plan": state.get("action_plan"),
                    "execution": state.get("execution_result"),
                    "report": state.get("report")
                },
                "errors": state.get("errors", []),
                "timestamp": datetime.now().isoformat()
            },
            "last_update": datetime.now()
        }
    
    async def _error_handler_node(self, state: AgentState) -> AgentState:
        """错误处理节点"""
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.config.max_retries)
        
        if retry_count < max_retries:
            return {
                **state,
                "stage": "error_handling",
                "retry_count": retry_count + 1,
                "status": "retrying",
                "last_update": datetime.now()
            }
        else:
            return {
                **state,
                "stage": "error_handling",
                "status": "failed",
                "task_output": {
                    "status": "failed",
                    "errors": errors,
                    "retry_count": retry_count,
                    "timestamp": datetime.now().isoformat()
                },
                "last_update": datetime.now()
            }
    
    # ==================== 条件函数 ====================
    
    def _route_task_condition(self, state: AgentState) -> str:
        """任务路由条件"""
        current_task = state.get("current_task")
        
        if not current_task:
            return "error"
            
        # 根据任务类型路由
        valid_tasks = ["process_alert", "diagnose_issue", "plan_actions", 
                      "execute_actions", "generate_report", "learn_feedback"]
        
        if current_task in valid_tasks:
            return current_task
        else:
            return "error"
    
    def _task_completion_condition(self, state: AgentState) -> str:
        """任务完成条件"""
        if state.get("errors"):
            return "error"
        else:
            return "finalize"
    
    def _error_recovery_condition(self, state: AgentState) -> str:
        """错误恢复条件"""
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.config.max_retries)
        
        if retry_count < max_retries:
            return "retry"
        else:
            return "END"
    
    # ==================== 公共接口 ====================
    
    async def process_alert(self, alert: Union[AlertInfo, Dict[str, Any]]) -> Dict[str, Any]:
        """处理告警"""
        # 转换告警格式
        if isinstance(alert, dict):
            alert_info = AlertInfo(**alert)
        else:
            alert_info = alert
        
        # 创建初始状态
        initial_state = self._create_initial_state(
            task="process_alert",
            alert_info=alert_info
        )
        
        # 运行智能体图
        return await self._run_agent_task(initial_state)
    
    async def diagnose_issue(self, symptoms: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """诊断问题"""
        # 创建初始状态
        initial_state = self._create_initial_state(
            task="diagnose_issue",
            symptoms=symptoms,
            context=context
        )
        
        # 运行智能体图
        return await self._run_agent_task(initial_state)
    
    async def plan_actions(self, diagnostic_result: Dict[str, Any], 
                          system_context: Dict[str, Any]) -> Dict[str, Any]:
        """规划行动"""
        # 创建初始状态
        initial_state = self._create_initial_state(
            task="plan_actions",
            diagnostic_result=diagnostic_result,
            context=system_context
        )
        
        # 运行智能体图
        return await self._run_agent_task(initial_state)
    
    async def execute_actions(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行行动"""
        # 创建初始状态
        initial_state = self._create_initial_state(
            task="execute_actions",
            action_plan=action_plan
        )
        
        # 运行智能体图
        return await self._run_agent_task(initial_state)
    
    async def generate_report(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        # 创建初始状态
        initial_state = self._create_initial_state(
            task="generate_report",
            task_input=incident_data
        )
        
        # 运行智能体图
        return await self._run_agent_task(initial_state)
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """从反馈中学习"""
        # 创建初始状态
        initial_state = self._create_initial_state(
            task="learn_feedback",
            task_input=feedback
        )
        
        # 运行智能体图
        return await self._run_agent_task(initial_state)
    
    # ==================== 辅助方法 ====================
    
    def _create_initial_state(self, task: str, **kwargs) -> AgentState:
        """创建初始状态"""
        now = datetime.now()
        workflow_id = f"{self.config.agent_id}_{task}_{now.strftime('%Y%m%d_%H%M%S')}"
        
        return AgentState(
            agent_id=self.config.agent_id,
            agent_type=self.config.agent_type,
            specialization=self.config.specialization,
            current_task=task,
            task_input=kwargs,
            task_output=None,
            status="idle",
            stage="input",
            alert_info=kwargs.get("alert_info"),
            symptoms=kwargs.get("symptoms"),
            context=kwargs.get("context"),
            analysis_result=None,
            diagnostic_result=kwargs.get("diagnostic_result"),
            action_plan=kwargs.get("action_plan"),
            execution_result=None,
            report=None,
            incident_history=[],
            learning_data={},
            performance_metrics={},
            errors=[],
            retry_count=0,
            max_retries=self.config.max_retries,
            start_time=now,
            last_update=now,
            workflow_id=workflow_id
        )
    
    async def _run_agent_task(self, initial_state: AgentState) -> Dict[str, Any]:
        """运行智能体任务"""
        try:
            if not self.compiled_graph:
                self.compile()
            
            # 运行智能体图
            final_state = await self.compiled_graph.ainvoke(
                initial_state,
                config={"recursion_limit": self.config.max_retries * 5}
            )
            
            # 返回任务输出
            return final_state.get("task_output", {
                "status": "completed",
                "results": final_state,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # ==================== 状态和指标 ====================
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "specialization": self.config.specialization,
            "status": "ready",
            "graph_compiled": self.compiled_graph is not None,
            "learning_enabled": self.config.enable_learning,
            "reporting_enabled": self.config.enable_reporting,
            "auto_execution_enabled": self.config.auto_execution,
            "last_update": datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "agent_id": self.config.agent_id,
            "incidents_processed": 0,  # 实际应该从状态中获取
            "average_resolution_time": 300.0,  # 模拟值
            "success_rate": 0.85,  # 模拟值
            "learning_data_points": 0,  # 实际应该从状态中获取
            "timestamp": datetime.now().isoformat()
        }


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