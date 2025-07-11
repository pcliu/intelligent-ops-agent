"""
智能运维智能体主类

基于 LangGraph 的智能运维智能体实现
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Annotated
from datetime import datetime
from dataclasses import dataclass
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from src.dspy_modules.alert_analyzer import AlertInfo, AlertAnalyzer
from src.dspy_modules.diagnostic_agent import DiagnosticAgent
from src.dspy_modules.action_planner import ActionPlanner
from src.dspy_modules.report_generator import ReportGenerator
from src.dspy_modules.task_router import TaskRouter
from src.dspy_modules.natural_language_understanding import NaturalLanguageUnderstanding
from src.utils.llm_config import setup_deepseek_llm, get_llm_config_from_env


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


class ChatState(TypedDict):
    """智能运维智能体统一状态 - 支持聊天模式和任务模式"""
    # LangGraph 标准聊天字段
    messages: Annotated[List[BaseMessage], add_messages]  # 聊天消息列表
    
    # 智能运维业务字段
    alert_info: Optional[AlertInfo]  # 告警信息
    symptoms: Optional[List[str]]  # 症状列表
    context: Optional[Dict[str, Any]]  # 上下文信息
    analysis_result: Optional[Dict[str, Any]]  # 告警分析结果
    diagnostic_result: Optional[Dict[str, Any]]  # 诊断结果
    action_plan: Optional[Dict[str, Any]]  # 行动计划
    execution_result: Optional[Dict[str, Any]]  # 执行结果
    report: Optional[Dict[str, Any]]  # 报告
    
    # 自然语言理解相关字段
    raw_input: Optional[str]  # 原始用户输入
    parsed_intent: Optional[str]  # 解析的用户意图
    extracted_info: Optional[Dict[str, Any]]  # 从自然语言中提取的结构化信息
    nlu_confidence: Optional[float]  # 自然语言理解的置信度
    nlu_reasoning: Optional[str]  # 自然语言理解的推理过程
    
    # 处理状态字段
    current_task: Optional[str]  # 当前任务类型
    status: Optional[str]  # 处理状态：idle, processing, completed, failed
    stage: Optional[str]  # 处理阶段：input, analyze, execute, output
    errors: Optional[List[str]]  # 错误列表
    
    # 智能体基本信息
    agent_id: Optional[str]
    agent_type: Optional[str]
    specialization: Optional[str]
    
    # 历史和学习
    incident_history: Optional[List[Dict[str, Any]]]
    learning_data: Optional[Dict[str, Any]]
    
    # 元数据
    start_time: Optional[datetime]
    last_update: Optional[datetime]
    workflow_id: Optional[str]


class IntelligentOpsAgent:
    """智能运维智能体
    
    基于 LangGraph 的智能运维智能体，将智能体本身实现为一个状态图
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # 初始化 LLM
        self._initialize_llm()
        
        # DSPy 模块 (所有模块共享同一个 LLM 实例)
        self.alert_analyzer = AlertAnalyzer()
        self.diagnostic_agent = DiagnosticAgent()
        self.action_planner = ActionPlanner()
        self.report_generator = ReportGenerator()
        self.task_router = TaskRouter()
        self.nlu = NaturalLanguageUnderstanding()
        
        # 构建智能体图
        self.graph = self._build_agent_graph()
        self.compiled_graph = None
        
        print(f"✅ 智能体图构建完成: {config.agent_id}")
    
    def _initialize_llm(self) -> None:
        """初始化 LLM，失败时抛出异常"""
        llm_config = get_llm_config_from_env()
        self.dspy_lm, self.langchain_llm = setup_deepseek_llm(llm_config)
        print(f"✅ LLM 初始化成功: {llm_config.provider} - {llm_config.model_name}")
    
    def _build_agent_graph(self) -> StateGraph:
        """构建智能体状态图"""
        # 创建状态图
        agent_graph = StateGraph(ChatState)
        
        # 添加节点
        agent_graph.add_node("initialize", self._initialize_node)
        agent_graph.add_node("understand_input", self._understand_input_node)
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
        agent_graph.add_edge("initialize", "understand_input")
        agent_graph.add_edge("understand_input", "route_task")
        
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
        
        # 连续工作流：告警处理 → 问题诊断 → 行动规划 → 执行行动 → 生成报告
        agent_graph.add_conditional_edges(
            "process_alert",
            self._process_alert_completion_condition,
            {
                "diagnose_issue": "diagnose_issue",
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        agent_graph.add_conditional_edges(
            "diagnose_issue",
            self._diagnose_completion_condition,
            {
                "plan_actions": "plan_actions",
                "finalize": "finalize", 
                "error": "error_handler"
            }
        )
        
        agent_graph.add_conditional_edges(
            "plan_actions",
            self._plan_completion_condition,
            {
                "execute_actions": "execute_actions",
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        agent_graph.add_conditional_edges(
            "execute_actions",
            self._execute_completion_condition,
            {
                "generate_report": "generate_report",
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        # 独立任务节点（直接结束）
        for task_node in ["generate_report", "learn_feedback"]:
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
    
    async def _initialize_node(self, state: ChatState) -> ChatState:
        """初始化节点 - 设置处理状态"""
        return {
            **state,
            "agent_id": self.config.agent_id,
            "stage": "initialize",
            "status": "processing",
            "workflow_id": f"{self.config.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "last_update": datetime.now()
        }
    
    async def _understand_input_node(self, state: ChatState) -> ChatState:
        """自然语言理解节点"""
        try:
            # 从消息中提取原始输入
            messages = state.get("messages", [])
            if not messages:
                return {
                    **state,
                    "stage": "input_understood",
                    "last_update": datetime.now()
                }
            
            # 获取最后一条用户消息
            last_message = messages[-1]
            
            # 处理不同类型的消息内容
            if hasattr(last_message, 'content'):
                content = last_message.content
                # 如果 content 是列表（如包含 text 对象），提取文本
                if isinstance(content, list):
                    raw_input = ""
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            raw_input += item['text']
                        elif isinstance(item, str):
                            raw_input += item
                elif isinstance(content, str):
                    raw_input = content
                else:
                    raw_input = str(content)
            else:
                raw_input = None
            
            if not raw_input or not raw_input.strip():
                return {
                    **state,
                    "stage": "input_understood", 
                    "last_update": datetime.now()
                }
            
            print(f"🧠 开始自然语言理解: {raw_input[:50]}...")
            
            # 使用 NLU 模块处理输入
            nlu_result = await asyncio.to_thread(self.nlu.forward, raw_input)
            
            # 更新状态
            updated_state = {
                **state,
                "stage": "input_understood",
                "parsed_intent": nlu_result.intent,
                "nlu_confidence": nlu_result.confidence,
                "nlu_reasoning": nlu_result.reasoning,
                "extracted_info": nlu_result.extracted_info,
                "last_update": datetime.now()
            }
            
            # 如果 NLU 提取了告警信息，更新 alert_info
            if nlu_result.alert_info:
                # 转换为 AlertInfo 对象
                alert_info = AlertInfo(**nlu_result.alert_info)
                updated_state["alert_info"] = alert_info
                print(f"📊 提取到告警信息: {alert_info.message}")
            
            # 如果 NLU 提取了症状，更新 symptoms
            if nlu_result.symptoms:
                updated_state["symptoms"] = nlu_result.symptoms
                print(f"🔍 提取到症状: {nlu_result.symptoms}")
            
            # 如果 NLU 提取了上下文，更新 context
            if nlu_result.context:
                existing_context = state.get("context", {}) or {}
                updated_state["context"] = {**existing_context, **nlu_result.context}
                print(f"📋 提取到上下文: {nlu_result.context}")
            
            print(f"✅ 自然语言理解完成 - 意图: {nlu_result.intent}, 置信度: {nlu_result.confidence:.2f}")
            
            return updated_state
            
        except Exception as e:
            return self._create_error_state(state, e, "understand_input")
    
    async def _route_task_node(self, state: ChatState) -> ChatState:
        """任务路由节点 - 智能决策执行路径"""
        try:
            print("🎯 开始任务路由...")
            
            # 1. 检查是否有明确指定的任务类型（向后兼容）
            current_task = state.get("current_task")
            valid_tasks = ["process_alert", "diagnose_issue", "plan_actions", 
                          "execute_actions", "generate_report", "learn_feedback"]
            
            if current_task and current_task in valid_tasks:
                self._log_routing_decision("explicit", current_task)
                return {
                    **state,
                    "stage": "routing",
                    "current_task": current_task,
                    "routing_method": "explicit",
                    "last_update": datetime.now()
                }
            
            # 2. 尝试使用 NLU 结果进行智能路由
            nlu_intent = state.get("parsed_intent")
            nlu_confidence = state.get("nlu_confidence", 0.0)
            
            if nlu_intent and nlu_confidence > 0.6:
                # 将 NLU 意图映射到具体任务
                task_mapping = {
                    "process_alert": "process_alert",
                    "diagnose_issue": "diagnose_issue", 
                    "plan_actions": "plan_actions",
                    "execute_actions": "execute_actions",
                    "generate_report": "generate_report",
                    "learn_feedback": "learn_feedback"
                }
                
                mapped_task = task_mapping.get(nlu_intent, nlu_intent)
                if mapped_task in valid_tasks:
                    self._log_routing_decision("nlu", mapped_task, nlu_confidence)
                    return {
                        **state,
                        "stage": "routing",
                        "current_task": mapped_task,
                        "routing_method": "nlu",
                        "routing_confidence": nlu_confidence,
                        "last_update": datetime.now()
                    }
            
            # 3. 使用 DSPy 智能路由作为备选
            try:
                # 提取用户输入用于路由判断
                user_input = self._extract_user_input_for_routing(state)
                
                # 执行智能路由
                routing_result = await asyncio.to_thread(self.task_router.forward, user_input)
                
                # 检查置信度
                if routing_result.confidence > 0.6:
                    self._log_routing_decision("dspy", routing_result.task_type, 
                                             routing_result.confidence, routing_result.reasoning)
                    return {
                        **state,
                        "stage": "routing",
                        "current_task": routing_result.task_type,
                        "routing_method": "dspy",
                        "routing_confidence": routing_result.confidence,
                        "routing_reasoning": routing_result.reasoning,
                        "last_update": datetime.now()
                    }
                else:
                    print(f"⚠️ DSPy 路由置信度不足 ({routing_result.confidence:.2f})")
                    
            except Exception as e:
                print(f"❌ DSPy 路由失败: {str(e)}")
                print(f"🔍 DSPy 错误详情: {repr(e)}")
                # 记录 DSPy 路由失败信息到状态中
                current_errors = state.get("errors", [])
                current_errors.append(f"DSPy routing failed: {str(e)}")
            
            # 4. 回退到基于规则的路由
            rule_based_task = self._rule_based_routing(state)
            self._log_routing_decision("rule_based", rule_based_task)
            
            return {
                **state,
                "stage": "routing",
                "current_task": rule_based_task,
                "routing_method": "rule_based",
                "last_update": datetime.now()
            }
            
        except Exception as e:
            context = {
                "current_task": state.get("current_task"),
                "parsed_intent": state.get("parsed_intent"),
                "nlu_confidence": state.get("nlu_confidence")
            }
            return self._create_error_state(state, e, "route_task", context)
    
    async def _process_alert_node(self, state: ChatState) -> ChatState:
        """处理告警节点"""
        try:
            alert_info_dict = state.get("alert_info")
            if not alert_info_dict:
                raise ValueError("No alert information provided")
            
            # 确保alert_info是AlertInfo对象
            if isinstance(alert_info_dict, dict):
                from src.dspy_modules.alert_analyzer import AlertInfo
                alert_info = AlertInfo(**alert_info_dict)
            else:
                alert_info = alert_info_dict
            
            # 分析告警
            # 获取历史告警信息，转换为 AlertInfo 格式
            historical_incidents = state.get("incident_history", [])
            historical_alerts = []
            for incident in historical_incidents:
                if isinstance(incident, dict) and "alert_info" in incident:
                    incident_alert = incident["alert_info"]
                    if isinstance(incident_alert, dict):
                        historical_alerts.append(AlertInfo(**incident_alert))
                    else:
                        historical_alerts.append(incident_alert)
            
            # 使用 asyncio.to_thread 处理同步的 DSPy 调用
            import asyncio
            analysis_result = await asyncio.to_thread(
                self.alert_analyzer.forward,
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
    
    async def _diagnose_issue_node(self, state: ChatState) -> ChatState:
        """诊断问题节点"""
        try:
            symptoms = state.get("symptoms", [])
            context = state.get("context", {})
            
            # 创建诊断上下文
            from src.dspy_modules.diagnostic_agent import DiagnosticContext
            from src.dspy_modules.alert_analyzer import AlertAnalysisResult
            
            # 使用现有的告警分析结果，或创建基于症状的分析结果
            analysis_result = state.get("analysis_result")
            if analysis_result:
                # 使用告警分析结果
                alert_info_dict = state.get("alert_info", {})
                if isinstance(alert_info_dict, dict):
                    alert_id = alert_info_dict.get("alert_id", "unknown")
                else:
                    alert_id = "unknown"
                    
                alert_analysis = AlertAnalysisResult(
                    alert_id=alert_id,
                    priority=analysis_result.get("priority", "medium"),
                    category=analysis_result.get("category", "investigation"), 
                    urgency_score=analysis_result.get("urgency_score", 0.5),
                    root_cause_hints=analysis_result.get("root_cause_hints", symptoms),
                    recommended_actions=analysis_result.get("recommended_actions", [])
                )
            else:
                # 基于症状创建分析结果
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
                system_metrics=context.get("system_metrics", {}) if context else {},
                log_entries=context.get("log_entries", []) if context else [],
                historical_incidents=state.get("incident_history", []),
                topology_info=context.get("topology_info", {}) if context else {}
            )
            
            # 执行诊断 - 使用 asyncio.to_thread 处理同步调用
            diagnostic_result = await asyncio.to_thread(
                self.diagnostic_agent.forward,
                diagnostic_context
            )
            
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
    
    async def _plan_actions_node(self, state: ChatState) -> ChatState:
        """规划行动节点"""
        try:
            diagnostic_result = state.get("diagnostic_result")
            if not diagnostic_result:
                raise ValueError("No diagnostic result available")
            
            # 转换诊断结果
            from src.dspy_modules.diagnostic_agent import DiagnosticResult
            
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
            
            # 生成行动计划 - 使用 asyncio.to_thread 处理同步调用
            context = state.get("context") or {}
            action_plan = await asyncio.to_thread(
                self.action_planner.forward,
                diag_result, 
                context
            )
            
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
    
    async def _execute_actions_node(self, state: ChatState) -> ChatState:
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
    
    async def _generate_report_node(self, state: ChatState) -> ChatState:
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
            # TODO: 实现 ReportGenerator 的调用
            report = {
                "incident_id": state.get("workflow_id", "unknown"),
                "title": f"智能运维报告 - {state['agent_id']}",
                "summary": f"任务完成: {state.get('current_task', 'unknown')}",
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
    
    async def _learn_feedback_node(self, state: ChatState) -> ChatState:
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
    
    async def _finalize_node(self, state: ChatState) -> ChatState:
        """完成节点 - 支持聊天模式和任务模式"""
        # 检查是否有聊天消息（聊天模式）
        messages = state.get("messages", [])
        if messages and not any(msg.type == "ai" for msg in messages):
            # 聊天模式：创建 AI 回复消息
            from langchain_core.messages import AIMessage
            
            # 构建回复内容
            response_text = f"""🤖 **智能运维助手**

📊 **任务状态**: {'成功' if not state.get("errors") else '完成但有错误'}
🎯 **任务类型**: {state.get('current_task', '自动分析')}

📋 **处理结果**:
"""
            
            # 添加具体的处理结果
            if state.get('parsed_intent'):
                response_text += f"🧠 **识别意图**: {state.get('parsed_intent')}\n"
                response_text += f"📈 **置信度**: {state.get('nlu_confidence', 0):.2f}\n"
            
            if state.get('alert_info'):
                alert_info = state.get('alert_info')
                message = alert_info.message if hasattr(alert_info, 'message') else 'N/A'
                response_text += f"📢 **告警**: {message}\n"
            
            if state.get('diagnostic_result'):
                diagnosis = state.get('diagnostic_result')
                response_text += f"🩺 **诊断结果**: {diagnosis.get('root_cause', 'N/A')}\n"
            
            if state.get('action_plan'):
                plan = state.get('action_plan')
                response_text += f"📋 **行动计划**: {plan.get('steps', 'N/A')}\n"
            
            if state.get('errors'):
                response_text += f"⚠️ **错误**: {', '.join(state.get('errors'))}\n"
            
            response_text += f"\n⏰ **处理时间**: {datetime.now().strftime('%H:%M:%S')}"
            
            # 创建 AI 消息
            ai_message = AIMessage(content=response_text)
            
            return {
                **state,
                "messages": state.get("messages", []) + [ai_message],
                "status": "completed",
                "stage": "finalized",
                "last_update": datetime.now()
            }
        
        # 任务模式：返回结构化结果
        return {
            **state,
            "status": "completed",
            "stage": "finalized",
            "last_update": datetime.now()
        }
    
    async def _error_handler_node(self, state: ChatState) -> ChatState:
        """错误处理节点 - 支持聊天模式和任务模式"""
        errors = state.get("errors", [])
        
        # 检查是否有聊天消息（聊天模式）
        messages = state.get("messages", [])
        if messages:
            # 聊天模式：创建错误回复消息
            from langchain_core.messages import AIMessage
            
            error_text = f"""❌ **处理失败**

🚨 **错误信息**: {', '.join(errors) if errors else '未知错误'}

💡 **建议**:
- 请检查输入是否正确
- 尝试重新描述问题
- 或联系技术支持

⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"""
            
            error_message = AIMessage(content=error_text)
            
            return {
                **state,
                "messages": state.get("messages", []) + [error_message],
                "status": "failed",
                "stage": "error_handled",
                "last_update": datetime.now()
            }
        
        # 任务模式：返回结构化错误
        return {
            **state,
            "stage": "error_handling",
            "status": "failed",
            "last_update": datetime.now()
        }
    
    # ==================== 辅助函数 ====================
    
    def _create_error_state(self, state: ChatState, error: Exception, node_name: str, 
                           context: Dict[str, Any] = None) -> ChatState:
        """创建标准化的错误状态"""
        error_msg = f"{node_name} error: {str(error)}"
        print(f"❌ {node_name} 失败: {error_msg}")
        print(f"🔍 错误详情: {repr(error)}")
        
        error_details = {
            "node": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            error_details["context"] = context
        
        return {
            **state,
            "stage": "error",
            "errors": state.get("errors", []) + [error_msg],
            "error_details": error_details,
            "last_update": datetime.now()
        }
    
    def _log_routing_decision(self, method: str, task: str, confidence: float = None, 
                             reasoning: str = None):
        """记录路由决策日志"""
        log_msg = f"🎯 路由决策: {method} → {task}"
        if confidence is not None:
            log_msg += f" (置信度: {confidence:.2f})"
        print(log_msg)
        
        if reasoning:
            print(f"💭 推理过程: {reasoning}")
    
    # ==================== 条件函数 ====================
    
    def _route_task_condition(self, state: ChatState) -> str:
        """智能任务路由条件 - 简化版本，仅返回路由结果"""
        # 路由逻辑已经在 _route_task_node 中实现
        # 这里只需要返回 current_task 字段的值
        current_task = state.get("current_task")
        if current_task:
            return current_task
        
        # 默认回退
        return "process_alert"
    
    def _extract_user_input_for_routing(self, state: ChatState) -> Dict[str, Any]:
        """提取用户输入用于路由判断"""
        user_input = {}
        
        # 收集所有可能的输入信息
        if state.get("alert_info"):
            user_input["alert_info"] = state["alert_info"]
        
        if state.get("symptoms"):
            user_input["symptoms"] = state["symptoms"]
        
        if state.get("context"):
            user_input["context"] = state["context"]
        
        if state.get("diagnostic_result"):
            user_input["diagnostic_result"] = state["diagnostic_result"]
        
        if state.get("action_plan"):
            user_input["action_plan"] = state["action_plan"]
        
        # 任务输入中的描述信息
        task_input = state.get("task_input", {})
        if isinstance(task_input, dict):
            user_input.update(task_input)
        
        # 工作流ID作为上下文
        if state.get("workflow_id"):
            user_input["workflow_id"] = state["workflow_id"]
        
        return user_input
    
    def _rule_based_routing(self, state: ChatState) -> str:
        """基于规则的回退路由"""
        # 根据输入数据结构进行简单推断
        if state.get("alert_info"):
            return "process_alert"
        elif state.get("symptoms"):
            return "diagnose_issue"
        elif state.get("diagnostic_result"):
            return "plan_actions"
        elif state.get("action_plan"):
            return "execute_actions"
        elif state.get("task_input", {}).get("incident_id"):
            return "generate_report"
        elif state.get("task_input", {}).get("feedback"):
            return "learn_feedback"
        else:
            # 默认任务：处理告警
            print("🔄 使用默认任务类型: process_alert")
            return "process_alert"
    
    def _task_completion_condition(self, state: ChatState) -> str:
        """任务完成条件"""
        if state.get("errors"):
            return "error"
        else:
            return "finalize"
    
    def _process_alert_completion_condition(self, state: ChatState) -> str:
        """告警处理完成条件"""
        if state.get("errors"):
            return "error"
        elif state.get("analysis_result"):
            # 告警分析完成，继续诊断
            return "diagnose_issue"
        else:
            return "finalize"
    
    def _diagnose_completion_condition(self, state: ChatState) -> str:
        """诊断完成条件"""
        if state.get("errors"):
            return "error"
        elif state.get("diagnostic_result"):
            # 诊断完成，继续规划行动
            return "plan_actions"
        else:
            return "finalize"
    
    def _plan_completion_condition(self, state: ChatState) -> str:
        """行动规划完成条件"""
        if state.get("errors"):
            return "error"
        elif state.get("action_plan") and self.config.auto_execution:
            # 规划完成且允许自动执行，继续执行
            return "execute_actions"
        else:
            # 规划完成但不自动执行，直接完成
            return "finalize"
    
    def _execute_completion_condition(self, state: ChatState) -> str:
        """执行完成条件"""
        if state.get("errors"):
            return "error"
        elif state.get("execution_result") and self.config.enable_reporting:
            # 执行完成且启用报告，生成报告
            return "generate_report"
        else:
            return "finalize"
    
    def _error_recovery_condition(self, state: ChatState) -> str:
        """错误恢复条件"""
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.config.max_retries)
        
        if retry_count < max_retries:
            return "retry"
        else:
            return "END"
    
    # ==================== 公共接口 ====================
    
    async def process_input(self, user_input: Any) -> Dict[str, Any]:
        """智能处理用户输入
        
        这是主要的公共接口，支持任意格式的输入，自动判断任务类型并处理。
        
        Args:
            user_input: 用户输入，可以是：
                - 告警信息字典
                - 症状列表
                - 自然语言描述
                - 诊断结果
                - 行动计划
                - 任何其他格式
        
        Returns:
            Dict[str, Any]: 处理结果
        """
        # 将用户输入标准化为状态参数
        state_kwargs = self._normalize_user_input(user_input)
        
        # 创建初始状态（自动判断任务类型）
        initial_state = self._create_initial_state(**state_kwargs)
        
        # 运行智能体图
        return await self._run_agent_task(initial_state)
    
    def _normalize_user_input(self, user_input: Any) -> Dict[str, Any]:
        """标准化用户输入为状态参数"""
        if isinstance(user_input, dict):
            # 字典格式：检查并补充必要字段
            normalized = user_input.copy()
            
            # 如果包含 alert_info，确保格式正确
            if "alert_info" in normalized:
                alert_info = normalized["alert_info"]
                if isinstance(alert_info, dict):
                    # 补充缺失的必要字段
                    if "timestamp" not in alert_info:
                        alert_info["timestamp"] = datetime.now().isoformat()
                    if "tags" not in alert_info:
                        alert_info["tags"] = []
                    if "metrics" not in alert_info:
                        alert_info["metrics"] = {}
            
            return normalized
        elif isinstance(user_input, str):
            # 字符串格式：作为自然语言输入处理
            return {
                "raw_input": user_input,  # 保存原始输入用于 NLU 处理
                "message": user_input     # 保持兼容性
            }
        elif isinstance(user_input, list):
            # 列表格式：假设是症状列表
            return {"symptoms": user_input}
        elif hasattr(user_input, "__dict__"):
            # 对象格式：转换为字典
            return user_input.__dict__
        else:
            # 其他格式：作为自然语言描述处理
            str_input = str(user_input)
            return {
                "raw_input": str_input,
                "description": str_input
            }
    
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
    
    def _create_initial_state(self, task: Optional[str] = None, **kwargs) -> ChatState:
        """创建初始状态
        
        Args:
            task: 任务类型，可选。如果不提供，将通过 DSPy 智能路由自动判断
            **kwargs: 其他状态参数
        """
        now = datetime.now()
        
        # 如果没有指定任务类型，先创建临时状态用于路由判断
        if not task:
            # 创建临时状态用于智能路由
            temp_state = ChatState(
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                specialization=self.config.specialization,
                current_task=None,  # 留空，让路由器判断
                task_input=kwargs,
                task_output=None,
                status="idle",
                stage="input",
                alert_info=kwargs.get("alert_info"),
                symptoms=kwargs.get("symptoms"),
                context=kwargs.get("context"),
                # 自然语言理解相关字段
                raw_input=kwargs.get("raw_input"),
                parsed_intent=kwargs.get("parsed_intent"),
                extracted_info=kwargs.get("extracted_info"),
                nlu_confidence=kwargs.get("nlu_confidence"),
                nlu_reasoning=kwargs.get("nlu_reasoning"),
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
                workflow_id=f"{self.config.agent_id}_auto_{now.strftime('%Y%m%d_%H%M%S')}"
            )
            
            # 使用智能路由判断任务类型
            task = self._route_task_condition(temp_state)
            print(f"🎯 自动判断任务类型: {task}")
        
        # 创建最终的工作流ID
        workflow_id = f"{self.config.agent_id}_{task}_{now.strftime('%Y%m%d_%H%M%S')}"
        
        return ChatState(
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
            # 自然语言理解相关字段
            raw_input=kwargs.get("raw_input"),
            parsed_intent=kwargs.get("parsed_intent"),
            extracted_info=kwargs.get("extracted_info"),
            nlu_confidence=kwargs.get("nlu_confidence"),
            nlu_reasoning=kwargs.get("nlu_reasoning"),
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
    
    async def _run_agent_task(self, initial_state: ChatState) -> Dict[str, Any]:
        """运行智能体任务"""
        try:
            if not self.compiled_graph:
                self.compile()
            
            # 运行智能体图
            final_state = await self.compiled_graph.ainvoke(
                initial_state,
                config={"recursion_limit": self.config.max_retries * 5}
            )
            
            # 检查返回状态
            if final_state is None:
                return {
                    "status": "error",
                    "error": "智能体图返回空状态",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 返回任务输出
            task_output = final_state.get("task_output")
            if task_output is None:
                # 如果没有 task_output，创建一个默认的
                return {
                    "status": "completed",
                    "task_type": final_state.get("current_task", "unknown"),
                    "results": {
                        "analysis": final_state.get("analysis_result"),
                        "diagnosis": final_state.get("diagnostic_result"),
                        "action_plan": final_state.get("action_plan"),
                        "execution": final_state.get("execution_result"),
                        "report": final_state.get("report")
                    },
                    "errors": final_state.get("errors", []),
                    "timestamp": datetime.now().isoformat()
                }
            return task_output
            
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


# ==================== LangGraph Studio 集成 ====================

# 创建默认的智能体配置供 Studio 使用
_default_studio_config = AgentConfig(
    agent_id="ops_agent_studio",
    agent_type="general", 
    specialization="studio_demo",
    enable_learning=True,
    enable_reporting=True,
    auto_execution=False
)

# 创建智能体实例并编译图
_studio_agent = IntelligentOpsAgent(_default_studio_config)
graph = _studio_agent.compile()

# 同时导出智能体实例供其他用途
agent = _studio_agent