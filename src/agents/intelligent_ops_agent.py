"""
智能运维智能体主类

基于 LangGraph 的智能运维智能体实现
"""

import asyncio
from typing import Dict, Any, List, Optional, Annotated
from datetime import datetime
from dataclasses import dataclass
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
# from langchain_core.tools import tool  # 不再需要，中断函数是普通函数
from langgraph.types import interrupt
from src.dspy_modules.alert_analyzer import AlertInfo, AlertAnalyzer
from src.dspy_modules.diagnostic_agent import DiagnosticAgent
from src.dspy_modules.action_planner import ActionPlanner
from src.dspy_modules.task_router import TaskRouter
from src.dspy_modules.report_generator import ReportGenerator
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
    enable_reporting: bool = True
    auto_execution: bool = False


class ChatState(TypedDict):
    """智能运维智能体核心状态 - 仅包含跨节点必需的数据"""
    # LangGraph 标准聊天字段
    messages: Annotated[List[BaseMessage], add_messages]  # 聊天消息列表
    
    # 跨节点业务数据
    alert_info: Optional[AlertInfo]  # 告警信息
    symptoms: Optional[List[str]]  # 症状列表
    context: Optional[Dict[str, Any]]  # 上下文信息
    extracted_info: Optional[Dict[str, Any]]  # 从自然语言中提取的结构化信息
    
    # 节点间传递的成果
    analysis_result: Optional[Dict[str, Any]]  # 告警分析结果
    diagnostic_result: Optional[Dict[str, Any]]  # 诊断结果
    action_plan: Optional[Dict[str, Any]]  # 行动计划
    execution_result: Optional[Dict[str, Any]]  # 执行结果
    report: Optional[Dict[str, Any]]  # 报告
    
    # 工作流控制
    current_task: Optional[str]  # 当前任务类型
    errors: Optional[List[str]]  # 错误列表
    workflow_id: Optional[str]  # 工作流标识


# ==================== 人类干预工具 ====================

def request_operator_input(query: str, context: dict = None) -> str:
    """请求运维人员输入和确认
    
    Args:
        query: 向运维人员提出的问题或请求
        context: 相关上下文信息
        
    Returns:
        str: 运维人员的回复
    """
    interrupt_data = {
        "query": query,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
        "type": "operator_input",
        "ai_message": f"🤖 **需要您的输入**\n\n💬 **问题**: {query}\n\n⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
    }
    human_response = interrupt(interrupt_data)
    
    # 处理不同类型的返回值 - LangGraph Studio 返回格式兼容
    # Studio Web UI 返回格式: {random_id: {'response': 'user_input'}}
    if isinstance(human_response, dict):
        # 首先尝试直接键名
        result = (human_response.get("response") or 
                 human_response.get("input") or 
                 human_response.get("value") or 
                 human_response.get("content"))
        
        if result:
            return result
        
        # 如果直接键名不匹配，尝试嵌套字典格式 {random_id: {'response': 'content'}}
        for key, value in human_response.items():
            if isinstance(value, dict):
                nested_result = (value.get("response") or 
                               value.get("input") or 
                               value.get("value") or 
                               value.get("content"))
                if nested_result:
                    return nested_result
            elif isinstance(value, str) and value.strip():
                return value
        
        # 如果都没有匹配到，返回字典的字符串表示
        return str(human_response)
    elif isinstance(human_response, str):
        return human_response
    elif hasattr(human_response, 'content'):
        # 处理消息对象
        return human_response.content
    else:
        return str(human_response) if human_response else ""


def request_execution_approval(action_plan: dict) -> str:
    """请求执行审批
    
    Args:
        action_plan: 需要审批的行动计划
        
    Returns:
        str: 审批决策 (approved/rejected/modified)
    """
    steps_summary = "\n".join([f"  {i+1}. {step.get('description', step.get('step_id', 'N/A'))}" for i, step in enumerate(action_plan.get('steps', []))])
    
    approval_data = {
        "action_plan": action_plan,
        "query": "请审批以下执行计划",
        "timestamp": datetime.now().isoformat(),
        "type": "execution_approval",
        "ai_message": f"⚠️ **需要执行审批**\n\n"
                     f"📋 **计划ID**: {action_plan.get('plan_id', 'unknown')}\n"
                     f"⚡ **优先级**: {action_plan.get('priority', 'unknown')}\n"
                     f"⚠️ **风险评估**: {action_plan.get('risk_assessment', 'unknown')}\n\n"
                     f"📝 **执行步骤**:\n{steps_summary}\n\n"
                     f"📋 **请选择**: approved(同意) / rejected(拒绝) / modified(修改)\n"
                     f"⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
    }
    human_response = interrupt(approval_data)
    
    # 处理不同类型的返回值 - 支持嵌套字典格式
    if isinstance(human_response, dict):
        # 首先尝试直接键名
        result = (human_response.get("decision") or 
                 human_response.get("response") or 
                 human_response.get("input") or 
                 human_response.get("value"))
        
        if result:
            return result
        
        # 尝试嵌套字典格式
        for key, value in human_response.items():
            if isinstance(value, dict):
                nested_result = (value.get("decision") or 
                               value.get("response") or 
                               value.get("input") or 
                               value.get("value"))
                if nested_result:
                    return nested_result
            elif isinstance(value, str) and value.strip():
                return value
        
        return "rejected"
    elif isinstance(human_response, str):
        return human_response
    else:
        return str(human_response) if human_response else "rejected"


def request_clarification(ambiguous_input: str, context: dict = None) -> str:
    """请求意图澄清
    
    Args:
        ambiguous_input: 需要澄清的模糊输入
        context: 相关上下文
        
    Returns:
        str: 澄清后的明确指令
    """
    confidence = context.get('confidence', 0) if context else 0
    
    clarification_data = {
        "ambiguous_input": ambiguous_input,
        "context": context or {},
        "query": "请澄清您的具体意图",
        "timestamp": datetime.now().isoformat(),
        "type": "clarification",
        "ai_message": f"🤔 **需要澄清意图**\n\n"
                     f"💬 **原始输入**: {ambiguous_input[:100]}{'...' if len(ambiguous_input) > 100 else ''}\n"
                     f"📊 **理解置信度**: {confidence:.2f}\n\n"
                     f"📝 **请明确说明**:\n- 您希望我做什么？\n"
                     f"- 有什么具体的问题或症状吗？\n"
                     f"- 需要什么帮助？\n\n"
                     f"⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
    }
    human_response = interrupt(clarification_data)
    
    # 处理不同类型的返回值 - 支持嵌套字典格式
    if isinstance(human_response, dict):
        # 首先尝试直接键名
        result = (human_response.get("clarification") or 
                 human_response.get("response") or 
                 human_response.get("input") or 
                 human_response.get("value"))
        
        if result:
            return result
        
        # 尝试嵌套字典格式
        for key, value in human_response.items():
            if isinstance(value, dict):
                nested_result = (value.get("clarification") or 
                               value.get("response") or 
                               value.get("input") or 
                               value.get("value"))
                if nested_result:
                    return nested_result
            elif isinstance(value, str) and value.strip():
                return value
        
        return ""
    elif isinstance(human_response, str):
        return human_response
    else:
        return str(human_response) if human_response else ""




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
        self.task_router = TaskRouter()
        self.report_generator = ReportGenerator()
        self.nlu = NaturalLanguageUnderstanding()
        
        # 人类干预工具
        self.request_operator_input = request_operator_input
        self.request_execution_approval = request_execution_approval
        self.request_clarification = request_clarification
        
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
        agent_graph.add_node("understand_and_route", self._understand_and_route_node)
        # agent_graph.add_node("understand_input", self._understand_input_node)
        # agent_graph.add_node("route_task", self._route_task_node)
        agent_graph.add_node("process_alert", self._process_alert_node)
        agent_graph.add_node("diagnose_issue", self._diagnose_issue_node)
        agent_graph.add_node("plan_actions", self._plan_actions_node)
        agent_graph.add_node("execute_actions", self._execute_actions_node)
        agent_graph.add_node("generate_report", self._generate_report_node)
        agent_graph.add_node("finalize", self._finalize_node)
        agent_graph.add_node("error_handler", self._error_handler_node)
        
        # 设置入口点
        agent_graph.set_entry_point("initialize")
        
        # 添加边
        agent_graph.add_edge("initialize", "understand_and_route")
        # agent_graph.add_edge("understand_input", "route_task")
        
        # 条件边：根据任务类型路由
        agent_graph.add_conditional_edges(
            "understand_and_route",
            self._route_task_condition,
            {
                "process_alert": "process_alert",
                "diagnose_issue": "diagnose_issue", 
                "plan_actions": "plan_actions",
                "execute_actions": "execute_actions",
                "generate_report": "generate_report",
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
                "error": "error_handler",
                "understand_and_route": "understand_and_route"
            }
        )
        
        agent_graph.add_conditional_edges(
            "diagnose_issue",
            self._diagnose_completion_condition,
            {
                "plan_actions": "plan_actions",
                "finalize": "finalize", 
                "error": "error_handler",
                "understand_and_route": "understand_and_route"
            }
        )
        
        agent_graph.add_conditional_edges(
            "plan_actions",
            self._plan_completion_condition,
            {
                "execute_actions": "execute_actions",
                "finalize": "finalize",
                "error": "error_handler",
                "understand_and_route": "understand_and_route"
            }
        )
        
        agent_graph.add_conditional_edges(
            "execute_actions",
            self._execute_completion_condition,
            {
                "generate_report": "generate_report",
                "finalize": "finalize",
                "error": "error_handler",
                "understand_and_route": "understand_and_route"
            }
        )
        
        # 独立任务节点（直接结束）
        for task_node in ["generate_report"]:
            agent_graph.add_conditional_edges(
                task_node,
                self._task_completion_condition,
                {
                    "finalize": "finalize",
                    "error": "error_handler",
                    "understand_and_route": "understand_and_route"
                }
            )
        
        # 错误处理
        agent_graph.add_conditional_edges(
            "error_handler",
            self._error_recovery_condition,
            {
                "retry": "understand_and_route",
                "finalize": "finalize",
                "END": END
            }
        )
        
        # 结束节点
        agent_graph.add_edge("finalize", END)
        
        return agent_graph
    
    def compile(self, checkpointer=None):
        """编译智能体图，支持checkpointer"""
        if not self.compiled_graph:
            compile_kwargs = {}
            if checkpointer:
                compile_kwargs["checkpointer"] = checkpointer
                # 可选：添加静态断点
                compile_kwargs["interrupt_before"] = ["execute_actions"]
            
            self.compiled_graph = self.graph.compile(**compile_kwargs)
        return self.compiled_graph
        
    # ==================== 节点函数 ====================
    
    async def _initialize_node(self, state: ChatState) -> ChatState:
        """初始化节点 - 设置处理状态"""
        return {
            **state,
            "workflow_id": f"{self.config.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    async def _understand_and_route_node(self, state: ChatState) -> ChatState:
        """合并节点：自然语言理解 + 任务路由 - 支持用户打断"""
        try:
            # ==================== Part 1: 自然语言理解 ====================
            user_input = self._get_latest_user_input(state)
            
            current_state = state
            if user_input and user_input.strip():
                print(f"🧠 开始自然语言理解: {user_input[:50]}...")
                nlu_result = await asyncio.to_thread(self.nlu.forward, user_input)
                
                # 如果NLU置信度很低，主动请求澄清
                if nlu_result.confidence < 0.5:
                    clarification = self.request_clarification(
                        ambiguous_input=user_input,
                        context={
                            "low_confidence_nlu": True,
                            "original_input": user_input,
                            "current_stage": state.get("stage", "unknown"),
                            "confidence": nlu_result.confidence
                        }
                    )
                    
                    if clarification and clarification.strip():
                        # 将澄清信息添加到消息中并重新处理
                        from langchain_core.messages import HumanMessage
                        new_message = HumanMessage(content=clarification)
                        
                        # 更新消息列表
                        state["messages"] = state.get("messages", []) + [new_message]
                        
                        # 重新进行 NLU
                        nlu_result = await asyncio.to_thread(self.nlu.forward, clarification)
                
                # 直接将NLU结果转换为最终current_task，不存储临时意图字段
                updated_state_from_nlu = {
                    "extracted_info": nlu_result.extracted_info,
                }
                if nlu_result.alert_info:
                    updated_state_from_nlu["alert_info"] = AlertInfo(**nlu_result.alert_info)
                    print(f"📊 提取到告警信息: {updated_state_from_nlu['alert_info'].message}")
                if nlu_result.symptoms:
                    updated_state_from_nlu["symptoms"] = nlu_result.symptoms
                    print(f"🔍 提取到症状: {nlu_result.symptoms}")
                if nlu_result.context:
                    existing_context = state.get("context", {}) or {}
                    updated_state_from_nlu["context"] = {**existing_context, **nlu_result.context}
                    print(f"📋 提取到上下文: {nlu_result.context}")

                # 直接转换意图为任务，无需中间状态
                task_mapping = {
                    "process_alert": "process_alert", "diagnose_issue": "diagnose_issue",
                    "plan_actions": "plan_actions", "execute_actions": "execute_actions",
                    "generate_report": "generate_report"
                }
                
                final_task = task_mapping.get(nlu_result.intent, "diagnose_issue")  # 默认任务
                updated_state_from_nlu["current_task"] = final_task
                
                current_state = {**state, **updated_state_from_nlu}
                print(f"✅ NLU直接路由完成 - 任务: {final_task}, 置信度: {nlu_result.confidence:.2f}")
                
                # 添加 AI 输出到 messages
                current_state = self._add_ai_message_to_state(
                    current_state,
                    f"🧠 **理解和路由完成**\n\n"
                    f"🎯 **识别任务**: {final_task}\n"
                    f"📊 **置信度**: {nlu_result.confidence:.2f}\n"
                    f"🔍 **提取信息**: {len(nlu_result.extracted_info) if nlu_result.extracted_info else 0} 项\n"
                    f"⏰ **处理时间**: {datetime.now().strftime('%H:%M:%S')}"
                )
                
                return current_state

            # ==================== 备用路由逻辑 ====================
            print("🎯 使用备用路由...")
            valid_tasks = ["process_alert", "diagnose_issue", "plan_actions", 
                           "execute_actions", "generate_report"]

            # 1. 检查是否有明确指定的任务类型
            if state.get("current_task") in valid_tasks:
                print(f"🎯 明确任务: {state['current_task']}")
                result_state = {**state, "current_task": state["current_task"]}
                return self._add_ai_message_to_state(
                    result_state,
                    f"🎯 **任务路由**: 明确任务 {state['current_task']}"
                )

            # 2. 使用 DSPy 智能路由作为备选
            try:
                user_input = state  # 简化：直接使用state作为路由输入
                routing_result = await asyncio.to_thread(self.task_router.forward, user_input)
                if routing_result.confidence > 0.6:
                    print(f"🎯 DSPy路由: {routing_result.task_type} (置信度: {routing_result.confidence:.2f})")
                    result_state = {
                        **state,
                        "current_task": routing_result.task_type
                    }
                    return self._add_ai_message_to_state(
                        result_state,
                        f"🎯 **智能路由**: {routing_result.task_type} (置信度: {routing_result.confidence:.2f})"
                    )
                else:
                    print(f"⚠️ DSPy 路由置信度不足 ({routing_result.confidence:.2f})")
            except Exception as e:
                print(f"❌ DSPy 路由失败: {str(e)}")

            # 3. 回退到基于规则的路由
            rule_based_task = self._rule_based_routing(state)
            print(f"🎯 规则路由: {rule_based_task}")
            result_state = {
                **state,
                "current_task": rule_based_task,
                "errors": (state.get("errors") or []) + [f"DSPy routing failed"]
            }
            return self._add_ai_message_to_state(
                result_state,
                f"🎯 **规则路由**: {rule_based_task} (备用方案)"
            )

        except Exception as e:
            # 检查是否是中断异常，如果是则重新抛出
            if "Interrupt" in type(e).__name__ or "interrupt" in str(e).lower():
                raise
            return self._create_error_state(state, e, "understand_and_route")
    
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
            historical_incidents = []  # 简化：不使用历史数据
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
            
            result_state = {
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
            
            # 添加 AI 输出到 messages
            return self._add_ai_message_to_state(
                result_state,
                f"🚨 **告警分析完成**\n\n"
                f"📊 **优先级**: {analysis_result.priority}\n"
                f"🏷️ **类别**: {analysis_result.category}\n"
                f"⚡ **紧急度分数**: {analysis_result.urgency_score:.2f}\n"
                f"🔍 **根因线索**: {len(analysis_result.root_cause_hints)} 项\n"
                f"💡 **建议操作**: {len(analysis_result.recommended_actions)} 项\n"
                f"⏰ **分析时间**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Alert processing error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    async def _diagnose_issue_node(self, state: ChatState) -> ChatState:
        """诊断问题节点 - 支持主动干预"""
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
                historical_incidents=[],
                topology_info=context.get("topology_info", {}) if context else {}
            )
            
            # 执行诊断 - 使用 asyncio.to_thread 处理同步调用
            diagnostic_result = await asyncio.to_thread(
                self.diagnostic_agent.forward,
                diagnostic_context
            )
            
            # 主动干预：如果诊断置信度低，请求额外信息
            if diagnostic_result.confidence_score < 0.7:
                print(f"🤔 诊断置信度较低 ({diagnostic_result.confidence_score:.2f})，请求运维人员提供额外信息...")
                
                # 先添加 AI 请求消息
                from langchain_core.messages import AIMessage
                request_message = AIMessage(
                    content=f"🤔 **诊断置信度低，需要您的帮助**\n\n"
                           f"📊 **当前置信度**: {diagnostic_result.confidence_score:.2f}\n"
                           f"🔍 **初步诊断**: {diagnostic_result.root_cause}\n"
                           f"🏢 **受影响组件**: {', '.join(diagnostic_result.affected_components)}\n\n"
                           f"📝 **需要您提供**:\n- 其他线索或观察到的异常\n"
                           f"- 相关日志信息\n"
                           f"- 其他可能的原因\n\n"
                           f"👤 **请输入您的观察**:"
                )
                state["messages"] = state.get("messages", []) + [request_message]
                
                additional_info = self.request_operator_input(
                    query=f"诊断置信度较低({diagnostic_result.confidence_score:.2f})，请提供额外信息：\n"
                          f"初步诊断：{diagnostic_result.root_cause}\n"
                          f"受影响组件：{', '.join(diagnostic_result.affected_components)}\n"
                          f"是否有其他线索、日志或观察到的异常？",
                    context={
                        "current_diagnosis": diagnostic_result.root_cause,
                        "confidence": diagnostic_result.confidence_score,
                        "affected_components": diagnostic_result.affected_components,
                        "evidence": diagnostic_result.evidence
                    }
                )
                
                # 如果获得了额外信息，基于新信息重新诊断
                if additional_info and additional_info.strip():
                    print(f"📋 收到额外信息，重新进行诊断: {additional_info[:100]}...")
                    
                    # 将用户的额外输入添加到消息中
                    from langchain_core.messages import HumanMessage
                    additional_message = HumanMessage(content=f"补充信息: {additional_info}")
                    state["messages"] = state.get("messages", []) + [additional_message]
                    
                    # 增强诊断上下文
                    enhanced_context = DiagnosticContext(
                        alert_analysis=alert_analysis,
                        system_metrics=context.get("system_metrics", {}) if context else {},
                        log_entries=context.get("log_entries", []) + [f"运维人员补充: {additional_info}"] if context else [f"运维人员补充: {additional_info}"],
                        historical_incidents=[],
                        topology_info=context.get("topology_info", {}) if context else {},
                        additional_context={"human_input": additional_info}
                    )
                    
                    # 重新执行诊断
                    diagnostic_result = await asyncio.to_thread(
                        self.diagnostic_agent.forward,
                        enhanced_context
                    )
                    print(f"✅ 基于额外信息重新诊断完成，新置信度: {diagnostic_result.confidence_score:.2f}")
                    
                    # 添加重新诊断的结果消息
                    reanalysis_message = AIMessage(
                        content=f"🔄 **重新诊断完成**\n\n"
                               f"📊 **新置信度**: {diagnostic_result.confidence_score:.2f}\n"
                               f"🔍 **更新诊断**: {diagnostic_result.root_cause}\n"
                               f"📋 **感谢**: 您的补充信息帮助提高了诊断准确性"
                    )
                    state["messages"] = state.get("messages", []) + [reanalysis_message]
                else:
                    # 调试：条件判断为假
                    print(f"🔍 DEBUG: 条件判断为假，没有收到有效的额外信息")
                    print(f"🔍 DEBUG: 保持原始诊断结果，置信度: {diagnostic_result.confidence_score:.2f}")
            
            result_state = {
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
            
            # 添加 AI 输出到 messages
            return self._add_ai_message_to_state(
                result_state,
                f"🩺 **故障诊断完成**\n\n"
                f"🔍 **根本原因**: {diagnostic_result.root_cause}\n"
                f"📊 **置信度**: {diagnostic_result.confidence_score:.2f}\n"
                f"💥 **影响评估**: {diagnostic_result.impact_assessment}\n"
                f"🏢 **业务影响**: {diagnostic_result.business_impact}\n"
                f"⏱️ **预计恢复时间**: {diagnostic_result.recovery_time_estimate}\n"
                f"🔧 **受影响组件**: {', '.join(diagnostic_result.affected_components) if diagnostic_result.affected_components else 'N/A'}\n"
                f"📋 **证据**: {len(diagnostic_result.evidence)} 项\n"
                f"⏰ **诊断时间**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            # 检查是否是中断异常，如果是则重新抛出
            if "Interrupt" in type(e).__name__ or "interrupt" in str(e).lower():
                raise
            return self._create_error_state(state, e, "diagnose_issue")
    
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
            
            result_state = {
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
            
            # 添加 AI 输出到 messages
            return self._add_ai_message_to_state(
                result_state,
                f"📋 **行动计划生成完成**\n\n"
                f"🆔 **计划ID**: {action_plan.plan_id}\n"
                f"⚡ **优先级**: {action_plan.priority}\n"
                f"⏱️ **预估时长**: {action_plan.estimated_duration}\n"
                f"⚠️ **风险评估**: {action_plan.risk_assessment}\n"
                f"✅ **需要审批**: {'是' if action_plan.approval_required else '否'}\n"
                f"📝 **执行步骤**: {len(action_plan.steps)} 步\n"
                f"🔄 **回滚计划**: {len(action_plan.rollback_plan)} 步\n"
                f"⏰ **规划时间**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Action planning error: {str(e)}"],
                "last_update": datetime.now()
            }
    
    def _requires_execution_approval(self, action_plan: Dict[str, Any]) -> bool:
        """判断是否需要执行审批"""
        if not action_plan:
            return False
        
        # 检查风险级别
        risk_assessment = action_plan.get("risk_assessment", "low")
        if risk_assessment in ["high", "critical"]:
            return True
        
        # 检查是否明确要求审批
        if action_plan.get("approval_required", False):
            return True
        
        # 检查步骤中是否有高风险操作
        steps = action_plan.get("steps", [])
        high_risk_actions = ["restart", "reboot", "delete", "remove", "kill", "stop"]
        for step in steps:
            action_type = step.get("action_type", "").lower()
            description = step.get("description", "").lower()
            command = step.get("command", "").lower()
            
            if any(risk_action in action_type or risk_action in description or risk_action in command 
                   for risk_action in high_risk_actions):
                return True
        
        return False

    async def _execute_actions_node(self, state: ChatState) -> ChatState:
        """执行行动节点 - 支持主动审批"""
        try:
            action_plan = state.get("action_plan")
            if not action_plan:
                raise ValueError("No action plan available")
            
            # 主动干预：检测高风险操作并请求审批
            if self._requires_execution_approval(action_plan):
                print(f"⚠️ 检测到高风险操作，请求执行审批...")
                
                approval_decision = self.request_execution_approval(action_plan=action_plan)
                
                # 将审批结果添加到消息中
                from langchain_core.messages import HumanMessage
                approval_message = HumanMessage(content=f"审批决策: {approval_decision}")
                state["messages"] = state.get("messages", []) + [approval_message]
                
                if approval_decision.lower() in ['rejected', 'deny', 'no', 'cancel']:
                    result_state = {
                        **state,
                        "stage": "execution_rejected",
                        "execution_result": {
                            "status": "rejected_by_operator",
                            "reason": f"执行被拒绝: {approval_decision}",
                            "plan_id": action_plan.get("plan_id", "unknown"),
                            "timestamp": datetime.now().isoformat()
                        },
                        "last_update": datetime.now()
                    }
                    return self._add_ai_message_to_state(
                        result_state,
                        f"❌ **执行被拒绝**\n\n"
                        f"🚫 **拒绝原因**: {approval_decision}\n"
                        f"📋 **计划ID**: {action_plan.get('plan_id', 'unknown')}\n"
                        f"⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
                    )
                elif approval_decision.lower() in ['modified', 'change', 'update']:
                    # 如果用户要求修改，返回到规划阶段
                    result_state = {
                        **state,
                        "execution_result": {
                            "status": "modification_requested",
                            "reason": f"需要修改计划: {approval_decision}",
                            "plan_id": action_plan.get("plan_id", "unknown"),
                            "timestamp": datetime.now().isoformat()
                        },
                        "current_task": "plan_actions"
                    }
                    return self._add_ai_message_to_state(
                        result_state,
                        f"🔄 **需要修改计划**\n\n"
                        f"📝 **修改要求**: {approval_decision}\n"
                        f"➡️ **返回**: 行动规划阶段\n"
                        f"⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
                    )
                else:
                    print(f"✅ 执行已获得审批: {approval_decision}")
            
            if not self.config.auto_execution and not self._requires_execution_approval(action_plan):
                result_state = {
                    **state,
                    "stage": "executed",
                    "execution_result": {
                        "status": "manual_approval_required",
                        "message": "Automatic execution is disabled. Manual approval required.",
                        "plan_id": action_plan.get("plan_id", "unknown")
                    },
                    "last_update": datetime.now()
                }
                return self._add_ai_message_to_state(
                    result_state,
                    f"⚠️ **需要手动审批**\n\n"
                    f"🔒 **原因**: 自动执行已禁用\n"
                    f"📋 **计划ID**: {action_plan.get('plan_id', 'unknown')}\n"
                    f"⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
                )
            
            # 模拟执行过程
            executed_steps = []
            failed_steps = []
            
            print(f"🚀 开始执行行动计划: {action_plan.get('plan_id', 'unknown')}")
            
            for step in action_plan.get("steps", []):
                try:
                    # 模拟步骤执行
                    print(f"  ⏳ 执行步骤: {step.get('description', step.get('step_id'))}")
                    await asyncio.sleep(0.1)  # 模拟执行时间
                    executed_steps.append(step["step_id"])
                except Exception as e:
                    failed_steps.append({
                        "step_id": step["step_id"],
                        "error": str(e)
                    })
            
            execution_status = "success" if not failed_steps else "partial" if executed_steps else "failed"
            print(f"✅ 执行完成，状态: {execution_status}")
            
            result_state = {
                **state,
                "stage": "executed",
                "execution_result": {
                    "status": execution_status,
                    "plan_id": action_plan.get("plan_id", "unknown"),
                    "executed_steps": executed_steps,
                    "failed_steps": failed_steps,
                    "execution_time": datetime.now().isoformat(),
                    "approval_received": True if self._requires_execution_approval(action_plan) else False
                },
                "last_update": datetime.now()
            }
            
            # 添加 AI 输出到 messages
            status_emoji = "✅" if execution_status == "success" else "⚠️" if execution_status == "partial" else "❌"
            return self._add_ai_message_to_state(
                result_state,
                f"{status_emoji} **执行完成**\n\n"
                f"📊 **执行状态**: {execution_status}\n"
                f"📋 **计划ID**: {action_plan.get('plan_id', 'unknown')}\n"
                f"✅ **成功步骤**: {len(executed_steps)} 步\n"
                f"❌ **失败步骤**: {len(failed_steps)} 步\n"
                f"🔐 **审批状态**: {'已获得' if self._requires_execution_approval(action_plan) else '无需审批'}\n"
                f"⏰ **执行时间**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            # 检查是否是中断异常，如果是则重新抛出
            if "Interrupt" in type(e).__name__ or "interrupt" in str(e).lower():
                raise
            return self._create_error_state(state, e, "execute_actions")
    
    async def _generate_report_node(self, state: ChatState) -> ChatState:
        """生成报告节点"""
        try:
            if not self.config.enable_reporting:
                result_state = {
                    **state,
                    "stage": "reported",
                    "report": {
                        "status": "disabled",
                        "message": "Reporting is disabled for this agent"
                    },
                    "last_update": datetime.now()
                }
                return self._add_ai_message_to_state(
                    result_state,
                    f"📋 **报告生成**: 已禁用\n\n"
                    f"⚠️ **状态**: 此智能体的报告功能已禁用\n"
                    f"⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
                )
            
            # 使用 ReportGenerator 生成专业报告
            try:
                # 准备报告输入数据
                incident_data = {
                    "incident_id": state.get("workflow_id", "unknown"),
                    "alert_info": state.get("alert_info"),
                    "symptoms": state.get("symptoms"),
                    "context": state.get("context"),
                    "timestamp": datetime.now().isoformat()
                }
                
                diagnostic_data = state.get("diagnostic_result", {})
                action_data = state.get("action_plan", {})
                execution_data = state.get("execution_result", {})
                
                # 调用ReportGenerator的forward方法
                report_result = await asyncio.to_thread(
                    self.report_generator.forward,
                    incident_data=str(incident_data),
                    diagnostic_result=str(diagnostic_data),
                    action_plan=str(action_data),
                    execution_result=str(execution_data)
                )
                
                # 构建最终报告
                report = {
                    "incident_id": state.get("workflow_id", "unknown"),
                    "title": f"智能运维报告 - {self.config.agent_id}",
                    "summary": report_result.report_summary,
                    "key_findings": report_result.key_findings,
                    "recommendations": report_result.recommendations,
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.config.agent_id,
                    "status": "generated",
                    "raw_data": {
                        "analysis": state.get("analysis_result"),
                        "diagnosis": state.get("diagnostic_result"),
                        "action_plan": state.get("action_plan"),
                        "execution": state.get("execution_result")
                    }
                }
                
            except Exception as e:
                print(f"❌ ReportGenerator 调用失败: {e}")
                # 回退到简化版本
                report = {
                    "incident_id": state.get("workflow_id", "unknown"),
                    "title": f"智能运维报告 - {self.config.agent_id}",
                    "summary": f"任务完成: {state.get('current_task', 'unknown')}",
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.config.agent_id,
                    "status": "generated_fallback",
                    "error": f"报告生成失败: {str(e)}",
                    "raw_data": {
                        "analysis": state.get("analysis_result"),
                        "diagnosis": state.get("diagnostic_result"),
                        "action_plan": state.get("action_plan"),
                        "execution": state.get("execution_result")
                    }
                }
            
            result_state = {
                **state,
                "stage": "reported",
                "report": report,
                "last_update": datetime.now()
            }
            
            # 添加 AI 输出到 messages
            return self._add_ai_message_to_state(
                result_state,
                f"📋 **运维报告生成完成**\n\n"
                f"🆔 **事件ID**: {report.get('incident_id', 'unknown')}\n"
                f"📊 **报告状态**: {report.get('status', 'unknown')}\n"
                f"📝 **主要发现**: {len(report.get('key_findings', [])) if isinstance(report.get('key_findings'), list) else 'N/A'} 项\n"
                f"💡 **建议**: {len(report.get('recommendations', [])) if isinstance(report.get('recommendations'), list) else 'N/A'} 项\n"
                f"🤖 **生成智能体**: {self.config.agent_id}\n"
                f"⏰ **生成时间**: {datetime.now().strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            return {
                **state,
                "stage": "error",
                "errors": state.get("errors", []) + [f"Report generation error: {str(e)}"],
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
            if state.get('current_task'):
                response_text += f"🎯 **当前任务**: {state.get('current_task')}\n"
            
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
    
    def _add_ai_message_to_state(self, state: ChatState, content: str) -> ChatState:
        """添加 AI 消息到状态中"""
        try:
            # 检查是否有消息列表（聊天模式）
            messages = state.get("messages")
            if messages is not None:
                from langchain_core.messages import AIMessage
                ai_message = AIMessage(content=content)
                return {
                    **state,
                    "messages": messages + [ai_message]
                }
            # 如果没有消息列表，直接返回原状态（任务模式）
            return state
        except Exception as e:
            print(f"⚠️ 添加 AI 消息失败: {e}")
            return state
    
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
    
    
    def _get_latest_user_input(self, state: ChatState) -> Optional[str]:
        """从消息历史中获取最新的用户输入"""
        messages = state.get("messages", [])
        if not messages:
            return None
            
        last_message = messages[-1]
        if not hasattr(last_message, 'content'):
            return None
            
        content = last_message.content
        if isinstance(content, list):
            return "".join(item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in content)
        else:
            return str(content)

    
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
            return "generate_report"  # 简化：反馈直接转为报告生成
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
    


# ==================== LangGraph Studio 集成 ====================

# 创建默认的智能体配置供 Studio 使用
_default_studio_config = AgentConfig(
    agent_id="ops_agent_studio",
    agent_type="general", 
    specialization="studio_demo",
    enable_reporting=True,
    auto_execution=False
)

# 创建智能体实例并编译图
_studio_agent = IntelligentOpsAgent(_default_studio_config)
graph = _studio_agent.compile()

# 同时导出智能体实例供其他用途
agent = _studio_agent