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
from src.dspy_modules.report_generator import ReportGenerator
from src.dspy_modules.intelligent_router import IntelligentRouter
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
    auto_execution: bool = True


class ChatState(TypedDict):
    """智能运维状态 - 集成记忆系统和信息收集功能"""
    # LangGraph 标准聊天字段
    messages: Annotated[List[BaseMessage], add_messages]  # 聊天消息列表
    
    # 核心业务数据
    alert_info: Optional[AlertInfo]  # 告警信息
    symptoms: Optional[List[str]]  # 症状列表
    context: Optional[Dict[str, Any]]  # 运维上下文信息
    
    # 处理结果
    analysis_result: Optional[Dict[str, Any]]  # 告警分析结果
    diagnostic_result: Optional[Dict[str, Any]]  # 诊断结果
    action_plan: Optional[Dict[str, Any]]  # 行动计划
    execution_result: Optional[Dict[str, Any]]  # 执行结果
    report: Optional[Dict[str, Any]]  # 报告
    
    # 记忆系统字段（精简版）
    memory_queries: Optional[List[str]]  # 待执行查询（空=不需要检索）
    memory_context: Optional[Dict[str, Any]]  # 检索结果
    
    # 信息收集字段（精简版）
    info_collection_queries: Optional[List[str]]  # 待收集信息类型（空=不需要收集）
    info_collection_context: Optional[Dict[str, Any]]  # 收集结果
    
    # 调试支持
    errors: Optional[List[str]]  # 错误列表


def validate_chat_state(state: ChatState) -> bool:
    """验证 ChatState 的有效性"""
    try:
        # 必须字段检查
        if not isinstance(state.get("messages", []), list):
            return False
        
        # 记忆查询字段验证
        memory_queries = state.get("memory_queries")
        if memory_queries is not None and not isinstance(memory_queries, list):
            return False
        
        # 记忆上下文字段验证  
        memory_context = state.get("memory_context")
        if memory_context is not None and not isinstance(memory_context, dict):
            return False
        
        # 信息收集查询字段验证
        info_queries = state.get("info_collection_queries")
        if info_queries is not None and not isinstance(info_queries, list):
            return False
        
        # 信息收集上下文字段验证
        info_context = state.get("info_collection_context")
        if info_context is not None and not isinstance(info_context, dict):
            return False
        
        return True
        
    except Exception:
        return False


def create_empty_chat_state() -> ChatState:
    """创建空的 ChatState"""
    return ChatState(
        messages=[],
        alert_info=None,
        symptoms=None,
        context=None,
        analysis_result=None,
        diagnostic_result=None,
        action_plan=None,
        execution_result=None,
        report=None,
        memory_queries=None,
        memory_context=None,
        info_collection_queries=None,
        info_collection_context=None,
        errors=None
    )


# ==================== 人类干预工具 ====================

def request_operator_input(query: str, context: dict = None) -> str:
    """请求运维人员输入和确认
    
    Args:
        query: 向运维人员提出的问题或请求
        context: 相关上下文信息
        
    Returns:
        str: 运维人员的回复
    """
    # 使用 LangGraph 的 interrupt() 函数来中断工作流并等待用户输入
    interrupt_data = {
        "query": query,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
        "type": "operator_input",
        "message": f"🤖 **需要您的输入**\n\n💬 **问题**: {query}\n\n⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
    }
    
    # 中断工作流并等待用户响应
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
        self.intelligent_router = IntelligentRouter()
        
        # 人类干预工具
        self.request_operator_input = request_operator_input
        self.request_execution_approval = request_execution_approval
        
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
        agent_graph.add_node("router", self._router_node)  # 重命名为router
        agent_graph.add_node("collect_info", self._collect_info_node)  # 信息收集节点
        agent_graph.add_node("process_alert", self._process_alert_node)
        agent_graph.add_node("diagnose_issue", self._diagnose_issue_node)
        agent_graph.add_node("plan_actions", self._plan_actions_node)
        agent_graph.add_node("execute_actions", self._execute_actions_node)
        agent_graph.add_node("generate_report", self._generate_report_node)
        agent_graph.add_node("finalize", self._finalize_node)
        agent_graph.add_node("error_handler", self._error_handler_node)
        
        # 设置入口点
        agent_graph.set_entry_point("initialize")
        
        # 新架构：initialize → router（router负责信息提取和路由决策）
        agent_graph.add_edge("initialize", "router")
        
        # 路由节点到各个任务节点的条件边
        agent_graph.add_conditional_edges(
            "router",
            self._route_condition,
            {
                "process_alert": "process_alert",
                "diagnose_issue": "diagnose_issue", 
                "plan_actions": "plan_actions",
                "execute_actions": "execute_actions",
                "generate_report": "generate_report",
                "collect_info": "collect_info",  # 信息收集路由
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        # 信息收集节点完成后回到路由节点
        agent_graph.add_edge("collect_info", "router")
        
        # 业务节点的正常流程连接：process_alert → diagnose_issue → plan_actions → execute_actions → finalize
        agent_graph.add_conditional_edges(
            "process_alert",
            self._process_alert_condition,
            {
                "diagnose_issue": "diagnose_issue",  # 正常流程：下一步
                "collect_info": "collect_info",     # 异常处理：收集信息
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        agent_graph.add_conditional_edges(
            "diagnose_issue",
            self._diagnose_issue_condition,
            {
                "plan_actions": "plan_actions",     # 正常流程：下一步
                "collect_info": "collect_info",     # 异常处理：收集信息
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        agent_graph.add_conditional_edges(
            "plan_actions",
            self._plan_actions_condition,
            {
                "execute_actions": "execute_actions",  # 正常流程：下一步
                "collect_info": "collect_info",        # 异常处理：收集信息
                "finalize": "finalize",
                "error": "error_handler"
            }
        )
        
        agent_graph.add_conditional_edges(
            "execute_actions",
            self._execute_actions_condition,
            {
                "finalize": "finalize",              # 正常流程：完成
                "collect_info": "collect_info",      # 异常处理：收集信息
                "generate_report": "generate_report",
                "error": "error_handler"
            }
        )
        
        # generate_report 节点保持原有逻辑
        agent_graph.add_conditional_edges(
            "generate_report",
            self._generate_report_condition,
            {
                "finalize": "finalize",
                "collect_info": "collect_info",
                "error": "error_handler"
            }
        )
        
        # 错误处理
        agent_graph.add_conditional_edges(
            "error_handler",
            self._error_recovery_condition,
            {
                "retry": "router",
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
                # 添加中断点：在信息收集和执行操作前中断
                #compile_kwargs["interrupt_before"] = ["collect_info", "execute_actions"]
            
            self.compiled_graph = self.graph.compile(**compile_kwargs)
        return self.compiled_graph
        
    # ==================== 节点函数 ====================
    
    async def _initialize_node(self, state: ChatState) -> ChatState:
        """初始化节点 - 设置处理状态"""
        return {**state}
    
    async def _router_node(self, state: ChatState) -> ChatState:
        """统一的智能路由节点 - 处理所有路由复杂性"""
        try:
            # 检查是否有消息历史（仅用于日志显示）
            messages = state.get("messages", [])
            
            # 如果没有消息历史，基于当前业务数据进行路由决策
            if not messages:
                print(f"🧠 基于业务数据进行路由分析...")
                # 直接基于现有状态进行路由决策
                return self._route_based_on_business_data(state)
            
            print(f"🧠 开始智能路由分析...") 
            
            # 调用统一的 DSPy 路由器（只传递 state）
            router_decision = await asyncio.to_thread(
                self.intelligent_router.forward,
                current_state=state
            )
            
            print(f"✅ 路由决策完成 - 目标: {router_decision.next_task}, 置信度: {router_decision.confidence:.2f}")
            
            # 如果置信度很低，跳转到信息收集节点
            if router_decision.confidence < 0.2:
                print(f"⚠️ 路由置信度较低 ({router_decision.confidence:.2f})，跳转到信息收集")
                return self._route_based_on_business_data(state)
            
            # 应用提取的信息到状态
            updated_state = self._apply_extracted_info(state, router_decision)
            
            # 添加路由分析消息
            analysis_message = self._create_routing_analysis_message(router_decision)
            updated_state = self._add_ai_message_to_state(updated_state, analysis_message)
            
            # 根据路由决策标记下一个目标节点
            updated_state["_target_node"] = router_decision.next_task
            
            return updated_state
            
        except Exception as e:
            # 不要捕获中断异常，让它们正常传播
            # 只处理真正的错误
            return self._create_error_state(state, e, "router")
    
    def _route_based_on_business_data(self, state: ChatState) -> ChatState:
        """基于业务数据进行路由决策（无用户输入时）"""
        # 根据当前业务数据状态决定下一步路由
        if state.get("action_plan") and not state.get("execution_result"):
            target_node = "execute_actions"
        elif state.get("diagnostic_result") and not state.get("action_plan"):
            target_node = "plan_actions"
        elif (state.get("symptoms") or state.get("alert_info")) and not state.get("diagnostic_result"):
            target_node = "diagnose_issue"
        elif state.get("alert_info") and not state.get("analysis_result"):
            target_node = "process_alert"
        elif state.get("execution_result") and not state.get("report") and self.config.enable_reporting:
            target_node = "generate_report"
        else:
            # 默认情况：收集信息
            target_node = "collect_info"
        
        # 标记目标节点
        updated_state = {
            **state,
            "_target_node": target_node
        }
        
        # 添加路由分析消息
        analysis_message = f"🧠 **业务数据路由分析**\n\n" \
                          f"🎯 **下一步任务**: {target_node}\n" \
                          f"📊 **分析基于**: 当前业务数据状态\n" \
                          f"⏰ **分析时间**: {datetime.now().strftime('%H:%M:%S')}"
        
        return self._add_ai_message_to_state(updated_state, analysis_message)
    
    async def _collect_info_node(self, state: ChatState) -> ChatState:
        """信息收集节点 - 通过request_operator_input收集用户补充信息"""
        try:
            # 获取上一个节点传递的信息需求
            messages = state.get("messages", [])
            info_request = "📝 **需要补充信息**\n\n请提供更多详细信息以继续智能诊断："
            
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content') and ("需要补充信息" in last_message.content or "请提供" in last_message.content):
                    info_request = last_message.content
            
            # 使用 request_operator_input 函数收集用户信息
            # 这会调用 interrupt() 并在 LangGraph Studio 中显示输入提示
            additional_info = request_operator_input(
                query=info_request,
                context={
                    "type": "info_collection",
                    "current_state": "collect_info_node",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # 如果返回空值，提供默认信息
            if not additional_info:
                additional_info = "用户未提供额外信息"
            
            # 将收集到的信息添加到messages
            from langchain_core.messages import HumanMessage
            new_message = HumanMessage(content=str(additional_info))
            
            messages = state.get("messages") or []
            updated_state = {
                **state,
                "messages": messages + [new_message]
            }
            
            return updated_state
            
        except Exception as e:
            if "Interrupt" in type(e).__name__ or "interrupt" in str(e).lower():
                raise
            return self._create_error_state(state, e, "collect_info")
    
    async def _process_alert_node(self, state: ChatState) -> ChatState:
        """处理告警节点 - 智能判断是否需要处理告警"""
        try:
            # 首先检查是否已经有告警信息
            alert_info_dict = state.get("alert_info")
            
            if not alert_info_dict:
                # 如果没有alert_info，从用户输入中分析是否需要处理告警
                user_input = self._get_latest_user_or_system_input(state)
                if user_input:
                    # 使用智能路由器分析用户意图
                    router_decision = await asyncio.to_thread(
                        self.intelligent_router.forward,
                        user_input=user_input,
                        current_state=state
                    )
                    
                    # 判断是否为告警处理类型的请求
                    if router_decision.next_task == "process_alert":
                        # 这是告警处理请求，继续处理
                        updated_state = self._apply_extracted_info(state, router_decision)
                        alert_info_dict = updated_state.get("alert_info")
                        
                        if not alert_info_dict:
                            return await self._redirect_to_collect_info(updated_state, "process_alert", "告警信息提取失败")
                        
                        state = updated_state
                    else:
                        # 不是告警处理请求，跳过告警处理，直接路由到合适的节点
                        print(f"🔄 跳过告警处理，用户意图: {router_decision.next_task}")
                        return await self._redirect_to_collect_info(state, "process_alert", 
                                                           f"非告警处理请求，意图: {router_decision.next_task}")
                else:
                    return await self._redirect_to_collect_info(state, "process_alert", "缺少用户输入")
            
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
            analysis_result = await asyncio.to_thread(
                self.alert_analyzer.forward,
                alert_info=alert_info,
                historical_alerts=historical_alerts
            )
            
            result_state = {
                **state,
                "analysis_result": {
                    "priority": analysis_result.priority,
                    "category": analysis_result.category,
                    "urgency_score": analysis_result.urgency_score,
                    "root_cause_hints": analysis_result.root_cause_hints,
                    "recommended_actions": analysis_result.recommended_actions
                }
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
                "errors": state.get("errors", []) + [f"Alert processing error: {str(e)}"]
            }
    
    async def _diagnose_issue_node(self, state: ChatState) -> ChatState:
        """诊断问题节点 - 支持主动干预"""
        try:
            # 前置条件检查 - 诊断节点比较灵活，但需要基本信息
            symptoms = state.get("symptoms", [])
            context = state.get("context", {})
            alert_info = state.get("alert_info")
            
            has_basic_info = any([symptoms, context, alert_info])
            if not has_basic_info:
                return await self._redirect_to_collect_info(state, "diagnose_issue", "缺少基本问题信息（症状、上下文或告警）")
            
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
            
            # 如果诊断置信度低，跳转到信息收集节点
            if diagnostic_result.confidence_score < 0.7:
                print(f"🤔 诊断置信度较低 ({diagnostic_result.confidence_score:.2f})，跳转到信息收集节点")
                
                # 先保存初步诊断结果
                preliminary_state = {
                    **state,
                    "diagnostic_result": {
                        "root_cause": diagnostic_result.root_cause,
                        "confidence_score": diagnostic_result.confidence_score,
                        "impact_assessment": diagnostic_result.impact_assessment,
                        "affected_components": diagnostic_result.affected_components,
                        "business_impact": diagnostic_result.business_impact,
                        "recovery_estimate": diagnostic_result.recovery_time_estimate,
                        "similar_incidents": diagnostic_result.similar_incidents,
                        "evidence": diagnostic_result.evidence
                    }
                }
                
                # 跳转到信息收集节点
                return await self._redirect_to_collect_info(
                    preliminary_state,
                    "diagnose_issue",
                    f"诊断置信度较低 ({diagnostic_result.confidence_score:.2f})，需要提供额外信息以提高诊断准确性"
                )
            
            result_state = {
                **state,
                "diagnostic_result": {
                    "root_cause": diagnostic_result.root_cause,
                    "confidence_score": diagnostic_result.confidence_score,
                    "impact_assessment": diagnostic_result.impact_assessment,
                    "affected_components": diagnostic_result.affected_components,
                    "business_impact": diagnostic_result.business_impact,
                    "recovery_estimate": diagnostic_result.recovery_time_estimate,
                    "similar_incidents": diagnostic_result.similar_incidents,
                    "evidence": diagnostic_result.evidence
                }
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
            return self._create_error_state(state, e, "diagnose_issue")
    
    async def _plan_actions_node(self, state: ChatState) -> ChatState:
        """规划行动节点"""
        try:
            # 前置条件检查
            diagnostic_result = state.get("diagnostic_result")
            if not diagnostic_result:
                return await self._redirect_to_collect_info(state, "plan_actions", "缺少诊断结果")
            
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
                }
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
                "errors": state.get("errors", []) + [f"Action planning error: {str(e)}"]
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
            # 前置条件检查
            action_plan = state.get("action_plan")
            if not action_plan:
                return await self._redirect_to_collect_info(state, "execute_actions", "缺少行动计划")
            
            # 主动干预：检测高风险操作并请求审批
            if self._requires_execution_approval(action_plan):
                print(f"⚠️ 检测到高风险操作，请求执行审批...")
                
                approval_decision = self.request_execution_approval(action_plan=action_plan)
                
                # 将审批结果添加到消息中
                from langchain_core.messages import HumanMessage
                approval_message = HumanMessage(content=f"审批决策: {approval_decision}")
                messages = state.get("messages") or []
                state["messages"] = messages + [approval_message]
                
                if approval_decision.lower() in ['rejected', 'deny', 'no', 'cancel']:
                    result_state = {
                        **state,
                        "execution_result": {
                            "status": "rejected_by_operator",
                            "reason": f"执行被拒绝: {approval_decision}",
                            "plan_id": action_plan.get("plan_id", "unknown"),
                            "timestamp": datetime.now().isoformat()
                        }
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
                        }
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
                    "execution_result": {
                        "status": "manual_approval_required",
                        "message": "Automatic execution is disabled. Manual approval required.",
                        "plan_id": action_plan.get("plan_id", "unknown")
                    }
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
                "execution_result": {
                    "status": execution_status,
                    "plan_id": action_plan.get("plan_id", "unknown"),
                    "executed_steps": executed_steps,
                    "failed_steps": failed_steps,
                    "execution_time": datetime.now().isoformat(),
                    "approval_received": True if self._requires_execution_approval(action_plan) else False
                }
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
            if "Interrupt" in type(e).__name__ or "interrupt" in str(e).lower():
                raise
            return self._create_error_state(state, e, "execute_actions")
    
    async def _generate_report_node(self, state: ChatState) -> ChatState:
        """生成报告节点"""
        try:
            if not self.config.enable_reporting:
                result_state = {
                    **state,
                    "report": {
                        "status": "disabled",
                        "message": "Reporting is disabled for this agent"
                    }
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
                    "incident_id": f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
                    "incident_id": incident_data["incident_id"],
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
                    "incident_id": incident_data["incident_id"],
                    "title": f"智能运维报告 - {self.config.agent_id}",
                    "summary": "任务完成，报告生成失败",
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
                "report": report
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
                "errors": state.get("errors", []) + [f"Report generation error: {str(e)}"]
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
            # 从消息历史推断当前任务
            current_task = self._infer_current_task(state)
            if current_task:
                response_text += f"🎯 **当前任务**: {current_task}\n"
            
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
                "messages": state.get("messages", []) + [ai_message]
            }
        
        # 任务模式：返回结构化结果
        return {**state}
    
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
            
            messages = state.get("messages") or []
            return {
                **state,
                "messages": messages + [error_message]
            }
        
        # 任务模式：返回结构化错误
        return {**state}
    
    # ==================== 辅助函数 ====================
    
    def _add_ai_message_to_state(self, state: ChatState, content: str) -> ChatState:
        """添加 AI 消息到状态中"""
        try:
            # 检查是否有消息列表（聊天模式）
            messages = state.get("messages")
            if messages is not None:
                from langchain_core.messages import AIMessage
                ai_message = AIMessage(content=content)
                messages_list = messages or []
                return {
                    **state,
                    "messages": messages_list + [ai_message]
                }
            # 如果没有消息列表，直接返回原状态（任务模式）
            return state
        except Exception as e:
            print(f"⚠️ 添加 AI 消息失败: {e}")
            return state
    
    def _create_error_state(self, state: ChatState, error: Exception, node_name: str) -> ChatState:
        """创建标准化的错误状态"""
        error_msg = f"{node_name} error: {str(error)}"
        print(f"❌ {node_name} 失败: {error_msg}")
        print(f"🔍 错误详情: {repr(error)}")
        
        return {
            **state,
            "errors": state.get("errors", []) + [error_msg]
        }
    
    
    
    def _apply_extracted_info(self, state: ChatState, router_decision) -> ChatState:
        """应用提取的信息到状态"""
        updated_state = {**state}
        
        # 更新告警信息
        if router_decision.extracted_alerts and isinstance(router_decision.extracted_alerts, dict):
            if not updated_state.get("alert_info"):
                # 使用新路由器提取的告警信息创建 AlertInfo
                try:
                    updated_state["alert_info"] = AlertInfo(**router_decision.extracted_alerts)
                except Exception as e:
                    print(f"⚠️ 创建AlertInfo失败: {e}")
                    # 创建基本的AlertInfo
                    updated_state["alert_info"] = self._create_basic_alert_info(router_decision.extracted_alerts)
        
        # 合并症状
        if router_decision.extracted_symptoms and isinstance(router_decision.extracted_symptoms, list):
            existing_symptoms = updated_state.get("symptoms") or []
            updated_state["symptoms"] = list(set(existing_symptoms + router_decision.extracted_symptoms))
        
        # 合并上下文
        if router_decision.extracted_context and isinstance(router_decision.extracted_context, dict):
            existing_context = updated_state.get("context") or {}
            updated_state["context"] = {**existing_context, **router_decision.extracted_context}
        
        return updated_state
    
    def _is_valid_alert_data(self, alert_data: Dict[str, Any]) -> bool:
        """检查告警数据是否包含必需字段"""
        required_fields = ["alert_id", "timestamp", "severity", "source", "message"]
        return all(field in alert_data for field in required_fields)
    
    def _create_basic_alert_info(self, extracted_alerts: Dict[str, Any]) -> AlertInfo:
        """从提取的告警数据创建基本的 AlertInfo"""
        from datetime import datetime
        
        # 提取可用的信息
        items = extracted_alerts.get("items", [])
        alert_message = items[0] if items else "未知告警"
        
        return AlertInfo(
            alert_id=f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            severity="medium",  # 默认中等严重程度
            source="intelligent_router",
            message=alert_message,
            metrics={},
            tags=["extracted", "router"]
        )
    
    def _create_routing_analysis_message(self, router_decision) -> str:
        """创建路由分析消息"""
        return f"🧠 **智能路由分析完成**\n\n" \
               f"🎯 **下一步任务**: {router_decision.next_task}\n" \
               f"📊 **置信度**: {router_decision.confidence:.2f}\n" \
               f"🚨 **紧急程度**: {router_decision.urgency_level}\n" \
               f"💡 **用户消息**: {router_decision.user_message}\n\n" \
               f"⏰ **分析时间**: {datetime.now().strftime('%H:%M:%S')}"
    
    def _infer_current_task(self, state: ChatState) -> str:
        """从消息历史推断当前任务"""
        messages = state.get("messages", [])
        if not messages:
            return "未知"
        
        # 从最近的 AI 消息推断当前任务
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == "ai" and hasattr(msg, 'content'):
                content = msg.content
                if "告警分析完成" in content:
                    return "告警处理"
                elif "诊断完成" in content:
                    return "问题诊断"
                elif "规划完成" in content:
                    return "行动规划"
                elif "执行完成" in content:
                    return "执行行动"
                elif "报告生成完成" in content:
                    return "报告生成"
                elif "目标任务" in content:
                    if "execute_actions" in content:
                        return "执行行动"
                    elif "plan_actions" in content:
                        return "行动规划"
                    elif "diagnose_issue" in content:
                        return "问题诊断"
                    elif "process_alert" in content:
                        return "告警处理"
                    elif "generate_report" in content:
                        return "报告生成"
        
        return "智能分析"
    
    async def _redirect_to_collect_info(self, state: ChatState, original_node: str, reason: str) -> ChatState:
        """统一的回跳到 collect_info 处理"""
        
        # 添加回跳提示消息 - 让 collect_info 了解需要收集什么信息
        redirect_message = f"⚠️ **需要补充信息**\n\n" \
                          f"💬 **原因**: {reason}\n" \
                          f"🎯 **原始任务**: {original_node}\n" \
                          f"🔄 **正在收集信息**: 请提供相关信息以继续处理\n\n" \
                          f"⏰ **时间**: {datetime.now().strftime('%H:%M:%S')}"
        
        # 更新状态，准备回跳
        updated_state = self._add_ai_message_to_state(state, redirect_message)
        
        # 直接调用 collect_info 节点
        return await self._collect_info_node(updated_state)

    
    # ==================== 条件函数 ====================
    
    def _route_condition(self, state: ChatState) -> str:
        """基于状态数据的简化路由条件"""
        # 优先检查Router节点设置的目标节点
        target_node = state.get("_target_node")
        if target_node:
            # 检查目标节点是否缺少前置条件
            if target_node == "process_alert":
                if not state.get("alert_info"):
                    return "collect_info"
                return "process_alert"
            elif target_node == "diagnose_issue":
                if not state.get("alert_info") and not state.get("symptoms"):
                    return "collect_info"
                return "diagnose_issue"
            elif target_node == "plan_actions":
                if not state.get("diagnostic_result"):
                    return "collect_info"
                return "plan_actions"
            elif target_node == "execute_actions":
                if not state.get("action_plan"):
                    return "collect_info"
                return "execute_actions"
            elif target_node == "generate_report":
                return "generate_report"
            elif target_node == "collect_info":
                return "collect_info"
        
        # 从消息历史中分析路由决策（兼容性代码）
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'content') and "🧠 **智能路由分析完成**" in msg.content:
                content = msg.content
                if "collect_info" in content:
                    return "collect_info"
                elif "execute_actions" in content:
                    return "execute_actions"
                elif "plan_actions" in content:
                    return "plan_actions"
                elif "diagnose_issue" in content:
                    return "diagnose_issue"
                elif "process_alert" in content:
                    return "process_alert"
                elif "generate_report" in content:
                    return "generate_report"
        
        # 基于现有状态数据进行路由（兼容性代码）
        if state.get("action_plan") and not state.get("execution_result"):
            return "execute_actions"
        elif state.get("diagnostic_result") and not state.get("action_plan"):
            return "plan_actions"
        elif (state.get("symptoms") or state.get("alert_info")) and not state.get("diagnostic_result"):
            return "diagnose_issue"
        elif state.get("alert_info") and not state.get("analysis_result"):
            return "process_alert"
        else:
            return "collect_info"  # 默认先收集信息
    
    
    
    def _has_backjump_request(self, state: ChatState) -> bool:
        """检查是否有回跳请求"""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content') and "⚠️ **前置条件缺失**" in last_message.content:
                return True
        return False
    
    def _process_alert_condition(self, state: ChatState) -> str:
        """process_alert 节点的条件判断"""
        if state.get("errors"):
            return "error"
        
        # 检查是否有告警信息
        if state.get("alert_info"):
            # 正常情况：有告警信息，继续到下一步 diagnose_issue
            return "diagnose_issue"
        else:
            # 异常情况：缺少告警信息，需要收集信息
            return "collect_info"
    
    def _diagnose_issue_condition(self, state: ChatState) -> str:
        """diagnose_issue 节点的条件判断"""
        if state.get("errors"):
            return "error"
        
        # 检查前置条件：需要告警信息或症状
        if not state.get("alert_info") and not state.get("symptoms"):
            # 异常情况：缺少前置条件，需要收集信息
            return "collect_info"
        
        # 检查是否有诊断结果
        if state.get("diagnostic_result"):
            # 正常情况：有诊断结果，继续到下一步 plan_actions
            return "plan_actions"
        else:
            # 异常情况：诊断失败，需要收集更多信息
            return "collect_info"
    
    def _plan_actions_condition(self, state: ChatState) -> str:
        """plan_actions 节点的条件判断"""
        if state.get("errors"):
            return "error"
        
        # 检查前置条件：需要诊断结果
        if not state.get("diagnostic_result"):
            # 异常情况：缺少前置条件，需要收集信息
            return "collect_info"
        
        # 检查是否有行动计划
        if state.get("action_plan"):
            # 正常情况：有行动计划，继续到下一步 execute_actions
            return "execute_actions"
        else:
            # 异常情况：规划失败，需要收集更多信息
            return "collect_info"
    
    def _execute_actions_condition(self, state: ChatState) -> str:
        """execute_actions 节点的条件判断"""
        if state.get("errors"):
            return "error"
        
        if self._has_backjump_request(state):
            return "router"
        
        # 正常情况下：如果有执行结果且启用报告，继续到报告生成
        if state.get("execution_result") and self.config.enable_reporting:
            return "generate_report"
        elif state.get("execution_result"):
            # 有执行结果但不启用报告，直接结束
            return "finalize"
        
        # 如果没有执行结果，跳转到collect_info收集信息
        return "collect_info"
    
    def _generate_report_condition(self, state: ChatState) -> str:
        """generate_report 节点的条件判断"""
        if state.get("errors"):
            return "error"
        
        if self._has_backjump_request(state):
            return "router"
        
        # 正常情况下：如果有报告，结束流程
        if state.get("report"):
            return "finalize"
        
        # 如果没有报告，跳转到collect_info收集信息
        return "collect_info"
    
    def _task_completion_condition(self, state: ChatState) -> str:
        """统一的任务完成条件（保留用于兼容性）"""
        # 检查是否有严重错误
        if state.get("errors"):
            return "error"
        
        # 检查是否需要继续处理（基于状态判断）
        # 如果刚完成诊断，且没有行动计划，继续到规划
        if state.get("diagnostic_result") and not state.get("action_plan"):
            return "router"
        
        # 如果刚完成规划，且没有执行结果，继续到执行
        if state.get("action_plan") and not state.get("execution_result"):
            return "router"
        
        # 如果刚完成执行，且启用报告，继续到报告
        if state.get("execution_result") and not state.get("report") and self.config.enable_reporting:
            return "router"
        
        # 其他情况，结束流程
        return "finalize"
    
    def _error_recovery_condition(self, _: ChatState) -> str:
        """错误恢复条件"""
        # 简化错误恢复逻辑，直接结束
        return "finalize"
    


# ==================== LangGraph Studio 集成 ====================

# 创建默认的智能体配置供 Studio 使用
_default_studio_config = AgentConfig(
    agent_id="ops_agent_studio",
    agent_type="general", 
    specialization="studio_demo",
    enable_reporting=True,
    auto_execution=True 
)

# 创建智能体实例并编译图
_studio_agent = IntelligentOpsAgent(_default_studio_config)
graph = _studio_agent.compile()

# 同时导出智能体实例供其他用途
agent = _studio_agent