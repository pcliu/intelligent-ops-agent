"""
智能路由器模块（重构版本）

基于 DSPy 的简化智能路由器，专注于信息提取和下一步路由决策
"""

import dspy
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MessageAnalysisSignature(dspy.Signature):
    """第一步：分析消息信息，提取告警、症状和上下文"""
    message_content: str = dspy.InputField(desc="用户消息内容")
    message_history: str = dspy.InputField(desc="消息历史摘要")
    current_state: str = dspy.InputField(desc="当前状态摘要")
    
    # 告警信息提取
    has_alert_info: bool = dspy.OutputField(desc="是否包含告警信息")
    alert_severity: str = dspy.OutputField(desc="告警严重程度: low|medium|high|critical")
    alert_source: str = dspy.OutputField(desc="告警来源或系统")
    alert_message: str = dspy.OutputField(desc="告警描述")
    alert_metrics: str = dspy.OutputField(desc="相关指标JSON格式")
    
    # 症状信息提取
    identified_symptoms: str = dspy.OutputField(desc="识别的症状列表JSON格式")
    
    # 上下文信息提取
    ops_context: str = dspy.OutputField(desc="运维上下文信息JSON格式: {environment, urgency, affected_systems}")
    
    reasoning: str = dspy.OutputField(desc="信息提取的推理过程")


class NextStepRoutingSignature(dspy.Signature):
    """第二步：基于当前信息决定下一步路由"""
    user_intent: str = dspy.InputField(desc="用户主要意图")
    available_info: str = dspy.InputField(desc="当前可用信息摘要")
    
    next_task: str = dspy.OutputField(desc="下一步任务: process_alert|diagnose_issue|plan_actions|execute_actions|generate_report|collect_info")
    urgency_level: str = dspy.OutputField(desc="紧急程度: low|medium|high|critical")
    confidence: float = dspy.OutputField(desc="路由决策置信度 0-1")
    user_message: str = dspy.OutputField(desc="给用户的反馈消息")
    reasoning: str = dspy.OutputField(desc="路由决策的推理过程")


class RouterDecision(BaseModel):
    """路由决策结果"""
    # 提取的信息
    extracted_alerts: Optional[Dict[str, Any]] = None
    extracted_symptoms: Optional[List[str]] = None
    extracted_context: Optional[Dict[str, Any]] = None
    
    # 路由决策
    next_task: str
    urgency_level: str
    confidence: float
    user_message: str
    
    # 推理过程
    analysis_reasoning: str = ""
    routing_reasoning: str = ""
    
    # 元信息
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    router_version: str = "3.0"


class IntelligentRouter(dspy.Module):
    """简化的智能路由器 - 专注于信息提取和下一步路由"""
    
    def __init__(self):
        super().__init__()
        self.message_analyzer = dspy.ChainOfThought(MessageAnalysisSignature)
        self.next_step_router = dspy.ChainOfThought(NextStepRoutingSignature)
    
    def forward(self, user_input: str, current_state: Dict[str, Any]) -> RouterDecision:
        """处理用户输入，提取信息并决定下一步"""
        
        # Step 1: 分析消息，提取告警、症状和上下文信息
        analysis = self.message_analyzer(
            message_content=user_input,
            message_history=self._extract_recent_messages(current_state),
            current_state=self._serialize_state(current_state)
        )
        
        # Step 2: 基于分析结果决定下一步路由
        available_info = self._build_available_info_summary(analysis, current_state)
        user_intent = self._infer_user_intent(user_input, analysis)
        
        routing = self.next_step_router(
            user_intent=user_intent,
            available_info=available_info
        )
        
        return self._build_router_decision(analysis, routing, user_input)
    
    def _extract_recent_messages(self, state: Dict[str, Any]) -> str:
        """提取最近的消息历史"""
        try:
            messages = state.get("messages", [])
            if not messages:
                return "No message history"
            
            # 取最近3条消息
            recent_messages = messages[-3:]
            msg_summary = []
            
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    msg_type = getattr(msg, 'type', 'unknown')
                    content = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                    msg_summary.append(f"{msg_type}: {content}")
            
            return " | ".join(msg_summary)
        except Exception as e:
            return f"Message extraction error: {str(e)}"
    
    def _serialize_state(self, state: Dict[str, Any]) -> str:
        """序列化当前状态"""
        try:
            summary = {
                "has_alert": bool(state.get("alert_info")),
                "has_symptoms": bool(state.get("symptoms")),
                "has_context": bool(state.get("context")),
                "has_analysis": bool(state.get("analysis_result")),
                "has_diagnostic": bool(state.get("diagnostic_result")),
                "has_action_plan": bool(state.get("action_plan")),
                "has_execution": bool(state.get("execution_result")),
                "has_report": bool(state.get("report")),
                "errors": state.get("errors", [])
            }
            return json.dumps(summary, ensure_ascii=False)
        except Exception as e:
            return f"State serialization error: {str(e)}"
    
    def _build_available_info_summary(self, analysis, current_state: Dict[str, Any]) -> str:
        """构建可用信息摘要"""
        info_parts = []
        
        if analysis.has_alert_info:
            info_parts.append(f"告警信息: {analysis.alert_severity} - {analysis.alert_message}")
        
        if analysis.identified_symptoms:
            info_parts.append(f"症状: {analysis.identified_symptoms}")
        
        if analysis.ops_context:
            info_parts.append(f"上下文: {analysis.ops_context}")
        
        # 添加现有状态信息
        if current_state.get("diagnostic_result"):
            info_parts.append("已有诊断结果")
        
        if current_state.get("action_plan"):
            info_parts.append("已有行动计划")
        
        if current_state.get("execution_result"):
            info_parts.append("已有执行结果")
        
        return " | ".join(info_parts) if info_parts else "No specific information available"
    
    def _infer_user_intent(self, user_input: str, analysis) -> str:
        """推断用户意图"""
        input_lower = user_input.lower()
        
        # 关键词匹配
        if any(keyword in input_lower for keyword in ["告警", "alert", "监控", "指标"]):
            return "alert_processing"
        elif any(keyword in input_lower for keyword in ["诊断", "分析", "原因", "diagnose"]):
            return "issue_diagnosis"
        elif any(keyword in input_lower for keyword in ["计划", "方案", "plan"]):
            return "action_planning"
        elif any(keyword in input_lower for keyword in ["执行", "运行", "重启", "execute", "run"]):
            return "action_execution"
        elif any(keyword in input_lower for keyword in ["报告", "总结", "report"]):
            return "report_generation"
        else:
            # 基于分析结果推断
            if analysis.has_alert_info:
                return "alert_processing"
            else:
                return "issue_diagnosis"
    
    def _build_router_decision(self, analysis, routing, user_input: str) -> RouterDecision:
        """构建路由决策结果"""
        try:
            # 构建告警信息
            alert_info = None
            if analysis.has_alert_info:
                alert_info = {
                    "alert_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "timestamp": datetime.now().isoformat(),
                    "severity": analysis.alert_severity,
                    "source": analysis.alert_source or "user_input",
                    "message": analysis.alert_message,
                    "metrics": self._parse_json_safe(analysis.alert_metrics),
                    "tags": ["extracted_from_message"]
                }
            
            # 构建症状列表
            symptoms = self._parse_json_safe(analysis.identified_symptoms)
            if isinstance(symptoms, str):
                symptoms = [symptoms]
            elif not isinstance(symptoms, list):
                symptoms = None
            
            # 构建上下文信息
            context = self._parse_json_safe(analysis.ops_context)
            if not isinstance(context, dict):
                context = None
            
            return RouterDecision(
                extracted_alerts=alert_info,
                extracted_symptoms=symptoms,
                extracted_context=context,
                next_task=routing.next_task,
                urgency_level=routing.urgency_level,
                confidence=routing.confidence,
                user_message=routing.user_message,
                analysis_reasoning=analysis.reasoning,
                routing_reasoning=routing.reasoning
            )
        except Exception as e:
            print(f"⚠️ 路由决策构建失败: {str(e)}")
            return RouterDecision(
                next_task="diagnose_issue",
                urgency_level="medium",
                confidence=0.3,
                user_message=f"路由决策构建失败，使用默认诊断流程",
                routing_reasoning=f"Router error: {str(e)}"
            )
    
    def _parse_json_safe(self, json_str: str) -> Any:
        """安全解析JSON字符串"""
        try:
            return json.loads(json_str)
        except:
            return json_str if json_str else None