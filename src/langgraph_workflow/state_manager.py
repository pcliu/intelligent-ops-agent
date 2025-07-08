from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from ..dspy_modules.alert_analyzer import AlertInfo, AlertAnalysisResult
from ..dspy_modules.diagnostic_agent import DiagnosticResult
from ..dspy_modules.action_planner import ActionPlan
from ..dspy_modules.report_generator import ExecutionResult, IncidentReport


class OpsState(TypedDict):
    """运维工作流状态"""
    # 告警相关
    current_alert: Optional[AlertInfo]
    alert_analysis: Optional[AlertAnalysisResult]
    historical_alerts: List[AlertInfo]
    
    # 诊断相关
    diagnostic_context: Dict[str, Any]
    diagnostic_result: Optional[DiagnosticResult]
    
    # 行动计划相关
    action_plan: Optional[ActionPlan]
    execution_result: Optional[ExecutionResult]
    
    # 系统状态
    system_metrics: Dict[str, Any]
    system_context: Dict[str, Any]
    
    # 工作流状态
    workflow_stage: str  # monitoring, alerting, diagnosis, planning, execution, reporting
    workflow_status: str  # active, paused, completed, failed
    
    # 历史和学习
    incident_history: List[Dict[str, Any]]
    learning_data: Dict[str, Any]
    
    # 报告
    incident_report: Optional[IncidentReport]
    
    # 元数据
    workflow_id: str
    start_time: datetime
    last_update: datetime
    
    # 错误处理
    errors: List[str]
    retry_count: int
    max_retries: int


class StateManager:
    """状态管理器"""
    
    def __init__(self):
        self.state_history: List[OpsState] = []
        self.current_state: Optional[OpsState] = None
    
    def initialize_state(self, workflow_id: str) -> OpsState:
        """初始化状态"""
        now = datetime.now()
        
        state = OpsState(
            current_alert=None,
            alert_analysis=None,
            historical_alerts=[],
            diagnostic_context={},
            diagnostic_result=None,
            action_plan=None,
            execution_result=None,
            system_metrics={},
            system_context={},
            workflow_stage="monitoring",
            workflow_status="active",
            incident_history=[],
            learning_data={},
            incident_report=None,
            workflow_id=workflow_id,
            start_time=now,
            last_update=now,
            errors=[],
            retry_count=0,
            max_retries=3
        )
        
        self.current_state = state
        self.state_history.append(state.copy())
        
        return state
    
    def update_state(self, state: OpsState, updates: Dict[str, Any]) -> OpsState:
        """更新状态"""
        for key, value in updates.items():
            if key in state:
                state[key] = value
        
        state["last_update"] = datetime.now()
        self.state_history.append(state.copy())
        
        return state
    
    def get_state_history(self) -> List[OpsState]:
        """获取状态历史"""
        return self.state_history.copy()
    
    def rollback_state(self, steps: int = 1) -> Optional[OpsState]:
        """回滚状态"""
        if len(self.state_history) > steps:
            target_state = self.state_history[-(steps + 1)]
            self.current_state = target_state.copy()
            return self.current_state
        return None
    
    def add_error(self, state: OpsState, error: str) -> OpsState:
        """添加错误"""
        state["errors"].append(error)
        state["last_update"] = datetime.now()
        return state
    
    def increment_retry(self, state: OpsState) -> OpsState:
        """增加重试次数"""
        state["retry_count"] += 1
        state["last_update"] = datetime.now()
        return state
    
    def reset_retry(self, state: OpsState) -> OpsState:
        """重置重试次数"""
        state["retry_count"] = 0
        state["last_update"] = datetime.now()
        return state
    
    def can_retry(self, state: OpsState) -> bool:
        """判断是否可以重试"""
        return state["retry_count"] < state["max_retries"]
    
    def transition_stage(self, state: OpsState, new_stage: str) -> OpsState:
        """转换工作流阶段"""
        valid_stages = ["monitoring", "alerting", "diagnosis", "planning", "execution", "reporting"]
        
        if new_stage in valid_stages:
            state["workflow_stage"] = new_stage
            state["last_update"] = datetime.now()
        else:
            raise ValueError(f"Invalid stage: {new_stage}")
        
        return state
    
    def update_workflow_status(self, state: OpsState, status: str) -> OpsState:
        """更新工作流状态"""
        valid_statuses = ["active", "paused", "completed", "failed"]
        
        if status in valid_statuses:
            state["workflow_status"] = status
            state["last_update"] = datetime.now()
        else:
            raise ValueError(f"Invalid status: {status}")
        
        return state
    
    def get_state_summary(self, state: OpsState) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "workflow_id": state["workflow_id"],
            "stage": state["workflow_stage"],
            "status": state["workflow_status"],
            "start_time": state["start_time"],
            "last_update": state["last_update"],
            "has_alert": state["current_alert"] is not None,
            "has_diagnosis": state["diagnostic_result"] is not None,
            "has_action_plan": state["action_plan"] is not None,
            "error_count": len(state["errors"]),
            "retry_count": state["retry_count"]
        }
    
    def validate_state(self, state: OpsState) -> List[str]:
        """验证状态"""
        errors = []
        
        # 检查必需字段
        required_fields = ["workflow_id", "workflow_stage", "workflow_status"]
        for field in required_fields:
            if not state.get(field):
                errors.append(f"Missing required field: {field}")
        
        # 检查状态一致性
        if state["workflow_stage"] == "alerting" and not state["current_alert"]:
            errors.append("Alert stage requires current_alert")
        
        if state["workflow_stage"] == "diagnosis" and not state["alert_analysis"]:
            errors.append("Diagnosis stage requires alert_analysis")
        
        if state["workflow_stage"] == "planning" and not state["diagnostic_result"]:
            errors.append("Planning stage requires diagnostic_result")
        
        if state["workflow_stage"] == "execution" and not state["action_plan"]:
            errors.append("Execution stage requires action_plan")
        
        # 检查重试次数
        if state["retry_count"] > state["max_retries"]:
            errors.append("Retry count exceeds maximum")
        
        return errors
    
    def cleanup_state(self, state: OpsState) -> OpsState:
        """清理状态"""
        # 清理历史数据，只保留最近的条目
        if len(state["historical_alerts"]) > 100:
            state["historical_alerts"] = state["historical_alerts"][-100:]
        
        if len(state["incident_history"]) > 50:
            state["incident_history"] = state["incident_history"][-50:]
        
        if len(state["errors"]) > 20:
            state["errors"] = state["errors"][-20:]
        
        state["last_update"] = datetime.now()
        return state
    
    def export_state(self, state: OpsState) -> Dict[str, Any]:
        """导出状态"""
        return {
            "workflow_id": state["workflow_id"],
            "stage": state["workflow_stage"],
            "status": state["workflow_status"],
            "start_time": state["start_time"].isoformat(),
            "last_update": state["last_update"].isoformat(),
            "current_alert": state["current_alert"].dict() if state["current_alert"] else None,
            "alert_analysis": state["alert_analysis"].dict() if state["alert_analysis"] else None,
            "diagnostic_result": state["diagnostic_result"].dict() if state["diagnostic_result"] else None,
            "action_plan": state["action_plan"].dict() if state["action_plan"] else None,
            "execution_result": state["execution_result"].dict() if state["execution_result"] else None,
            "system_metrics": state["system_metrics"],
            "errors": state["errors"],
            "retry_count": state["retry_count"]
        }
    
    def import_state(self, state_data: Dict[str, Any]) -> OpsState:
        """导入状态"""
        now = datetime.now()
        
        return OpsState(
            current_alert=AlertInfo(**state_data["current_alert"]) if state_data.get("current_alert") else None,
            alert_analysis=AlertAnalysisResult(**state_data["alert_analysis"]) if state_data.get("alert_analysis") else None,
            historical_alerts=[],
            diagnostic_context=state_data.get("diagnostic_context", {}),
            diagnostic_result=DiagnosticResult(**state_data["diagnostic_result"]) if state_data.get("diagnostic_result") else None,
            action_plan=ActionPlan(**state_data["action_plan"]) if state_data.get("action_plan") else None,
            execution_result=ExecutionResult(**state_data["execution_result"]) if state_data.get("execution_result") else None,
            system_metrics=state_data.get("system_metrics", {}),
            system_context=state_data.get("system_context", {}),
            workflow_stage=state_data.get("workflow_stage", "monitoring"),
            workflow_status=state_data.get("workflow_status", "active"),
            incident_history=state_data.get("incident_history", []),
            learning_data=state_data.get("learning_data", {}),
            incident_report=None,
            workflow_id=state_data["workflow_id"],
            start_time=datetime.fromisoformat(state_data["start_time"]),
            last_update=datetime.fromisoformat(state_data["last_update"]),
            errors=state_data.get("errors", []),
            retry_count=state_data.get("retry_count", 0),
            max_retries=state_data.get("max_retries", 3)
        )