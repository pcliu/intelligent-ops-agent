import dspy
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from .diagnostic_agent import DiagnosticResult


class ActionType(str, Enum):
    """行动类型枚举"""
    RESTART_SERVICE = "restart_service"
    SCALE_RESOURCES = "scale_resources"
    UPDATE_CONFIG = "update_config"
    DEPLOY_PATCH = "deploy_patch"
    ROLLBACK = "rollback"
    NOTIFICATION = "notification"
    MONITORING = "monitoring"
    INVESTIGATION = "investigation"


class ActionStep(BaseModel):
    """行动步骤模型"""
    step_id: str
    action_type: ActionType
    description: str
    command: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: int = 300  # 超时时间（秒）
    rollback_command: Optional[str] = None
    validation_command: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high, critical


class ActionPlan(BaseModel):
    """行动计划模型"""
    plan_id: str
    incident_id: str
    priority: str
    estimated_duration: int  # 分钟
    risk_assessment: str
    approval_required: bool = False
    rollback_plan: List[ActionStep] = Field(default_factory=list)
    pre_checks: List[str] = Field(default_factory=list)
    post_checks: List[str] = Field(default_factory=list)
    steps: List[ActionStep] = Field(default_factory=list)
    notifications: List[str] = Field(default_factory=list)


class ActionGeneration(dspy.Signature):
    """行动生成签名"""
    root_cause: str = dspy.InputField(desc="根本原因")
    impact_assessment: str = dspy.InputField(desc="影响评估")
    system_context: str = dspy.InputField(desc="系统上下文")
    
    action_steps: str = dspy.OutputField(desc="行动步骤列表")
    estimated_duration: int = dspy.OutputField(desc="预计持续时间（分钟）")
    risk_level: str = dspy.OutputField(desc="风险级别")


class RiskAssessment(dspy.Signature):
    """风险评估签名"""
    action_plan: str = dspy.InputField(desc="行动计划")
    system_state: str = dspy.InputField(desc="系统状态")
    business_context: str = dspy.InputField(desc="业务上下文")
    
    risk_score: float = dspy.OutputField(desc="风险评分 0-1")
    risk_factors: str = dspy.OutputField(desc="风险因素列表")
    mitigation_strategies: str = dspy.OutputField(desc="缓解策略")


class RollbackPlanning(dspy.Signature):
    """回滚计划签名"""
    action_steps: str = dspy.InputField(desc="执行步骤")
    current_state: str = dspy.InputField(desc="当前状态")
    
    rollback_steps: str = dspy.OutputField(desc="回滚步骤列表")
    rollback_conditions: str = dspy.OutputField(desc="回滚触发条件")


class ActionPlanner(dspy.Module):
    """行动规划器模块
    
    功能：
    - 修复策略生成
    - 风险评估
    - 执行步骤规划
    - 回滚计划制定
    """
    
    def __init__(self):
        super().__init__()
        self.action_generator = dspy.ChainOfThought(ActionGeneration)
        self.risk_assessor = dspy.ChainOfThought(RiskAssessment)
        self.rollback_planner = dspy.ChainOfThought(RollbackPlanning)
        
    def forward(self, diagnostic_result: DiagnosticResult, system_context: Dict[str, Any]) -> ActionPlan:
        """
        生成行动计划
        
        Args:
            diagnostic_result: 诊断结果
            system_context: 系统上下文
            
        Returns:
            ActionPlan: 行动计划
        """
        # 1. 生成行动步骤
        action_result = self.action_generator(
            root_cause=diagnostic_result.root_cause,
            impact_assessment=diagnostic_result.impact_assessment,
            system_context=self._format_system_context(system_context)
        )
        
        # 2. 风险评估
        risk_result = self.risk_assessor(
            action_plan=action_result.action_steps,
            system_state=self._format_system_state(system_context),
            business_context=diagnostic_result.business_impact
        )
        
        # 3. 生成回滚计划
        rollback_result = self.rollback_planner(
            action_steps=action_result.action_steps,
            current_state=self._format_system_state(system_context)
        )
        
        # 4. 解析行动步骤
        steps = self._parse_action_steps(action_result.action_steps)
        rollback_steps = self._parse_rollback_steps(rollback_result.rollback_steps)
        
        # 5. 生成完整行动计划
        return ActionPlan(
            plan_id=f"plan_{diagnostic_result.incident_id}",
            incident_id=diagnostic_result.incident_id,
            priority=diagnostic_result.impact_assessment,
            estimated_duration=action_result.estimated_duration,
            risk_assessment=risk_result.risk_factors,
            approval_required=self._requires_approval(risk_result.risk_score),
            rollback_plan=rollback_steps,
            pre_checks=self._generate_pre_checks(diagnostic_result),
            post_checks=self._generate_post_checks(diagnostic_result),
            steps=steps,
            notifications=self._generate_notifications(diagnostic_result)
        )
    
    def _format_system_context(self, context: Dict[str, Any]) -> str:
        """格式化系统上下文"""
        formatted = []
        for key, value in context.items():
            formatted.append(f"{key}: {value}")
        return "; ".join(formatted)
    
    def _format_system_state(self, context: Dict[str, Any]) -> str:
        """格式化系统状态"""
        return self._format_system_context(context)
    
    def _parse_action_steps(self, steps_text: str) -> List[ActionStep]:
        """解析行动步骤"""
        steps = []
        lines = steps_text.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                step = ActionStep(
                    step_id=f"step_{i+1}",
                    action_type=ActionType.INVESTIGATION,  # 默认类型
                    description=line.strip(),
                    command=self._generate_command(line.strip()),
                    parameters={},
                    timeout=300,
                    risk_level="medium"
                )
                steps.append(step)
        
        return steps
    
    def _parse_rollback_steps(self, rollback_text: str) -> List[ActionStep]:
        """解析回滚步骤"""
        steps = []
        lines = rollback_text.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                step = ActionStep(
                    step_id=f"rollback_{i+1}",
                    action_type=ActionType.ROLLBACK,
                    description=line.strip(),
                    command=self._generate_rollback_command(line.strip()),
                    parameters={},
                    timeout=300,
                    risk_level="low"
                )
                steps.append(step)
        
        return steps
    
    def _generate_command(self, description: str) -> str:
        """根据描述生成命令"""
        # 简单的命令生成逻辑
        if "restart" in description.lower():
            return "systemctl restart service_name"
        elif "scale" in description.lower():
            return "kubectl scale deployment deployment_name --replicas=N"
        elif "update" in description.lower():
            return "kubectl apply -f config.yaml"
        else:
            return "echo 'Manual action required'"
    
    def _generate_rollback_command(self, description: str) -> str:
        """生成回滚命令"""
        if "restart" in description.lower():
            return "systemctl stop service_name"
        elif "scale" in description.lower():
            return "kubectl scale deployment deployment_name --replicas=1"
        elif "update" in description.lower():
            return "kubectl rollout undo deployment/deployment_name"
        else:
            return "echo 'Manual rollback required'"
    
    def _requires_approval(self, risk_score: float) -> bool:
        """判断是否需要审批"""
        return risk_score > 0.7
    
    def _generate_pre_checks(self, diagnostic_result: DiagnosticResult) -> List[str]:
        """生成预检查项"""
        checks = [
            "验证系统备份完整性",
            "确认业务流量路由正常",
            "检查依赖服务状态"
        ]
        
        # 基于诊断结果添加特定检查
        if "database" in diagnostic_result.root_cause.lower():
            checks.append("验证数据库连接")
        if "network" in diagnostic_result.root_cause.lower():
            checks.append("检查网络连通性")
        
        return checks
    
    def _generate_post_checks(self, diagnostic_result: DiagnosticResult) -> List[str]:
        """生成后检查项"""
        checks = [
            "验证服务健康状态",
            "确认业务功能正常",
            "检查性能指标恢复"
        ]
        
        # 基于受影响组件添加特定检查
        for component in diagnostic_result.affected_components:
            checks.append(f"验证{component}组件状态")
        
        return checks
    
    def _generate_notifications(self, diagnostic_result: DiagnosticResult) -> List[str]:
        """生成通知列表"""
        notifications = []
        
        # 基于影响级别确定通知范围
        if diagnostic_result.impact_assessment == "critical":
            notifications.extend([
                "通知运维团队负责人",
                "通知业务负责人",
                "更新事件管理系统"
            ])
        elif diagnostic_result.impact_assessment == "high":
            notifications.extend([
                "通知运维团队",
                "更新事件管理系统"
            ])
        else:
            notifications.append("记录到运维日志")
        
        return notifications


class ActionValidator(dspy.Module):
    """行动验证器"""
    
    def __init__(self):
        super().__init__()
        self.validation_signature = dspy.Signature(
            "action_plan, system_constraints -> is_valid: bool, validation_errors: str"
        )
        self.validator = dspy.ChainOfThought(self.validation_signature)
    
    def forward(self, action_plan: ActionPlan, system_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证行动计划
        
        Args:
            action_plan: 行动计划
            system_constraints: 系统约束
            
        Returns:
            Dict: 验证结果
        """
        result = self.validator(
            action_plan=str(action_plan),
            system_constraints=str(system_constraints)
        )
        
        return {
            "is_valid": result.is_valid,
            "validation_errors": result.validation_errors.split(',') if result.validation_errors else []
        }


class ActionOptimizer(dspy.Module):
    """行动优化器"""
    
    def __init__(self):
        super().__init__()
        self.optimization_signature = dspy.Signature(
            "action_plan, performance_data -> optimized_plan: str, improvements: str"
        )
        self.optimizer = dspy.ChainOfThought(self.optimization_signature)
    
    def forward(self, action_plan: ActionPlan, performance_data: Dict[str, Any]) -> ActionPlan:
        """
        优化行动计划
        
        Args:
            action_plan: 原始行动计划
            performance_data: 性能数据
            
        Returns:
            ActionPlan: 优化后的行动计划
        """
        result = self.optimizer(
            action_plan=str(action_plan),
            performance_data=str(performance_data)
        )
        
        # 这里应该解析优化结果并更新行动计划
        # 为简化示例，直接返回原计划
        return action_plan