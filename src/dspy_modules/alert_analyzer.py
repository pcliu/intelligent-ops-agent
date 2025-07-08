import dspy
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class AlertInfo(BaseModel):
    """告警信息模型"""
    alert_id: str
    timestamp: str
    severity: str
    source: str
    message: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class AlertAnalysisResult(BaseModel):
    """告警分析结果模型"""
    alert_id: str
    priority: str  # critical, high, medium, low
    category: str  # network, cpu, memory, disk, application
    urgency_score: float  # 0-1 紧急程度评分
    related_alerts: List[str] = Field(default_factory=list)
    root_cause_hints: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class AlertClassification(dspy.Signature):
    """告警分类签名"""
    alert_message: str = dspy.InputField(desc="告警消息内容")
    severity: str = dspy.InputField(desc="告警严重程度")
    source: str = dspy.InputField(desc="告警来源系统")
    
    category: str = dspy.OutputField(desc="告警分类: network, cpu, memory, disk, application")
    priority: str = dspy.OutputField(desc="优先级: critical, high, medium, low")
    urgency_score: float = dspy.OutputField(desc="紧急程度评分 0-1")


class AlertCorrelation(dspy.Signature):
    """告警关联分析签名"""
    current_alert: str = dspy.InputField(desc="当前告警信息")
    historical_alerts: str = dspy.InputField(desc="历史告警信息")
    
    related_alerts: str = dspy.OutputField(desc="相关告警ID列表，用逗号分隔")
    correlation_reason: str = dspy.OutputField(desc="关联原因分析")


class RootCauseHints(dspy.Signature):
    """根因提示签名"""
    alert_info: str = dspy.InputField(desc="告警详细信息")
    system_context: str = dspy.InputField(desc="系统上下文信息")
    
    root_cause_hints: str = dspy.OutputField(desc="可能的根因提示列表")
    confidence_score: float = dspy.OutputField(desc="置信度评分 0-1")


class AlertAnalyzer(dspy.Module):
    """告警分析模块
    
    功能：
    - 告警信息解析和分类
    - 紧急程度评估
    - 告警关联分析
    - 根因提示生成
    """
    
    def __init__(self):
        super().__init__()
        self.classify_alert = dspy.ChainOfThought(AlertClassification)
        self.correlate_alerts = dspy.ChainOfThought(AlertCorrelation)
        self.generate_hints = dspy.ChainOfThought(RootCauseHints)
        
    def forward(self, alert_info: AlertInfo, historical_alerts: List[AlertInfo] = None) -> AlertAnalysisResult:
        """
        分析告警信息
        
        Args:
            alert_info: 当前告警信息
            historical_alerts: 历史告警信息列表
            
        Returns:
            AlertAnalysisResult: 分析结果
        """
        # 1. 告警分类和优先级评估
        classification = self.classify_alert(
            alert_message=alert_info.message,
            severity=alert_info.severity,
            source=alert_info.source
        )
        
        # 2. 告警关联分析
        related_alerts = []
        if historical_alerts:
            historical_info = self._format_historical_alerts(historical_alerts)
            correlation = self.correlate_alerts(
                current_alert=self._format_alert_info(alert_info),
                historical_alerts=historical_info
            )
            related_alerts = correlation.related_alerts.split(',') if correlation.related_alerts else []
        
        # 3. 根因提示生成
        system_context = self._build_system_context(alert_info)
        hints = self.generate_hints(
            alert_info=self._format_alert_info(alert_info),
            system_context=system_context
        )
        
        # 4. 生成推荐行动
        recommended_actions = self._generate_recommended_actions(
            category=classification.category,
            priority=classification.priority,
            hints=hints.root_cause_hints
        )
        
        return AlertAnalysisResult(
            alert_id=alert_info.alert_id,
            priority=classification.priority,
            category=classification.category,
            urgency_score=classification.urgency_score,
            related_alerts=related_alerts,
            root_cause_hints=hints.root_cause_hints.split(',') if hints.root_cause_hints else [],
            recommended_actions=recommended_actions
        )
    
    def _format_alert_info(self, alert_info: AlertInfo) -> str:
        """格式化告警信息"""
        return f"""
        Alert ID: {alert_info.alert_id}
        Timestamp: {alert_info.timestamp}
        Severity: {alert_info.severity}
        Source: {alert_info.source}
        Message: {alert_info.message}
        Metrics: {alert_info.metrics}
        Tags: {alert_info.tags}
        """
    
    def _format_historical_alerts(self, alerts: List[AlertInfo]) -> str:
        """格式化历史告警信息"""
        formatted = []
        for alert in alerts[-10:]:  # 只取最近10条
            formatted.append(f"ID: {alert.alert_id}, Time: {alert.timestamp}, Message: {alert.message}")
        return "; ".join(formatted)
    
    def _build_system_context(self, alert_info: AlertInfo) -> str:
        """构建系统上下文"""
        return f"""
        System: {alert_info.source}
        Current metrics: {alert_info.metrics}
        Environment tags: {alert_info.tags}
        """
    
    def _generate_recommended_actions(self, category: str, priority: str, hints: str) -> List[str]:
        """生成推荐行动"""
        actions = []
        
        # 基于分类的通用建议
        if category == "cpu":
            actions.append("检查CPU使用率最高的进程")
            actions.append("验证系统负载是否异常")
        elif category == "memory":
            actions.append("检查内存使用情况")
            actions.append("查找可能的内存泄漏")
        elif category == "network":
            actions.append("检查网络连接状态")
            actions.append("验证网络配置")
        elif category == "disk":
            actions.append("检查磁盘空间使用情况")
            actions.append("查看磁盘I/O性能")
        elif category == "application":
            actions.append("检查应用程序日志")
            actions.append("验证应用程序配置")
        
        # 基于优先级的紧急处理
        if priority == "critical":
            actions.insert(0, "立即通知相关运维人员")
            actions.insert(1, "启动紧急响应流程")
        
        return actions


class AlertFilter(dspy.Module):
    """告警过滤模块"""
    
    def __init__(self):
        super().__init__()
        self.filter_signature = dspy.Signature(
            "alert_message, severity, source -> should_process: bool, filter_reason: str"
        )
        self.filter_predictor = dspy.ChainOfThought(self.filter_signature)
    
    def forward(self, alert_info: AlertInfo) -> bool:
        """
        过滤告警信息
        
        Args:
            alert_info: 告警信息
            
        Returns:
            bool: 是否应该处理此告警
        """
        result = self.filter_predictor(
            alert_message=alert_info.message,
            severity=alert_info.severity,
            source=alert_info.source
        )
        
        return result.should_process