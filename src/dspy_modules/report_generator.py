import dspy
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from .diagnostic_agent import DiagnosticResult
from .action_planner import ActionPlan


class ExecutionResult(BaseModel):
    """执行结果模型"""
    plan_id: str
    execution_start: datetime
    execution_end: datetime
    status: str  # success, failed, partial
    executed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    rollback_executed: bool = False
    final_state: Dict[str, Any] = Field(default_factory=dict)


class IncidentReport(BaseModel):
    """事件报告模型"""
    incident_id: str
    title: str
    summary: str
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    root_cause_analysis: str
    impact_analysis: str
    resolution_summary: str
    lessons_learned: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PerformanceReport(BaseModel):
    """性能报告模型"""
    report_id: str
    period: str
    system_health: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    trend_analysis: str
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_suggestions: List[str] = Field(default_factory=list)


class ReportGeneration(dspy.Signature):
    """报告生成签名"""
    incident_data: str = dspy.InputField(desc="事件数据")
    diagnostic_result: str = dspy.InputField(desc="诊断结果")
    execution_result: str = dspy.InputField(desc="执行结果")
    
    report_summary: str = dspy.OutputField(desc="报告摘要")
    key_findings: str = dspy.OutputField(desc="关键发现")
    recommendations: str = dspy.OutputField(desc="建议列表")


class TimelineGeneration(dspy.Signature):
    """时间线生成签名"""
    incident_events: str = dspy.InputField(desc="事件序列")
    execution_events: str = dspy.InputField(desc="执行序列")
    
    timeline: str = dspy.OutputField(desc="时间线描述")
    critical_moments: str = dspy.OutputField(desc="关键时刻")


class LessonsLearned(dspy.Signature):
    """经验教训签名"""
    incident_analysis: str = dspy.InputField(desc="事件分析")
    resolution_process: str = dspy.InputField(desc="解决过程")
    
    lessons: str = dspy.OutputField(desc="经验教训列表")
    improvement_areas: str = dspy.OutputField(desc="改进领域")


class ReportGenerator(dspy.Module):
    """报告生成器模块
    
    功能：
    - 事件报告生成
    - 性能分析报告
    - 趋势分析
    - 优化建议
    """
    
    def __init__(self):
        super().__init__()
        self.report_generator = dspy.ChainOfThought(ReportGeneration)
        self.timeline_generator = dspy.ChainOfThought(TimelineGeneration)
        self.lessons_analyzer = dspy.ChainOfThought(LessonsLearned)
        
    def generate_incident_report(self, 
                                diagnostic_result: DiagnosticResult,
                                action_plan: ActionPlan,
                                execution_result: ExecutionResult) -> IncidentReport:
        """
        生成事件报告
        
        Args:
            diagnostic_result: 诊断结果
            action_plan: 行动计划
            execution_result: 执行结果
            
        Returns:
            IncidentReport: 事件报告
        """
        # 1. 生成报告主体
        report_result = self.report_generator(
            incident_data=self._format_incident_data(diagnostic_result),
            diagnostic_result=str(diagnostic_result),
            execution_result=str(execution_result)
        )
        
        # 2. 生成时间线
        timeline_result = self.timeline_generator(
            incident_events=self._format_incident_events(diagnostic_result),
            execution_events=self._format_execution_events(execution_result)
        )
        
        # 3. 生成经验教训
        lessons_result = self.lessons_analyzer(
            incident_analysis=str(diagnostic_result),
            resolution_process=str(execution_result)
        )
        
        # 4. 构建完整报告
        return IncidentReport(
            incident_id=diagnostic_result.incident_id,
            title=self._generate_title(diagnostic_result),
            summary=report_result.report_summary,
            timeline=self._parse_timeline(timeline_result.timeline),
            root_cause_analysis=diagnostic_result.root_cause,
            impact_analysis=diagnostic_result.business_impact,
            resolution_summary=self._generate_resolution_summary(execution_result),
            lessons_learned=lessons_result.lessons.split(',') if lessons_result.lessons else [],
            recommendations=report_result.recommendations.split(',') if report_result.recommendations else [],
            metrics=self._calculate_metrics(diagnostic_result, execution_result)
        )
    
    def generate_performance_report(self, 
                                   system_metrics: Dict[str, Any],
                                   time_period: str) -> PerformanceReport:
        """
        生成性能报告
        
        Args:
            system_metrics: 系统指标
            time_period: 时间周期
            
        Returns:
            PerformanceReport: 性能报告
        """
        # 性能报告生成逻辑
        trend_analysis = self._analyze_trends(system_metrics)
        anomalies = self._detect_anomalies(system_metrics)
        suggestions = self._generate_optimization_suggestions(system_metrics)
        
        return PerformanceReport(
            report_id=f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            period=time_period,
            system_health=self._assess_system_health(system_metrics),
            performance_metrics=system_metrics,
            trend_analysis=trend_analysis,
            anomalies=anomalies,
            optimization_suggestions=suggestions
        )
    
    def _format_incident_data(self, diagnostic_result: DiagnosticResult) -> str:
        """格式化事件数据"""
        return f"""
        Incident ID: {diagnostic_result.incident_id}
        Root Cause: {diagnostic_result.root_cause}
        Impact: {diagnostic_result.business_impact}
        Affected Components: {diagnostic_result.affected_components}
        Confidence Score: {diagnostic_result.confidence_score}
        """
    
    def _format_incident_events(self, diagnostic_result: DiagnosticResult) -> str:
        """格式化事件序列"""
        events = []
        events.append(f"Incident detected: {diagnostic_result.incident_id}")
        events.append(f"Root cause identified: {diagnostic_result.root_cause}")
        events.append(f"Impact assessed: {diagnostic_result.business_impact}")
        return "; ".join(events)
    
    def _format_execution_events(self, execution_result: ExecutionResult) -> str:
        """格式化执行序列"""
        events = []
        events.append(f"Execution started: {execution_result.execution_start}")
        for step in execution_result.executed_steps:
            events.append(f"Step executed: {step}")
        events.append(f"Execution ended: {execution_result.execution_end}")
        return "; ".join(events)
    
    def _generate_title(self, diagnostic_result: DiagnosticResult) -> str:
        """生成报告标题"""
        return f"{diagnostic_result.incident_id} - {diagnostic_result.root_cause}"
    
    def _generate_resolution_summary(self, execution_result: ExecutionResult) -> str:
        """生成解决方案摘要"""
        if execution_result.status == "success":
            return f"Successfully resolved incident. Executed {len(execution_result.executed_steps)} steps."
        elif execution_result.status == "failed":
            return f"Failed to resolve incident. {len(execution_result.failed_steps)} steps failed."
        else:
            return f"Partially resolved incident. {len(execution_result.executed_steps)} steps succeeded, {len(execution_result.failed_steps)} failed."
    
    def _parse_timeline(self, timeline_text: str) -> List[Dict[str, Any]]:
        """解析时间线"""
        timeline = []
        events = timeline_text.split(';')
        
        for i, event in enumerate(events):
            if event.strip():
                timeline.append({
                    "sequence": i + 1,
                    "timestamp": datetime.now().isoformat(),
                    "event": event.strip(),
                    "type": "action"
                })
        
        return timeline
    
    def _calculate_metrics(self, diagnostic_result: DiagnosticResult, execution_result: ExecutionResult) -> Dict[str, Any]:
        """计算指标"""
        duration = (execution_result.execution_end - execution_result.execution_start).total_seconds()
        
        return {
            "resolution_time_seconds": duration,
            "confidence_score": diagnostic_result.confidence_score,
            "steps_executed": len(execution_result.executed_steps),
            "steps_failed": len(execution_result.failed_steps),
            "success_rate": len(execution_result.executed_steps) / (len(execution_result.executed_steps) + len(execution_result.failed_steps)) if (execution_result.executed_steps or execution_result.failed_steps) else 0
        }
    
    def _analyze_trends(self, metrics: Dict[str, Any]) -> str:
        """分析趋势"""
        return "System performance trending upward with occasional spikes in CPU usage."
    
    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        # 简单的异常检测逻辑
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and value > 0.8:  # 阈值示例
                anomalies.append({
                    "metric": metric_name,
                    "value": value,
                    "threshold": 0.8,
                    "severity": "high"
                })
        
        return anomalies
    
    def _generate_optimization_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于指标生成建议
        if metrics.get("cpu_usage", 0) > 0.7:
            suggestions.append("Consider CPU optimization or scaling")
        if metrics.get("memory_usage", 0) > 0.8:
            suggestions.append("Review memory allocation and cleanup")
        if metrics.get("disk_usage", 0) > 0.9:
            suggestions.append("Implement disk cleanup policies")
        
        return suggestions
    
    def _assess_system_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估系统健康状态"""
        health_score = 1.0
        
        # 简单的健康评分计算
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and value > 0.8:
                health_score -= 0.1
        
        return {
            "overall_score": max(0, health_score),
            "status": "healthy" if health_score > 0.7 else "warning" if health_score > 0.5 else "critical",
            "components": {
                "cpu": "healthy",
                "memory": "healthy", 
                "disk": "healthy",
                "network": "healthy"
            }
        }


class ReportFormatter(dspy.Module):
    """报告格式化器"""
    
    def __init__(self):
        super().__init__()
        self.formatting_signature = dspy.Signature(
            "report_content, format_type -> formatted_report: str"
        )
        self.formatter = dspy.ChainOfThought(self.formatting_signature)
    
    def format_report(self, report: IncidentReport, format_type: str = "markdown") -> str:
        """
        格式化报告
        
        Args:
            report: 事件报告
            format_type: 格式类型
            
        Returns:
            str: 格式化后的报告
        """
        result = self.formatter(
            report_content=str(report),
            format_type=format_type
        )
        
        return result.formatted_report


class ReportArchiver(dspy.Module):
    """报告归档器"""
    
    def __init__(self):
        super().__init__()
        self.archiving_signature = dspy.Signature(
            "report_data, retention_policy -> archive_location: str, metadata: str"
        )
        self.archiver = dspy.ChainOfThought(self.archiving_signature)
    
    def archive_report(self, report: IncidentReport, retention_policy: str) -> Dict[str, str]:
        """
        归档报告
        
        Args:
            report: 事件报告
            retention_policy: 保留策略
            
        Returns:
            Dict: 归档信息
        """
        result = self.archiver(
            report_data=str(report),
            retention_policy=retention_policy
        )
        
        return {
            "archive_location": result.archive_location,
            "metadata": result.metadata
        }