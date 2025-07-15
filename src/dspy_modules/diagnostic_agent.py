import dspy
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from .alert_analyzer import AlertAnalysisResult


class DiagnosticContext(BaseModel):
    """诊断上下文模型"""
    alert_analysis: AlertAnalysisResult
    system_metrics: Dict[str, Any] = Field(default_factory=dict)
    log_entries: List[str] = Field(default_factory=list)
    historical_incidents: List[Dict[str, Any]] = Field(default_factory=list)
    topology_info: Dict[str, Any] = Field(default_factory=dict)
    additional_context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DiagnosticResult(BaseModel):
    """诊断结果模型"""
    incident_id: str
    root_cause: str
    confidence_score: float  # 0-1
    impact_assessment: str
    affected_components: List[str] = Field(default_factory=list)
    business_impact: str
    recovery_time_estimate: str
    similar_incidents: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class RootCauseAnalysis(dspy.Signature):
    """根因分析签名"""
    alert_info: str = dspy.InputField(desc="告警分析结果")
    system_metrics: str = dspy.InputField(desc="系统指标数据")
    log_entries: str = dspy.InputField(desc="相关日志条目")
    
    root_cause: str = dspy.OutputField(desc="根本原因分析")
    confidence_score: float = dspy.OutputField(desc="置信度评分 0-1")
    evidence: str = dspy.OutputField(desc="支持证据列表")


class ImpactAssessment(dspy.Signature):
    """影响评估签名"""
    root_cause: str = dspy.InputField(desc="根本原因")
    topology_info: str = dspy.InputField(desc="系统拓扑信息")
    system_metrics: str = dspy.InputField(desc="系统指标")
    
    impact_level: str = dspy.OutputField(desc="影响级别: critical, high, medium, low")
    affected_components: str = dspy.OutputField(desc="受影响组件列表")
    business_impact: str = dspy.OutputField(desc="业务影响描述")
    recovery_time_estimate: str = dspy.OutputField(desc="恢复时间估计")


class SimilarIncidentRetrieval(dspy.Signature):
    """相似事件检索签名"""
    current_incident: str = dspy.InputField(desc="当前事件描述")
    historical_incidents: str = dspy.InputField(desc="历史事件数据")
    
    similar_incidents: str = dspy.OutputField(desc="相似事件ID列表")
    similarity_reasons: str = dspy.OutputField(desc="相似性原因分析")


class DiagnosticAgent(dspy.Module):
    """诊断智能体模块
    
    功能：
    - 根因分析
    - 影响范围评估
    - 历史案例检索
    - 诊断报告生成
    """
    
    def __init__(self):
        super().__init__()
        self.root_cause_analyzer = dspy.ChainOfThought(RootCauseAnalysis)
        self.impact_assessor = dspy.ChainOfThought(ImpactAssessment)
        self.incident_retriever = dspy.ChainOfThought(SimilarIncidentRetrieval)
        
    def forward(self, diagnostic_context: DiagnosticContext) -> DiagnosticResult:
        """
        执行诊断分析
        
        Args:
            diagnostic_context: 诊断上下文
            
        Returns:
            DiagnosticResult: 诊断结果
        """
        # 1. 根因分析
        root_cause_result = self.root_cause_analyzer(
            alert_info=self._format_alert_analysis(diagnostic_context.alert_analysis),
            system_metrics=self._format_system_metrics(diagnostic_context.system_metrics),
            log_entries=self._format_log_entries_with_context(diagnostic_context)
        )
        
        # 2. 影响评估
        impact_result = self.impact_assessor(
            root_cause=root_cause_result.root_cause,
            topology_info=self._format_topology_info(diagnostic_context.topology_info),
            system_metrics=self._format_system_metrics(diagnostic_context.system_metrics)
        )
        
        # 3. 相似事件检索
        similar_incidents = []
        if diagnostic_context.historical_incidents:
            similar_result = self.incident_retriever(
                current_incident=self._format_current_incident(diagnostic_context, root_cause_result.root_cause),
                historical_incidents=self._format_historical_incidents(diagnostic_context.historical_incidents)
            )
            similar_incidents = similar_result.similar_incidents.split(',') if similar_result.similar_incidents else []
        
        # 4. 生成诊断结果
        return DiagnosticResult(
            incident_id=diagnostic_context.alert_analysis.alert_id,
            root_cause=root_cause_result.root_cause,
            confidence_score=root_cause_result.confidence_score,
            impact_assessment=impact_result.impact_level,
            affected_components=impact_result.affected_components.split(',') if impact_result.affected_components else [],
            business_impact=impact_result.business_impact,
            recovery_time_estimate=impact_result.recovery_time_estimate,
            similar_incidents=similar_incidents,
            evidence=root_cause_result.evidence.split(',') if root_cause_result.evidence else []
        )
    
    def _format_alert_analysis(self, analysis: AlertAnalysisResult) -> str:
        """格式化告警分析结果"""
        return f"""
        Alert ID: {analysis.alert_id}
        Priority: {analysis.priority}
        Category: {analysis.category}
        Urgency Score: {analysis.urgency_score}
        Root Cause Hints: {analysis.root_cause_hints}
        Recommended Actions: {analysis.recommended_actions}
        """
    
    def _format_system_metrics(self, metrics: Dict[str, Any]) -> str:
        """格式化系统指标"""
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"{key}: {value}")
        return "; ".join(formatted)
    
    def _format_log_entries(self, log_entries: List[str]) -> str:
        """格式化日志条目"""
        return "; ".join(log_entries[-20:])  # 只取最近20条
    
    def _format_log_entries_with_context(self, diagnostic_context: DiagnosticContext) -> str:
        """格式化日志条目，包含额外上下文信息"""
        log_entries = diagnostic_context.log_entries[-20:]  # 只取最近20条
        
        # 添加额外上下文信息
        if diagnostic_context.additional_context:
            human_input = diagnostic_context.additional_context.get("human_input")
            if human_input:
                log_entries.append(f"运维人员提供的额外信息: {human_input}")
        
        return "; ".join(log_entries)
    
    def _format_topology_info(self, topology: Dict[str, Any]) -> str:
        """格式化拓扑信息"""
        return str(topology)
    
    def _format_current_incident(self, context: DiagnosticContext, root_cause: str) -> str:
        """格式化当前事件"""
        return f"""
        Category: {context.alert_analysis.category}
        Priority: {context.alert_analysis.priority}
        Root Cause: {root_cause}
        Symptoms: {context.alert_analysis.root_cause_hints}
        """
    
    def _format_historical_incidents(self, incidents: List[Dict[str, Any]]) -> str:
        """格式化历史事件"""
        formatted = []
        for incident in incidents[-10:]:  # 只取最近10个
            formatted.append(f"ID: {incident.get('id', 'N/A')}, "
                           f"Category: {incident.get('category', 'N/A')}, "
                           f"Root Cause: {incident.get('root_cause', 'N/A')}")
        return "; ".join(formatted)


class KnowledgeBaseRetriever(dspy.Module):
    """知识库检索模块"""
    
    def __init__(self):
        super().__init__()
        self.retrieve_signature = dspy.Signature(
            "query, context -> relevant_knowledge: str, confidence: float"
        )
        self.retriever = dspy.ChainOfThought(self.retrieve_signature)
    
    def forward(self, query: str, context: str = "") -> str:
        """
        从知识库检索相关信息
        
        Args:
            query: 查询内容
            context: 上下文信息
            
        Returns:
            str: 相关知识
        """
        result = self.retriever(query=query, context=context)
        return result.relevant_knowledge


class ExpertSystemReasoner(dspy.Module):
    """专家系统推理模块"""
    
    def __init__(self):
        super().__init__()
        self.reasoning_signature = dspy.Signature(
            "symptoms, rules, facts -> diagnosis: str, reasoning_chain: str"
        )
        self.reasoner = dspy.ChainOfThought(self.reasoning_signature)
    
    def forward(self, symptoms: List[str], rules: List[str], facts: List[str]) -> Dict[str, str]:
        """
        基于专家系统规则进行推理
        
        Args:
            symptoms: 症状列表
            rules: 规则列表
            facts: 事实列表
            
        Returns:
            Dict: 推理结果
        """
        result = self.reasoner(
            symptoms="; ".join(symptoms),
            rules="; ".join(rules),
            facts="; ".join(facts)
        )
        
        return {
            "diagnosis": result.diagnosis,
            "reasoning_chain": result.reasoning_chain
        }