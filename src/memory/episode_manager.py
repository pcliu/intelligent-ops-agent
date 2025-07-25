"""
运维情节管理器

负责运维情节的分类、生成和管理
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class OpsEpisodeType(Enum):
    """运维情节类型枚举"""
    ALERT = "alert"                    # 告警情节
    DIAGNOSIS = "diagnosis"            # 诊断情节  
    ACTION = "action"                  # 执行情节
    INCIDENT = "incident"              # 完整事件情节
    KNOWLEDGE = "knowledge"            # 运维知识情节


class EpisodeMetadata(BaseModel):
    """情节元数据"""
    episode_type: str = Field(description="情节类型")
    confidence_score: Optional[float] = Field(default=None, description="置信度评分")
    alert_id: Optional[str] = Field(default=None, description="告警ID")
    incident_id: Optional[str] = Field(default=None, description="事件ID")
    resolution_status: Optional[str] = Field(default=None, description="解决状态")
    severity: Optional[str] = Field(default=None, description="严重程度")
    source_system: Optional[str] = Field(default=None, description="来源系统")
    tags: List[str] = Field(default_factory=list, description="标签")


class OpsEpisode(BaseModel):
    """运维情节数据结构"""
    name: str = Field(description="情节名称")
    episode_body: str = Field(description="情节内容")
    source_description: str = Field(description="来源描述")
    reference_time: datetime = Field(description="参考时间")
    episode_type: OpsEpisodeType = Field(description="情节类型")
    metadata: EpisodeMetadata = Field(description="元数据")
    
    class Config:
        arbitrary_types_allowed = True


class EpisodeManager:
    """运维情节管理器"""
    
    def __init__(self):
        self.episode_templates = {
            OpsEpisodeType.ALERT: self._create_alert_episode,
            OpsEpisodeType.DIAGNOSIS: self._create_diagnosis_episode,
            OpsEpisodeType.ACTION: self._create_action_episode,
            OpsEpisodeType.INCIDENT: self._create_incident_episode,
            OpsEpisodeType.KNOWLEDGE: self._create_knowledge_episode
        }
    
    def generate_episodes_from_state(self, state: Dict[str, Any]) -> List[OpsEpisode]:
        """从状态字段自动生成需要存储的情节"""
        episodes = []
        
        # 1. 告警情节
        if alert_info := state.get("alert_info"):
            episodes.append(self._create_alert_episode(alert_info, state))
        
        # 2. 诊断情节
        if diagnostic_result := state.get("diagnostic_result"):
            if state.get("analysis_result"):  # 确保有分析结果
                episodes.append(self._create_diagnosis_episode(diagnostic_result, state))
        
        # 3. 执行情节
        if execution_result := state.get("execution_result"):
            if state.get("action_plan"):  # 确保有行动计划
                episodes.append(self._create_action_episode(execution_result, state))
        
        # 4. 完整事件情节（当所有主要步骤都完成时）
        if self._is_complete_incident(state):
            episodes.append(self._create_incident_episode(state))
        
        return episodes
    
    def _create_alert_episode(self, alert_info: Any, state: Dict[str, Any]) -> OpsEpisode:
        """创建告警情节"""
        alert_id = getattr(alert_info, 'alert_id', 'unknown')
        source = getattr(alert_info, 'source', 'unknown')
        message = getattr(alert_info, 'message', '')
        severity = getattr(alert_info, 'severity', 'unknown')
        
        metadata = EpisodeMetadata(
            episode_type=OpsEpisodeType.ALERT.value,
            alert_id=alert_id,
            severity=severity,
            source_system=source,
            tags=[f"alert_{severity}", f"source_{source}"]
        )
        
        return OpsEpisode(
            name=f"Alert_{alert_id}",
            episode_body=f"告警ID: {alert_id}, 来源: {source}, 消息: {message}, 严重程度: {severity}",
            source_description="智能运维告警处理",
            reference_time=datetime.now(),
            episode_type=OpsEpisodeType.ALERT,
            metadata=metadata
        )
    
    def _create_diagnosis_episode(self, diagnostic_result: Dict[str, Any], state: Dict[str, Any]) -> OpsEpisode:
        """创建诊断情节"""
        alert_info = state.get("alert_info")
        symptoms = state.get("symptoms", [])
        
        alert_id = getattr(alert_info, 'alert_id', 'unknown') if alert_info else 'unknown'
        root_cause = diagnostic_result.get("root_cause", "未确定")
        confidence = diagnostic_result.get("confidence_score", 0.0)
        potential_causes = diagnostic_result.get("potential_causes", [])
        
        metadata = EpisodeMetadata(
            episode_type=OpsEpisodeType.DIAGNOSIS.value,
            confidence_score=confidence,
            alert_id=alert_id,
            tags=[f"diagnosis_{root_cause}", f"confidence_{int(confidence*100)}"]
        )
        
        return OpsEpisode(
            name=f"Diagnosis_{alert_id}",
            episode_body=f"症状: {', '.join(symptoms)}, 根因: {root_cause}, 置信度: {confidence}, 可能原因: {', '.join(potential_causes)}",
            source_description="智能运维故障诊断",
            reference_time=datetime.now(),
            episode_type=OpsEpisodeType.DIAGNOSIS,
            metadata=metadata
        )
    
    def _create_action_episode(self, execution_result: Dict[str, Any], state: Dict[str, Any]) -> OpsEpisode:
        """创建执行情节"""
        alert_info = state.get("alert_info")
        action_plan = state.get("action_plan", {})
        
        alert_id = getattr(alert_info, 'alert_id', 'unknown') if alert_info else 'unknown'
        actions = action_plan.get("actions", [])
        success = execution_result.get("success", False)
        results = execution_result.get("results", [])
        
        metadata = EpisodeMetadata(
            episode_type=OpsEpisodeType.ACTION.value,
            alert_id=alert_id,
            resolution_status="success" if success else "failed",
            tags=[f"action_{len(actions)}", f"result_{'success' if success else 'failed'}"]
        )
        
        return OpsEpisode(
            name=f"Action_{alert_id}",
            episode_body=f"执行动作: {', '.join(actions)}, 成功: {success}, 结果: {', '.join(str(r) for r in results)}",
            source_description="智能运维自动化执行",
            reference_time=datetime.now(),
            episode_type=OpsEpisodeType.ACTION,
            metadata=metadata
        )
    
    def _create_incident_episode(self, state: Dict[str, Any]) -> OpsEpisode:
        """创建完整事件情节"""
        alert_info = state.get("alert_info")
        alert_id = getattr(alert_info, 'alert_id', 'unknown') if alert_info else 'unknown'
        
        # 构建完整的事件描述
        stages = []
        if state.get("analysis_result"):
            stages.append("告警分析")
        if state.get("diagnostic_result"):
            stages.append("故障诊断")
        if state.get("action_plan"):
            stages.append("行动规划")
        if state.get("execution_result"):
            stages.append("执行处理")
        if state.get("report"):
            stages.append("报告生成")
        
        final_status = "已解决" if state.get("execution_result", {}).get("success") else "未解决"
        
        metadata = EpisodeMetadata(
            episode_type=OpsEpisodeType.INCIDENT.value,
            alert_id=alert_id,
            resolution_status=final_status.lower(),
            tags=[f"incident_complete", f"stages_{len(stages)}"]
        )
        
        return OpsEpisode(
            name=f"Incident_{alert_id}",
            episode_body=f"完整事件处理: 告警 → {' → '.join(stages)}, 最终状态: {final_status}",
            source_description="智能运维完整事件处理",
            reference_time=datetime.now(),
            episode_type=OpsEpisodeType.INCIDENT,
            metadata=metadata
        )
    
    def _create_knowledge_episode(self, knowledge_data: Dict[str, Any], state: Dict[str, Any]) -> OpsEpisode:
        """创建知识情节（用于存储运维知识和经验）"""
        knowledge_type = knowledge_data.get("type", "general")
        content = knowledge_data.get("content", "")
        
        metadata = EpisodeMetadata(
            episode_type=OpsEpisodeType.KNOWLEDGE.value,
            tags=[f"knowledge_{knowledge_type}"]
        )
        
        return OpsEpisode(
            name=f"Knowledge_{knowledge_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            episode_body=content,
            source_description="智能运维知识积累",
            reference_time=datetime.now(),
            episode_type=OpsEpisodeType.KNOWLEDGE,
            metadata=metadata
        )
    
    def _is_complete_incident(self, state: Dict[str, Any]) -> bool:
        """判断是否为完整的事件（需要存储完整事件情节）"""
        required_stages = ["alert_info", "analysis_result", "diagnostic_result"]
        return all(state.get(stage) is not None for stage in required_stages)
    
    def create_episode_from_template(self, episode_type: OpsEpisodeType, data: Dict[str, Any], state: Dict[str, Any]) -> OpsEpisode:
        """根据模板创建情节"""
        if episode_type not in self.episode_templates:
            raise ValueError(f"不支持的情节类型: {episode_type}")
        
        return self.episode_templates[episode_type](data, state)
    
    def get_episode_summary(self, episode: OpsEpisode) -> str:
        """获取情节摘要"""
        return f"[{episode.episode_type.value}] {episode.name}: {episode.episode_body[:100]}..."