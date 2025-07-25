"""
智能信息收集 DSPy 模块

升级版的信息收集器，支持智能分析、上下文推理和结构化提取
"""

import dspy
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ContextAnalysisSignature(dspy.Signature):
    """上下文分析签名"""
    current_state: str = dspy.InputField(desc="当前状态摘要，包括已有信息和处理阶段")
    missing_info: str = dspy.InputField(desc="缺失的信息类型或字段")
    business_stage: str = dspy.InputField(desc="业务阶段：initialization, alert_processing, diagnosis, action_planning, execution")
    
    info_priority: str = dspy.OutputField(desc="信息优先级：critical, important, helpful, optional")
    collection_strategy: str = dspy.OutputField(desc="收集策略：direct_query, guided_interview, structured_form, contextual_prompt")
    urgency_level: str = dspy.OutputField(desc="紧急程度：immediate, high, medium, low")
    reasoning: str = dspy.OutputField(desc="上下文分析推理过程")


class InfoExtractionSignature(dspy.Signature):
    """信息提取签名"""
    raw_input: str = dspy.InputField(desc="用户提供的原始输入信息")
    expected_info_type: str = dspy.InputField(desc="期望的信息类型")
    extraction_context: str = dspy.InputField(desc="提取上下文和格式要求")
    
    extracted_data: str = dspy.OutputField(desc="提取的结构化数据，JSON格式")
    confidence_level: float = dspy.OutputField(desc="提取置信度：0-1之间的浮点数")
    validation_status: str = dspy.OutputField(desc="验证状态：valid, partially_valid, invalid, needs_clarification")
    reasoning: str = dspy.OutputField(desc="信息提取推理过程")


class QueryGenerationSignature(dspy.Signature):
    """查询生成签名"""
    context_analysis: str = dspy.InputField(desc="上下文分析结果")
    missing_info_type: str = dspy.InputField(desc="缺失信息类型")
    user_friendly: bool = dspy.InputField(desc="是否需要用户友好的表达")
    
    optimized_query: str = dspy.OutputField(desc="优化后的查询问题")
    expected_format: str = dspy.OutputField(desc="期望的回答格式说明")
    extraction_hints: str = dspy.OutputField(desc="信息提取提示")
    reasoning: str = dspy.OutputField(desc="查询生成推理过程")


class InfoCollectionRequest(BaseModel):
    """信息收集请求"""
    user_query: str = Field(description="用户友好的查询问题")
    expected_format: str = Field(description="期望格式")
    extraction_hints: str = Field(description="提取提示")
    priority: str = Field(description="优先级")
    urgency: str = Field(description="紧急程度")
    strategy: str = Field(description="收集策略")


class InfoExtractionResult(BaseModel):
    """信息提取结果"""
    extracted_data: Dict[str, Any] = Field(description="提取的数据")
    confidence: float = Field(description="置信度")
    validation_status: str = Field(description="验证状态")
    original_input: str = Field(description="原始输入")
    reasoning: str = Field(description="提取推理")


class IntelligentInfoCollector(dspy.Module):
    """智能信息收集器
    
    基于 DSPy 的智能化信息收集系统，支持：
    - 上下文感知的信息分析
    - 智能化查询生成
    - 结构化信息提取
    - 多轮交互支持
    """
    
    def __init__(self):
        super().__init__()
        self.context_analyzer = dspy.ChainOfThought(ContextAnalysisSignature)
        self.info_extractor = dspy.ChainOfThought(InfoExtractionSignature)
        self.query_generator = dspy.ChainOfThought(QueryGenerationSignature)
        
        # 信息类型配置
        self.info_type_config = {
            "alert_info": {
                "fields": ["alert_id", "timestamp", "severity", "source", "message"],
                "format": "告警ID、时间戳、严重程度、来源系统、详细消息",
                "example": "ID: ALT001, 时间: 2025-01-25 10:00, 级别: high, 来源: web_server, 消息: HTTP响应超时"
            },
            "symptoms": {
                "fields": ["symptom_description", "occurrence_time", "frequency"],
                "format": "症状描述列表，每个症状包括具体表现和发生频率",
                "example": "1. CPU使用率持续高于90% (持续30分钟), 2. 响应时间超过5秒 (间歇性)"
            },
            "system_context": {
                "fields": ["environment", "architecture", "recent_changes"],
                "format": "系统环境、架构信息、近期变更",
                "example": "环境: 生产环境, 架构: 微服务+K8s, 变更: 昨日发布新版本"
            },
            "error_logs": {
                "fields": ["log_content", "error_patterns", "time_range"],
                "format": "关键错误日志内容、错误模式、时间范围",
                "example": "时间范围: 最近1小时, 错误模式: 数据库连接超时, 日志: connection timeout after 30s"
            }
        }
    
    def analyze_missing_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析缺失信息需求
        
        Args:
            state: 当前状态
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 确定缺失的信息类型
            missing_info_type = self._identify_missing_info(state)
            if not missing_info_type:
                return {"needs_collection": False, "reason": "无缺失信息"}
            
            # 构建上下文分析输入
            current_state_summary = self._build_state_summary(state)
            business_stage = self._determine_business_stage(state)
            
            # 执行上下文分析
            analysis = self.context_analyzer(
                current_state=current_state_summary,
                missing_info=missing_info_type,
                business_stage=business_stage
            )
            
            return {
                "needs_collection": True,
                "missing_type": missing_info_type,
                "priority": analysis.info_priority,
                "strategy": analysis.collection_strategy,
                "urgency": analysis.urgency_level,
                "reasoning": analysis.reasoning,
                "can_use_memory": self._can_supplement_from_memory(missing_info_type),
                "context_hints": self._get_context_hints(missing_info_type, state)
            }
            
        except Exception as e:
            return self._create_fallback_analysis(state, str(e))
    
    def generate_collection_query(self, state: Dict[str, Any], 
                                analysis: Dict[str, Any]) -> InfoCollectionRequest:
        """
        生成信息收集查询
        
        Args:
            state: 当前状态
            analysis: 缺失信息分析结果
            
        Returns:
            InfoCollectionRequest: 信息收集请求
        """
        try:
            missing_type = analysis.get("missing_type", "")
            
            # 生成智能查询
            query_result = self.query_generator(
                context_analysis=analysis.get("reasoning", ""),
                missing_info_type=missing_type,
                user_friendly=True
            )
            
            return InfoCollectionRequest(
                user_query=query_result.optimized_query,
                expected_format=query_result.expected_format,
                extraction_hints=query_result.extraction_hints,
                priority=analysis.get("priority", "medium"),
                urgency=analysis.get("urgency", "medium"),
                strategy=analysis.get("strategy", "direct_query")
            )
            
        except Exception as e:
            return self._create_fallback_query(analysis, str(e))
    
    def extract_structured_info(self, raw_input: str, 
                              missing_info_analysis: Dict[str, Any]) -> InfoExtractionResult:
        """
        从用户输入中提取结构化信息
        
        Args:
            raw_input: 用户原始输入
            missing_info_analysis: 缺失信息分析
            
        Returns:
            InfoExtractionResult: 提取结果
        """
        try:
            missing_type = missing_info_analysis.get("missing_type", "")
            extraction_context = self._build_extraction_context(missing_type)
            
            # 执行信息提取
            extraction = self.info_extractor(
                raw_input=raw_input,
                expected_info_type=missing_type,
                extraction_context=extraction_context
            )
            
            # 解析提取的数据
            extracted_data = self._parse_extracted_data(extraction.extracted_data, missing_type)
            
            return InfoExtractionResult(
                extracted_data=extracted_data,
                confidence=extraction.confidence_level,
                validation_status=extraction.validation_status,
                original_input=raw_input,
                reasoning=extraction.reasoning
            )
            
        except Exception as e:
            return self._create_fallback_extraction(raw_input, missing_info_analysis, str(e))
    
    def _identify_missing_info(self, state: Dict[str, Any]) -> str:
        """识别缺失的信息类型"""
        # 检查关键信息字段
        if not state.get("alert_info"):
            return "alert_info"
        
        # 检查症状信息（诊断阶段需要）
        if state.get("analysis_result") and not state.get("symptoms"):
            return "symptoms"
        
        # 检查系统上下文
        if not state.get("context"):
            return "system_context"
        
        # 检查用户指定的信息收集需求
        if info_queries := state.get("info_collection_queries"):
            return info_queries[0] if info_queries else ""
        
        return ""
    
    def _build_state_summary(self, state: Dict[str, Any]) -> str:
        """构建状态摘要"""
        summary_parts = []
        
        # 已有信息统计
        present_info = []
        if state.get("alert_info"):
            present_info.append("告警信息")
        if state.get("symptoms"):
            present_info.append("症状描述")
        if state.get("context"):
            present_info.append("系统上下文")
        if state.get("analysis_result"):
            present_info.append("分析结果")
        if state.get("diagnostic_result"):
            present_info.append("诊断结果")
        
        if present_info:
            summary_parts.append(f"已有信息：{', '.join(present_info)}")
        
        # 处理阶段
        stage = self._determine_business_stage(state)
        summary_parts.append(f"当前阶段：{stage}")
        
        # 错误状态
        if errors := state.get("errors"):
            summary_parts.append(f"存在错误：{len(errors)}个")
        
        return "; ".join(summary_parts)
    
    def _determine_business_stage(self, state: Dict[str, Any]) -> str:
        """确定业务阶段"""
        if state.get("execution_result"):
            return "execution"
        elif state.get("action_plan"):
            return "action_planning"
        elif state.get("diagnostic_result"):
            return "diagnosis"
        elif state.get("analysis_result"):
            return "alert_processing"
        else:
            return "initialization"
    
    def _can_supplement_from_memory(self, info_type: str) -> bool:
        """判断是否可以从记忆中补充信息"""
        # 某些信息类型可以从历史记忆中获取
        memory_supplementable = [
            "system_context",
            "common_symptoms",
            "standard_procedures"
        ]
        return info_type in memory_supplementable
    
    def _get_context_hints(self, info_type: str, state: Dict[str, Any]) -> str:
        """获取上下文提示"""
        hints = []
        
        # 基于信息类型的提示
        if info_type == "symptoms":
            if alert_info := state.get("alert_info"):
                source = getattr(alert_info, 'source', '')
                hints.append(f"关注{source}系统的异常表现")
        
        elif info_type == "system_context":
            hints.append("包括系统架构、部署环境、近期变更等")
        
        elif info_type == "alert_info":
            hints.append("提供完整的告警详细信息")
        
        return "; ".join(hints) if hints else "请提供详细准确的信息"
    
    def _build_extraction_context(self, info_type: str) -> str:
        """构建提取上下文"""
        config = self.info_type_config.get(info_type, {})
        
        context_parts = []
        if fields := config.get("fields"):
            context_parts.append(f"需要提取的字段：{', '.join(fields)}")
        
        if format_desc := config.get("format"):
            context_parts.append(f"期望格式：{format_desc}")
        
        if example := config.get("example"):
            context_parts.append(f"示例：{example}")
        
        return "; ".join(context_parts)
    
    def _parse_extracted_data(self, extracted_json: str, info_type: str) -> Dict[str, Any]:
        """解析提取的数据"""
        try:
            import json
            data = json.loads(extracted_json)
            return data if isinstance(data, dict) else {"raw_data": data}
        except:
            # 回退解析
            return {"raw_text": extracted_json, "type": info_type}
    
    def _create_fallback_analysis(self, state: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """创建回退分析"""
        return {
            "needs_collection": True,
            "missing_type": "general_info",
            "priority": "medium",
            "strategy": "direct_query",
            "urgency": "medium",
            "reasoning": f"回退分析：{error_msg}",
            "can_use_memory": False,
            "context_hints": "请提供相关信息"
        }
    
    def _create_fallback_query(self, analysis: Dict[str, Any], error_msg: str) -> InfoCollectionRequest:
        """创建回退查询"""
        missing_type = analysis.get("missing_type", "信息")
        
        return InfoCollectionRequest(
            user_query=f"请提供{missing_type}的详细信息",
            expected_format="结构化文本",
            extraction_hints="请尽可能详细和准确",
            priority=analysis.get("priority", "medium"),
            urgency=analysis.get("urgency", "medium"),
            strategy="direct_query"
        )
    
    def _create_fallback_extraction(self, raw_input: str, 
                                  analysis: Dict[str, Any], 
                                  error_msg: str) -> InfoExtractionResult:
        """创建回退提取"""
        return InfoExtractionResult(
            extracted_data={"raw_input": raw_input},
            confidence=0.3,
            validation_status="needs_clarification",
            original_input=raw_input,
            reasoning=f"回退提取：{error_msg}"
        )