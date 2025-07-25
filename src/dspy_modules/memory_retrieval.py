"""
记忆检索 DSPy 组件 (精简版)

单次推理完成记忆需求分析和查询生成
"""

import dspy
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class SimpleMemorySignature(dspy.Signature):
    """精简记忆处理签名 - 一次推理完成所有任务"""
    current_context: str = dspy.InputField(desc="当前业务上下文，包括告警、症状、诊断等信息")
    processing_stage: str = dspy.InputField(desc="当前处理阶段：alert_analysis, diagnosis, action_planning, execution")
    
    need_memory: str = dspy.OutputField(desc="是否需要记忆检索：yes/no")
    queries: str = dspy.OutputField(desc="检索查询，用逗号分隔，最多3个（如果需要检索）")
    reasoning: str = dspy.OutputField(desc="决策推理过程")


class SimpleMemoryResult(BaseModel):
    """精简记忆处理结果"""
    need_memory: bool = Field(description="是否需要记忆检索")
    queries: List[str] = Field(default_factory=list, description="检索查询列表")
    reasoning: str = Field(description="决策推理")


class SimpleMemoryProcessor(dspy.Module):
    """精简记忆处理器
    
    一次推理完成记忆需求分析和查询生成
    """
    
    def __init__(self):
        super().__init__()
        self.memory_processor = dspy.ChainOfThought(SimpleMemorySignature)
    
    def forward(self, context: Dict[str, Any], stage: str = "diagnosis") -> SimpleMemoryResult:
        """
        处理记忆需求并生成查询
        
        Args:
            context: 当前上下文
            stage: 处理阶段
            
        Returns:
            SimpleMemoryResult: 记忆处理结果
        """
        try:
            # 构建上下文摘要
            current_context = self._build_context_summary(context)
            
            # 执行记忆处理
            result = self.memory_processor(
                current_context=current_context,
                processing_stage=stage
            )
            
            # 解析结果
            need_memory = result.need_memory.lower() == "yes"
            queries = self._parse_queries(result.queries) if need_memory else []
            
            return SimpleMemoryResult(
                need_memory=need_memory,
                queries=queries,
                reasoning=result.reasoning
            )
            
        except Exception as e:
            # 简单回退策略
            return self._create_fallback_result(context, stage, str(e))
    
    def _build_context_summary(self, context: Dict[str, Any]) -> str:
        """构建上下文摘要"""
        summary_parts = []
        
        # 告警信息
        if alert_info := context.get("alert_info"):
            severity = getattr(alert_info, 'severity', 'unknown')
            source = getattr(alert_info, 'source', 'unknown')
            message = getattr(alert_info, 'message', '')[:80]
            summary_parts.append(f"告警：{severity}级别来自{source}，{message}")
        
        # 症状
        if symptoms := context.get("symptoms"):
            summary_parts.append(f"症状：{', '.join(symptoms[:2])}")
        
        # 诊断结果
        if diagnostic := context.get("diagnostic_result"):
            root_cause = diagnostic.get("root_cause", "")[:50]
            summary_parts.append(f"诊断：{root_cause}")
        
        return "; ".join(summary_parts) if summary_parts else "无具体上下文"
    
    def _parse_queries(self, query_string: str) -> List[str]:
        """解析查询字符串"""
        if not query_string:
            return []
        
        queries = [q.strip() for q in query_string.split(",") if q.strip()]
        return queries[:3]  # 限制最多3个查询
    
    def _create_fallback_result(self, context: Dict[str, Any], stage: str, error_msg: str) -> SimpleMemoryResult:
        """创建回退结果"""
        # 简单启发式：诊断和行动规划阶段更需要记忆
        need_memory = stage in ["diagnosis", "action_planning"]
        
        # 基础查询
        fallback_queries = []
        if need_memory:
            if stage == "diagnosis":
                fallback_queries = ["故障诊断", "历史案例"]
            elif stage == "action_planning":
                fallback_queries = ["解决方案", "最佳实践"]
        
        return SimpleMemoryResult(
            need_memory=need_memory,
            queries=fallback_queries,
            reasoning=f"回退策略：{error_msg}"
        )

