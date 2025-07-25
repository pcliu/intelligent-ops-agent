"""
智能路由 DSPy 模块

基于 DSPy 的智能化中央路由器，支持记忆驱动的动态流程决策
"""

import dspy
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class StateAnalysisSignature(dspy.Signature):
    """状态分析签名"""
    business_data: str = dspy.InputField(desc="业务数据摘要：告警、症状、诊断等信息")
    completion_status: str = dspy.InputField(desc="完成状态：各个处理阶段的完成情况")
    error_status: str = dspy.InputField(desc="错误状态：当前是否存在错误或异常")
    
    business_priority: str = dspy.OutputField(desc="业务优先级：high, medium, low")
    current_stage: str = dspy.OutputField(desc="当前阶段：alert_analysis, diagnosis, action_planning, execution, reporting, completed")
    data_completeness: float = dspy.OutputField(desc="数据完整性评分：0-1之间的浮点数")
    reasoning: str = dspy.OutputField(desc="状态分析推理过程")


class MemoryNeedSignature(dspy.Signature):
    """记忆需求分析签名"""
    current_state: str = dspy.InputField(desc="当前完整状态信息")
    state_analysis: str = dspy.InputField(desc="状态分析结果")
    memory_context: str = dspy.InputField(desc="现有记忆上下文，如果为空表示没有记忆")
    
    memory_priority: str = dspy.OutputField(desc="记忆优先级：high, medium, low, none")
    should_retrieve: bool = dspy.OutputField(desc="是否需要检索记忆：true或false")
    suggested_queries: str = dspy.OutputField(desc="建议的记忆查询，用逗号分隔")
    reasoning: str = dspy.OutputField(desc="记忆需求分析推理")


class RouteOptimizationSignature(dspy.Signature):
    """路由优化决策签名"""
    state_analysis: str = dspy.InputField(desc="状态分析结果")
    memory_analysis: str = dspy.InputField(desc="记忆需求分析结果")
    available_nodes: str = dspy.InputField(desc="可用的节点列表")
    performance_context: str = dspy.InputField(desc="性能上下文：历史执行时间、成功率等")
    
    next_node: str = dspy.OutputField(desc="下一个执行节点：retrieve_memory, collect_info, process_alert, diagnose_issue, plan_actions, execute_actions, generate_report, update_memory, END")
    confidence: float = dspy.OutputField(desc="决策置信度：0-1之间的浮点数")
    alternative_paths: str = dspy.OutputField(desc="替代路径，用逗号分隔")
    reasoning: str = dspy.OutputField(desc="路由决策推理过程")


class RouterDecision(BaseModel):
    """路由决策结果"""
    next_node: str = Field(description="下一个执行节点")
    reasoning: str = Field(description="决策推理")
    confidence: float = Field(description="决策置信度")
    alternative_paths: List[str] = Field(default_factory=list, description="替代路径")
    memory_priority: str = Field(default="none", description="记忆优先级")
    business_priority: str = Field(default="medium", description="业务优先级")
    suggested_queries: List[str] = Field(default_factory=list, description="建议查询")
    requires_info_collection: bool = Field(default=False, description="是否需要信息收集")
    missing_info_queries: List[str] = Field(default_factory=list, description="缺失信息查询")


class SmartIntelligentRouter(dspy.Module):
    """智能化的中央路由器
    
    基于 DSPy 推理的智能路由系统，支持：
    - 状态分析和业务阶段判断
    - 记忆需求分析和优先级决策
    - 动态路由优化和替代路径选择
    - 多轮交互支持
    """
    
    def __init__(self):
        super().__init__()
        self.state_analyzer = dspy.ChainOfThought(StateAnalysisSignature)
        self.memory_need_analyzer = dspy.ChainOfThought(MemoryNeedSignature)
        self.route_optimizer = dspy.ChainOfThought(RouteOptimizationSignature)
        
        # 路由规则配置
        self.node_priority_map = {
            "retrieve_memory": 10,    # 记忆检索最高优先级
            "collect_info": 9,        # 信息收集次高优先级
            "process_alert": 8,       # 业务流程按顺序
            "diagnose_issue": 7,
            "plan_actions": 6,
            "execute_actions": 5,
            "generate_report": 4,
            "update_memory": 3,       # 记忆更新较低优先级
            "END": 1                  # 结束最低优先级
        }
    
    def forward(self, current_state: Dict[str, Any]) -> RouterDecision:
        """
        执行智能路由决策
        
        Args:
            current_state: 当前完整状态
            
        Returns:
            RouterDecision: 路由决策结果
        """
        try:
            # 1. 状态分析
            state_analysis = self.state_analyzer(
                business_data=self._extract_business_data(current_state),
                completion_status=self._check_completion_status(current_state),
                error_status=self._check_error_status(current_state)
            )
            
            # 2. 记忆需求分析
            memory_analysis = self.memory_need_analyzer(
                current_state=self._serialize_complete_state(current_state),
                state_analysis=state_analysis.reasoning,
                memory_context=str(current_state.get("memory_context", {}))
            )
            
            # 3. 路由优化决策
            route_decision = self.route_optimizer(
                state_analysis=state_analysis.reasoning,
                memory_analysis=memory_analysis.reasoning,
                available_nodes=self._get_available_nodes(),
                performance_context=self._get_performance_context(current_state)
            )
            
            # 4. 构建决策结果
            return self._build_router_decision(state_analysis, memory_analysis, route_decision, current_state)
            
        except Exception as e:
            # 错误回退处理
            return self._create_fallback_decision(current_state, str(e))
    
    def _extract_business_data(self, state: Dict[str, Any]) -> str:
        """提取业务数据摘要"""
        business_summary = []
        
        # 告警信息
        if alert_info := state.get("alert_info"):
            severity = getattr(alert_info, 'severity', 'unknown')
            source = getattr(alert_info, 'source', 'unknown')
            message = getattr(alert_info, 'message', '')[:100]
            business_summary.append(f"告警：{severity}级别，来源{source}，{message}")
        
        # 症状信息
        if symptoms := state.get("symptoms"):
            business_summary.append(f"症状：{', '.join(symptoms[:3])}")
        
        # 分析结果
        if analysis_result := state.get("analysis_result"):
            category = analysis_result.get("category", "")
            priority = analysis_result.get("priority", "")
            if category or priority:
                business_summary.append(f"分析结果：{category} {priority}")
        
        # 诊断结果
        if diagnostic_result := state.get("diagnostic_result"):
            root_cause = diagnostic_result.get("root_cause", "")
            confidence = diagnostic_result.get("confidence_score", 0)
            if root_cause:
                business_summary.append(f"诊断：{root_cause} (置信度{confidence:.2f})")
        
        # 行动计划
        if action_plan := state.get("action_plan"):
            actions = action_plan.get("actions", [])
            if actions:
                business_summary.append(f"计划：{len(actions)}个行动")
        
        # 执行结果
        if execution_result := state.get("execution_result"):
            success = execution_result.get("success", False)
            business_summary.append(f"执行：{'成功' if success else '失败'}")
        
        return "; ".join(business_summary) if business_summary else "无业务数据"
    
    def _check_completion_status(self, state: Dict[str, Any]) -> str:
        """检查完成状态"""
        stages = {
            "alert_info": "告警接收",
            "analysis_result": "告警分析", 
            "diagnostic_result": "故障诊断",
            "action_plan": "行动规划",
            "execution_result": "执行处理",
            "report": "报告生成"
        }
        
        completed_stages = []
        for field, stage_name in stages.items():
            if state.get(field):
                completed_stages.append(stage_name)
        
        return f"已完成阶段：{', '.join(completed_stages)}" if completed_stages else "未开始"
    
    def _check_error_status(self, state: Dict[str, Any]) -> str:
        """检查错误状态"""
        errors = state.get("errors", [])
        if errors:
            return f"存在{len(errors)}个错误：{'; '.join(errors[:2])}"
        return "无错误"
    
    def _serialize_complete_state(self, state: Dict[str, Any]) -> str:
        """序列化完整状态"""
        serialized_parts = []
        
        # 核心业务数据
        for key in ["alert_info", "symptoms", "context", "analysis_result", 
                   "diagnostic_result", "action_plan", "execution_result", "report"]:
            if value := state.get(key):
                serialized_parts.append(f"{key}: {str(value)[:200]}")
        
        # 记忆相关数据
        if memory_queries := state.get("memory_queries"):
            serialized_parts.append(f"memory_queries: {memory_queries}")
        
        if memory_context := state.get("memory_context"):
            serialized_parts.append(f"memory_context: 存在记忆上下文")
        
        # 信息收集相关数据
        if info_queries := state.get("info_collection_queries"):
            serialized_parts.append(f"info_collection_queries: {info_queries}")
        
        if info_context := state.get("info_collection_context"):
            serialized_parts.append(f"info_collection_context: 存在收集信息")
        
        return "; ".join(serialized_parts) if serialized_parts else "空状态"
    
    def _get_available_nodes(self) -> str:
        """获取可用节点列表"""
        nodes = [
            "retrieve_memory (记忆检索)",
            "collect_info (信息收集)", 
            "process_alert (告警处理)",
            "diagnose_issue (故障诊断)",
            "plan_actions (行动规划)",
            "execute_actions (执行处理)",
            "generate_report (报告生成)",
            "update_memory (记忆更新)",
            "END (结束)"
        ]
        return ", ".join(nodes)
    
    def _get_performance_context(self, state: Dict[str, Any]) -> str:
        """获取性能上下文"""
        # 这里可以集成历史性能数据
        # 暂时返回简单的上下文信息
        message_count = len(state.get("messages", []))
        processing_time = datetime.now().isoformat()
        
        return f"消息数量：{message_count}，当前时间：{processing_time}"
    
    def _build_router_decision(self, state_analysis, memory_analysis, route_decision, current_state: Dict[str, Any]) -> RouterDecision:
        """构建路由决策结果"""
        
        # 解析建议查询
        suggested_queries = []
        if memory_analysis.suggested_queries:
            suggested_queries = [q.strip() for q in memory_analysis.suggested_queries.split(",") if q.strip()]
        
        # 解析替代路径
        alternative_paths = []
        if route_decision.alternative_paths:
            alternative_paths = [p.strip() for p in route_decision.alternative_paths.split(",") if p.strip()]
        
        # 检查是否需要信息收集
        requires_info_collection = self._check_info_collection_needs(current_state)
        missing_info_queries = []
        if requires_info_collection:
            missing_info_queries = self._generate_info_collection_queries(current_state)
        
        return RouterDecision(
            next_node=route_decision.next_node,
            reasoning=route_decision.reasoning,
            confidence=route_decision.confidence,
            alternative_paths=alternative_paths,
            memory_priority=memory_analysis.memory_priority,
            business_priority=state_analysis.business_priority,
            suggested_queries=suggested_queries,
            requires_info_collection=requires_info_collection,
            missing_info_queries=missing_info_queries
        )
    
    def _check_info_collection_needs(self, state: Dict[str, Any]) -> bool:
        """检查是否需要信息收集"""
        # 检查关键信息是否缺失
        if not state.get("alert_info"):
            return True
        
        # 检查诊断阶段是否缺少症状信息
        if state.get("analysis_result") and not state.get("symptoms"):
            return True
        
        # 检查现有的信息收集需求
        if state.get("info_collection_queries"):
            return True
        
        return False
    
    def _generate_info_collection_queries(self, state: Dict[str, Any]) -> List[str]:
        """生成信息收集查询"""
        queries = []
        
        if not state.get("alert_info"):
            queries.append("告警详细信息")
        
        if state.get("analysis_result") and not state.get("symptoms"):
            queries.append("故障症状描述")
        
        if not state.get("context"):
            queries.append("系统运行环境信息")
        
        return queries
    
    def _create_fallback_decision(self, state: Dict[str, Any], error_msg: str) -> RouterDecision:
        """创建回退决策"""
        # 简单的回退逻辑
        if state.get("memory_queries"):
            next_node = "retrieve_memory"
        elif state.get("info_collection_queries"):
            next_node = "collect_info"
        elif not state.get("alert_info"):
            next_node = "collect_info" 
        elif not state.get("analysis_result"):
            next_node = "process_alert"
        elif not state.get("diagnostic_result"):
            next_node = "diagnose_issue"
        elif not state.get("action_plan"):
            next_node = "plan_actions"
        elif not state.get("execution_result"):
            next_node = "execute_actions"
        elif not state.get("report"):
            next_node = "generate_report"
        else:
            next_node = "END"
        
        return RouterDecision(
            next_node=next_node,
            reasoning=f"回退决策：{error_msg}",
            confidence=0.5,
            alternative_paths=["collect_info", "END"],
            memory_priority="none",
            business_priority="medium"
        )