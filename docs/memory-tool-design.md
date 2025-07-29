# Memory Tool Integration Design

## 概述

本设计文档描述了将 Graphiti 长期记忆功能作为工具集成到 DSPy 模块中的架构方案。该方案旨在简化状态管理，避免复杂的同步逻辑，同时保持业务流程的清晰性。

## 设计原则

1. **工具化集成**：将记忆功能包装为 DSPy Tool，供 ReAct 模块调用
2. **状态极简**：ChatState 完全不涉及记忆字段，降低节点间耦合
3. **业务流程不变**：保持原有的业务节点执行顺序和信息收集逻辑
4. **集中存储**：通过专门的 `update_memories` 节点统一处理记忆更新
5. **零状态传递**：记忆搜索完全在 ReAct 内部处理，无需跨节点传递

## 架构设计

### 1. ChatState 保持不变

```python
class ChatState(TypedDict):
    # 保持原有的极简设计，无需额外记忆字段
    messages: Annotated[List[BaseMessage], add_messages]
    alert_info: Optional[AlertInfo]
    symptoms: Optional[List[str]]
    context: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    diagnostic_result: Optional[Dict[str, Any]]
    action_plan: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    report: Optional[Dict[str, Any]]
    errors: Optional[List[str]]
    
    # 不需要 memory_context 字段
    # ReAct 工具会在需要时直接搜索记忆，无需状态传递
```

### 2. Memory 工具类设计

```python
class MemoryTool:
    """记忆工具类 - 封装 Graphiti 功能供 DSPy 模块使用"""
    
    def __init__(self, graphiti_client: GraphitiClientWrapper):
        self.graphiti = graphiti_client
    
    async def search_memory(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """搜索历史记忆"""
        # 执行 Graphiti 搜索
        # 返回格式化的记忆上下文
        
    async def should_search_memory(self, stage: str, context: Dict[str, Any]) -> bool:
        """判断是否需要搜索记忆"""
        # 简单的启发式规则
        
    def format_memory_context(self, search_results: List[Dict]) -> Dict[str, Any]:
        """格式化记忆上下文"""
        # 将搜索结果转换为易用的上下文格式
```

### 3. DSPy 模块集成 - ReAct 方式

使用 DSPy 的 ReAct 框架和 Tool 系统集成记忆功能：

#### 3.1 Memory 工具函数定义

```python
async def search_historical_cases(query: str, case_type: str = "diagnosis") -> str:
    """搜索历史案例
    
    Args:
        query: 搜索查询，描述要查找的问题或症状
        case_type: 案例类型，如 diagnosis, solution, alert 等
        
    Returns:
        str: 格式化的历史案例信息
    """
    memory_tool = get_memory_tool()  # 获取全局记忆工具实例
    
    results = await memory_tool.search_memory(
        query=query,
        context={"case_type": case_type}
    )
    
    if not results:
        return "未找到相关历史案例"
    
    # 格式化返回结果
    formatted_cases = []
    for i, case in enumerate(results[:3], 1):  # 限制返回前3个结果
        formatted_cases.append(f"案例{i}: {case.get('content', '')}")
    
    return "\n".join(formatted_cases)

async def search_solution_patterns(problem_description: str) -> str:
    """搜索解决方案模式
    
    Args:
        problem_description: 问题描述
        
    Returns:
        str: 相关的解决方案和最佳实践
    """
    memory_tool = get_memory_tool()
    
    results = await memory_tool.search_memory(
        query=f"解决方案 {problem_description}",
        context={"case_type": "solution"}
    )
    
    if not results:
        return "未找到相关解决方案"
    
    solutions = [f"方案: {r.get('content', '')}" for r in results[:2]]
    return "\n".join(solutions)
```

#### 3.2 ReAct 模块实现

```python
from dspy import Tool
import dspy

class DiagnosticSignature(dspy.Signature):
    """智能诊断签名"""
    alert_info: str = dspy.InputField(desc="告警信息详情")
    symptoms: str = dspy.InputField(desc="观察到的症状列表")
    
    diagnosis: str = dspy.OutputField(desc="诊断结果和根因分析")
    confidence: float = dspy.OutputField(desc="诊断置信度 0-1")
    recommendations: str = dspy.OutputField(desc="建议的后续行动")

class DiagnosticAgent(dspy.Module):
    """基于 ReAct 的诊断智能体"""
    
    def __init__(self):
        super().__init__()
        
        # 定义记忆工具
        self.search_cases_tool = Tool(
            search_historical_cases,
            name="search_historical_cases",
            desc="搜索历史诊断案例，帮助分析当前问题"
        )
        
        self.search_solutions_tool = Tool(
            search_solution_patterns,
            name="search_solution_patterns", 
            desc="搜索相关问题的解决方案和最佳实践"
        )
        
        # 创建 ReAct 智能体
        self.react_agent = dspy.ReAct(
            signature=DiagnosticSignature,
            tools=[self.search_cases_tool, self.search_solutions_tool],
            max_iters=3  # 限制推理迭代次数
        )
    
    def forward(self, alert_info: AlertInfo, symptoms: List[str]) -> DiagnosticResult:
        """执行诊断推理"""
        
        # 准备输入
        alert_str = f"来源: {alert_info.source}, 消息: {alert_info.message}, 严重程度: {alert_info.severity}"
        symptoms_str = "; ".join(symptoms) if symptoms else "暂无明确症状"
        
        # ReAct 推理过程
        # 智能体会自动决定是否需要搜索历史案例或解决方案
        result = self.react_agent(
            alert_info=alert_str,
            symptoms=symptoms_str
        )
        
        return DiagnosticResult(
            root_cause=result.diagnosis,
            confidence_score=result.confidence,
            impact_analysis="基于历史案例分析",
            recommended_actions=result.recommendations.split(";") if result.recommendations else []
        )
```

#### 3.3 其他模块的 ReAct 集成

```python
# Alert Analyzer with ReAct
class AlertAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.search_alert_patterns_tool = Tool(
            search_alert_patterns,
            name="search_alert_patterns",
            desc="搜索类似告警的历史模式和关联性"
        )
        
        self.react_agent = dspy.ReAct(
            signature=AlertAnalysisSignature,
            tools=[self.search_alert_patterns_tool]
        )

# Action Planner with ReAct  
class ActionPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.search_action_history_tool = Tool(
            search_action_history,
            name="search_action_history", 
            desc="搜索类似问题的成功处理经验和行动方案"
        )
        
        self.react_agent = dspy.ReAct(
            signature=ActionPlanningSignature,
            tools=[self.search_action_history_tool]
        )
```

### 4. 工作流集成

#### 4.1 节点架构

```
initialize → router → [business_nodes | collect_info] → router → update_memories → finalize
```

#### 4.2 业务节点调用模式

```python
async def _diagnose_issue_node(self, state: ChatState) -> ChatState:
    """诊断问题节点 - ReAct 会自动调用记忆工具"""
    try:
        # 直接调用 ReAct 模块，内部会自动决定是否使用工具
        result = await asyncio.to_thread(
            self.diagnostic_agent.forward,
            state["alert_info"],
            state.get("symptoms", [])
        )
        
        # 更新状态
        return {**state, "diagnostic_result": result.dict()}
        
    except Exception as e:
        return self._create_error_state(state, e, "diagnose_issue")
```

#### 4.3 记忆更新节点

```python
async def _update_memories_node(self, state: ChatState) -> ChatState:
    """更新记忆节点 - 将处理过程中的信息存储为情节"""
    try:
        episodes = []
        
        # 根据处理结果生成情节
        if state.get("diagnostic_result"):
            episodes.append(self._create_diagnostic_episode(state))
        
        if state.get("action_plan"):
            episodes.append(self._create_action_episode(state))
        
        # 批量存储情节
        if episodes:
            await self.memory_tool.graphiti.batch_add_episodes(episodes)
        
        return state
        
    except Exception as e:
        return self._create_error_state(state, e, "update_memories")
```

## 优势分析

### 1. 架构简洁性
- **DSPy 标准化**：使用官方推荐的 ReAct + Tool 模式
- **零状态耦合**：完全消除记忆相关的状态字段和节点间依赖
- **职责清晰**：工具函数专注记忆搜索，ReAct 负责推理和工具调用

### 2. 灵活性
- **智能工具选择**：ReAct 根据推理需要自动选择合适的工具
- **多工具支持**：每个模块可定义多个记忆相关工具
- **独立决策**：每个业务节点独立决定记忆使用，无需协调

### 3. 可维护性
- **标准工具接口**：符合 DSPy Tool 规范，易于扩展
- **独立测试**：工具函数可单独测试，ReAct 模块可 Mock 工具
- **推理可观测**：ReAct 的推理过程可追踪和调试

### 4. ReAct 特有优势
- **思维链推理**：ReAct 提供 Thought-Action-Observation 循环
- **动态工具调用**：根据推理进展动态决定工具使用
- **内存管理**：记忆搜索结果仅在单次推理中有效，无需持久化

### 5. 极简设计优势
- **状态纯净**：ChatState 保持原有设计，无额外字段污染
- **降低复杂度**：消除记忆状态的验证、传递、清理逻辑
- **提高可读性**：业务节点代码更简洁，专注核心逻辑

## 实施计划

### Phase 1: 基础架构
1. **创建记忆工具函数**：实现 `search_historical_cases` 等工具函数
2. **创建 MemoryTool 类**：封装 Graphiti 基础功能
3. **添加 update_memories 节点**：实现记忆存储逻辑

### Phase 2: ReAct 集成
1. **修改 DiagnosticAgent**：改造为 ReAct 模式，集成记忆工具
2. **验证 ReAct 效果**：测试自动化记忆搜索和推理
3. **优化工具描述**：调整工具的 name 和 desc 以提高调用准确性

### Phase 3: 全面集成
1. **扩展到其他模块**：AlertAnalyzer, ActionPlanner 等都改造为 ReAct
2. **丰富工具生态**：为不同业务场景定义专门的记忆工具
3. **性能优化**：工具函数缓存、异步优化等

## 风险与缓解

### 1. 性能风险
- **风险**：每个模块都可能触发记忆搜索，增加延迟
- **缓解**：实现智能的 `should_search_memory` 逻辑，避免不必要搜索

### 2. 一致性风险
- **风险**：多个模块可能搜索到不同的记忆结果
- **缓解**：在 ChatState 中缓存搜索结果，模块间共享

### 3. 复杂性风险
- **风险**：记忆工具的复杂性可能影响模块的简洁性
- **缓解**：保持工具接口简单，复杂逻辑封装在工具内部

## 总结

这个设计方案通过将记忆功能工具化，实现了简洁的架构和灵活的集成。DSPy 模块保持了推理的纯粹性，同时获得了强大的记忆能力。工作流保持了清晰的业务逻辑，记忆管理被优雅地整合到现有架构中。

该方案既满足了项目早期"简单有效"的需求，又为后续的优化和扩展提供了良好的基础。