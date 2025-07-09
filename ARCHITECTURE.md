# 智能运维系统架构说明

## 智能体图架构

### 双层架构设计

我们现在采用双层架构，既有工作流图，也有智能体图：

```
langgraph.json 
    ↓ 包含两个图
├── ops-workflow: 底层工作流图
│   └── ops_workflow_studio.py:graph
└── ops-agent: 高层智能体图
    └── intelligent_ops_agent_studio.py:graph
```

### 关键文件说明

#### 1. `langgraph.json`
- **作用**: LangGraph Studio 的主配置文件
- **关键配置**: 
  - `"ops-workflow"`: 底层工作流图
  - `"ops-agent"`: 高层智能体图
- **说明**: 每个图都直接指向编译好的图对象

#### 2. 工作流层 (`src/langgraph_workflow/`)
- **`ops_workflow.py`**: 核心工作流定义
- **`ops_workflow_studio.py`**: 工作流 Studio 集成
- **职责**: 
  - 完整的运维流程编排
  - 监控→告警→诊断→行动→报告→反馈
  - 7个核心节点的复杂工作流

#### 3. 智能体层 (`src/agents/`)
- **`intelligent_ops_agent.py`**: 智能体图实现
- **`intelligent_ops_agent_studio.py`**: 智能体 Studio 集成
- **职责**:
  - 任务导向的智能体行为
  - 初始化→路由→执行→完成
  - 支持多种任务类型 (process_alert, diagnose_issue, etc.)

### 架构优势

1. **分层设计**: 工作流处理复杂流程，智能体处理任务导向
2. **独立编译**: 每个图单独编译，避免耦合
3. **灵活组合**: 可以单独使用工作流或智能体
4. **Studio 支持**: 两个图都可以在 Studio 中可视化和调试

### 图的区别

#### 工作流图 (ops-workflow)
- **用途**: 完整的智能运维流程
- **状态**: `OpsState` - 包含完整的运维状态
- **节点**: 7个专业节点 (monitor_collect, alert_process, etc.)
- **适用**: 端到端的运维自动化

#### 智能体图 (ops-agent)
- **用途**: 任务导向的智能体行为
- **状态**: `AgentState` - 包含智能体任务状态
- **节点**: 10个通用节点 (initialize, route_task, etc.)
- **适用**: 特定任务的智能处理

### 编译流程

```mermaid
graph TD
    A[系统启动] --> B[加载 langgraph.json]
    B --> C[ops_workflow_studio.py]
    B --> D[intelligent_ops_agent_studio.py]
    C --> E[编译工作流图]
    D --> F[编译智能体图]
    E --> G[Studio: ops-workflow]
    F --> H[Studio: ops-agent]
```

### 使用方式

#### LangGraph Studio
```
访问: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
图选择: 
  - ops-workflow: 完整运维流程
  - ops-agent: 智能体任务处理
```

#### Python 代码
```python
# 使用工作流图
from src.langgraph_workflow.ops_workflow_studio import graph as workflow_graph
result = await workflow_graph.ainvoke(ops_state)

# 使用智能体图
from src.agents.intelligent_ops_agent import IntelligentOpsAgent
agent = IntelligentOpsAgent(config)
result = await agent.process_alert(alert_data)

# 或直接使用编译好的智能体图
from src.agents.intelligent_ops_agent_studio import graph as agent_graph
result = await agent_graph.ainvoke(agent_state)
```

### 状态结构

#### OpsState (工作流状态)
```python
{
    "current_alert": AlertInfo,
    "alert_analysis": AlertAnalysisResult,
    "diagnostic_result": DiagnosticResult,
    "action_plan": ActionPlan,
    "execution_result": ExecutionResult,
    "incident_report": IncidentReport,
    "workflow_stage": str,
    "workflow_status": str,
    # ... 更多运维相关状态
}
```

#### AgentState (智能体状态)
```python
{
    "agent_id": str,
    "current_task": str,
    "task_input": Dict,
    "task_output": Dict,
    "status": str,
    "stage": str,
    "analysis_result": Dict,
    "diagnostic_result": Dict,
    # ... 更多智能体相关状态
}
```

### 配置更新

根据需要修改不同层级：

1. **工作流层修改**:
   - 修改 `ops_workflow.py`
   - 重启 LangGraph 服务器
   - 影响 ops-workflow 图

2. **智能体层修改**:
   - 修改 `intelligent_ops_agent.py`
   - 重启 LangGraph 服务器
   - 影响 ops-agent 图

这种架构提供了更好的模块化和可扩展性，让不同层级的逻辑分离，便于维护和扩展。