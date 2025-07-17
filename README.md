# 智能运维智能体 (Intelligent Operations Agent)

**基于 LangGraph 和 DSPy 的状态驱动智能运维系统**

结合工作流编排和模块化推理，实现智能化的运维决策、故障诊断和自动化处理。

## ✨ 核心特性

### 🎯 智能化运维能力
- **🔄 状态驱动工作流**: 基于 LangGraph 的统一状态管理
- **🧠 模块化推理**: 使用 DSPy 优化决策过程
- **🎯 智能路由**: 基于完整状态的智能决策路由
- **📊 自动化诊断**: 根因分析和影响评估
- **🛠️ 行动规划**: 智能生成修复策略
- **📈 中断恢复**: 支持人工干预和工作流恢复

### 🏗️ 统一架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      智能运维智能体系统                            │
│                   (State-Driven Architecture)                   │
├─────────────────────────────────────────────────────────────────┤
│  LangGraph 工作流编排层 (统一在 IntelligentOpsAgent 中)          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  initialize → router → [business_nodes | collect_info]      │ │
│  │                 ↓                                           │ │
│  │  process_alert → diagnose_issue → plan_actions              │ │
│  │                 ↓                                           │ │
│  │  execute_actions → generate_report → finalize               │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  DSPy 模块化推理层 (Chain-of-Thought)                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  IntelligentRouter  │  AlertAnalyzer  │  DiagnosticAgent   │ │
│  │         ↓                   ↓                   ↓           │ │
│  │  状态路由决策       │  告警分析分类   │   根因分析        │ │
│  │                                                             │ │
│  │  ActionPlanner    │  ReportGenerator                       │ │
│  │         ↓                   ↓                               │ │
│  │  策略生成规划     │   报告生成                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  统一状态管理层 (ChatState)                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  messages        │  alert_info     │  symptoms           │ │
│  │  (用户交互)      │  (业务数据)     │  (诊断信息)         │ │
│  │                                                             │ │
│  │  diagnostic_result │ action_plan    │  execution_result  │ │
│  │  (诊断结果)        │ (执行计划)     │  (执行结果)        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求
- **Python 3.11+**
- **uv** (现代 Python 包管理器)
- **DeepSeek API Key** (推荐，成本低且中文友好)

### 1. 安装 uv 包管理器
```bash
# macOS 和 Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 pip
pip install uv
```

### 2. 项目设置
```bash
# 克隆项目
git clone <repository-url>
cd intelligent-ops-agent

# 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
```

### 3. 配置环境
```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，配置 DeepSeek API
DEEPSEEK_API_KEY="your-deepseek-api-key-here"
LLM_PROVIDER="deepseek"
LLM_MODEL_NAME="deepseek-chat"
LLM_BASE_URL="https://api.deepseek.com/v1"
LLM_TEMPERATURE="0.1"
LLM_MAX_TOKENS="2000"
```

### 4. 验证安装
```bash
# 测试 DeepSeek 连接
python examples/deepseek_test.py

# 运行基础示例
python examples/basic_usage.py
```

## 💡 核心概念

### 状态驱动架构
系统采用统一的 `ChatState` 作为唯一数据源：
- **业务数据**: `alert_info`, `symptoms`, `diagnostic_result`, `action_plan` 等
- **用户交互**: `messages` 字段用于 LangGraph Studio 聊天界面
- **智能路由**: Router 分析完整状态，决定下一步执行节点

### 工作流程
1. **初始化** → **智能路由** → **业务节点**
2. **信息缺失** → **收集信息** → **更新状态** → **重新路由**
3. **完整流程**: `告警处理` → `故障诊断` → `行动规划` → `执行操作` → `生成报告`

## 🛠️ 使用示例

### 基础告警处理
```python
import asyncio
from src.agents.intelligent_ops_agent import IntelligentOpsAgent
from src.utils.state_models import ChatState, AlertInfo

# 创建智能体
agent = IntelligentOpsAgent()

# 创建告警状态
state = ChatState(
    messages=[],
    alert_info=AlertInfo(
        alert_id="cpu_high_001",
        timestamp="2024-01-01T12:00:00Z",
        severity="high",
        source="prometheus",
        message="CPU usage exceeded 90% for 5 minutes",
        metrics={"cpu_usage": 0.95, "duration": 300},
        tags=["cpu", "performance", "critical"]
    )
)

# 运行工作流
async def main():
    result = await agent.run_workflow(state)
    print(f"处理完成: {result.get('report', {}).get('summary', 'No report')}")

asyncio.run(main())
```

### 交互式诊断 (LangGraph Studio)
```python
# 在 LangGraph Studio 中使用
# 系统会自动处理中断和人工干预

# 1. 启动工作流
initial_state = ChatState(messages=[], alert_info=alert_data)

# 2. 系统自动路由到合适的处理节点
# 3. 如需额外信息，会中断并请求用户输入
# 4. 用户提供信息后，系统继续处理
# 5. 生成最终的诊断报告和行动计划
```

### 多步骤诊断场景
```python
# 复杂故障诊断
complex_state = ChatState(
    messages=[],
    symptoms=[
        "API 响应时间增加到 5 秒",
        "数据库连接池耗尽",
        "内存使用率持续上升",
        "错误日志显示 OutOfMemoryError"
    ],
    context={
        "system_type": "微服务架构",
        "peak_traffic": True,
        "recent_deployments": ["user-service-v2.1", "payment-service-v1.3"]
    }
)

# 运行诊断流程
diagnosis_result = await agent.run_workflow(complex_state)
```

## 📁 项目结构

```
intelligent-ops-agent/
├── src/
│   ├── agents/
│   │   └── intelligent_ops_agent.py    # 统一工作流实现
│   ├── dspy_modules/                    # DSPy 推理模块
│   │   ├── __init__.py
│   │   ├── intelligent_router.py       # 智能路由器
│   │   ├── alert_analyzer.py           # 告警分析器
│   │   ├── diagnostic_agent.py         # 诊断智能体
│   │   ├── action_planner.py           # 行动规划器
│   │   └── report_generator.py         # 报告生成器
│   ├── langgraph_workflow/             # LangGraph 工作流(保留目录)
│   └── utils/
│       ├── __init__.py
│       └── llm_config.py               # LLM 配置管理
├── examples/                           # 使用示例
│   └── quick_deepseek_test.py          # DeepSeek 配置测试
├── tests/                             # 测试套件
│   ├── unit/                          # 单元测试(空)
│   └── integration/                   # 集成测试(空)
├── scripts/                           # 脚本工具
│   ├── setup_chat_ui_manual.sh        # 聊天界面设置
│   └── start_server.sh                # 服务器启动脚本
├── docs/                              # 文档
│   └── studio_test_examples.md        # Studio 测试示例
├── intelligent-ops-chat-ui/           # 聊天界面(可选)
├── pyproject.toml                     # 项目配置
├── uv.lock                            # 依赖锁文件
├── langgraph.json                     # LangGraph 配置
├── requirements.txt                   # 依赖列表
├── CLAUDE.md                          # Claude Code 项目指南
└── README.md                          # 项目说明
```

## 🔧 开发工具

### 依赖管理
```bash
# 安装生产依赖
uv sync

# 安装开发依赖
uv sync --extra dev

# 添加新依赖
uv add package-name
uv add --dev package-name  # 开发依赖

# 移除依赖
uv remove package-name
```

### 代码质量
```bash
# 激活虚拟环境
source .venv/bin/activate

# 代码格式化
uv run black .
uv run isort .

# 代码检查
uv run flake8 .
uv run mypy src/

# 运行测试
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest --cov=src --cov-report=html
```

## 🎛️ 配置选项

### DeepSeek (推荐)
```bash
DEEPSEEK_API_KEY="your-api-key"
LLM_PROVIDER="deepseek"
LLM_MODEL_NAME="deepseek-chat"
LLM_BASE_URL="https://api.deepseek.com/v1"
```

### 其他 LLM 提供商
```bash
# OpenAI
LLM_PROVIDER="openai"
OPENAI_API_KEY="your-openai-key"
LLM_MODEL_NAME="gpt-4"

# 本地 Ollama
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://localhost:11434"
LLM_MODEL_NAME="llama3"
```

## 🔍 核心模块详解

### 1. IntelligentRouter (智能路由器)
- 分析完整的 ChatState 状态
- 基于业务数据和用户交互决定下一步
- 支持信息收集和业务节点的智能切换

### 2. AlertAnalyzer (告警分析器)
- 告警分类和优先级评估
- 关联分析和根因提示
- 输出结构化的告警信息

### 3. DiagnosticAgent (诊断智能体)
- 基于症状的根因分析
- 影响范围评估
- 历史案例匹配

### 4. ActionPlanner (行动规划器)
- 生成修复策略
- 风险评估和回滚计划
- 执行步骤规划

### 5. ReportGenerator (报告生成器)
- 事件总结和分析报告
- 性能趋势分析
- 优化建议生成

## 🚀 运行示例

```bash
# 基础功能演示
python examples/basic_usage.py

# 完整功能演示 (包含多步诊断)
python examples/complete_demo.py

# DeepSeek 配置验证
python examples/deepseek_test.py
```

## 🤝 贡献指南

1. Fork 本项目
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - 工作流编排框架
- **[DSPy](https://github.com/stanfordnlp/dspy)** - 模块化推理框架  
- **[DeepSeek](https://www.deepseek.com/)** - 高性价比 LLM 服务

---

## 🤔 常见问题解答

### Q: 为什么没有使用 checkpointer 仍然支持消息历史和中断恢复？

这是一个很好的技术问题！让我们深入解析 LangGraph 的状态管理机制：

#### 🔍 技术原理分析

**1. Messages 字段的特殊处理**

```python
# ChatState 中的关键设计
messages: Annotated[List[BaseMessage], add_messages]  # 聊天消息列表
```

**`add_messages` 注解的作用**：
- 这是 LangGraph 的内置消息处理器
- 自动处理消息的累积和状态管理
- 即使没有持久化 checkpointer，也会在**内存中**维护消息历史

**2. LangGraph Studio 的内置状态管理**

```python
# 代码中的编译方式 - 注意没有传入 checkpointer
graph = _studio_agent.compile()
```

**Studio 的状态管理特性**：
- **会话级内存存储**：Studio 在运行时维护工作流状态
- **自动状态快照**：每个节点执行后自动保存状态
- **内置中断支持**：不需要持久化就能处理中断和恢复

**3. 中断机制的工作原理**

```python
# interrupt() 函数的使用
human_response = interrupt(interrupt_data)
```

**LangGraph 的 `interrupt()` 函数特点**：
- **不依赖 checkpointer**：中断是工作流执行的暂停，不是状态持久化
- **基于执行流控制**：通过异常机制暂停当前执行流
- **Studio 集成**：LangGraph Studio 自动捕获中断并提供 UI 交互

#### 📊 功能对比表

| 功能 | 有 Checkpointer | 无 Checkpointer |
|------|-----------------|------------------|
| **消息历史** | ✅ 持久化存储 | ✅ 内存中维护 |
| **中断/恢复** | ✅ 可跨会话恢复 | ✅ 会话内恢复 |
| **状态保持** | ✅ 永久保存 | ✅ 执行期间保持 |
| **错误恢复** | ✅ 可回滚到任意点 | ✅ 当前会话内恢复 |
| **跨会话持续性** | ✅ 支持 | ❌ 不支持 |

#### 🎯 设计理念

**LangGraph Studio 的设计哲学**：
- **开发友好**：快速测试和调试，不需要配置持久化
- **即时反馈**：实时查看状态变化和工作流执行
- **简化部署**：减少外部依赖，专注于工作流逻辑

**内存状态管理的优势**：
- **性能高效**：无 I/O 开销，状态访问极快
- **开发调试**：状态变化实时可见，便于调试
- **简单部署**：无需数据库或持久化存储

#### 🔧 何时需要 Checkpointer？

**需要 Checkpointer 的场景**：
- 🔄 **长期会话**：需要跨服务重启恢复状态
- 💾 **状态持久化**：重要的状态数据需要永久保存  
- 🌐 **生产环境**：多用户、高可用性要求
- 📊 **审计需求**：需要追踪完整的执行历史

**启用 Checkpointer 的方式**：
```python
# 生产环境中的配置示例
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
compiled_graph = agent.compile(checkpointer=checkpointer)
```

#### 💡 关键要点

LangGraph 的巧妙设计在于：

1. **分层的状态管理**：区分运行时状态和持久化状态
2. **Studio 的内置支持**：提供开发时的便利功能
3. **灵活的配置选项**：根据需求选择是否启用持久化
4. **消息的特殊处理**：`add_messages` 注解提供自动状态管理

这种设计让开发者能够：
- **快速开发**：无需配置即可测试完整功能
- **渐进增强**：需要时再添加持久化支持
- **专注逻辑**：将注意力集中在业务逻辑而非基础设施

---

**智能运维智能体** - 让运维决策更智能，让故障处理更高效！ 🚀