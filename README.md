# 智能运维智能体 (Intelligent Operations Agent)

基于 LangGraph 和 DSPy 框架构建的智能运维系统，实现智能化的故障诊断、自动化运维和持续学习优化。

## 🌟 项目特性

### 核心能力
- **🔄 工作流编排**: 基于 LangGraph 的智能运维工作流
- **🧠 模块化推理**: 使用 DSPy 优化推理和决策过程
- **🤖 多智能体协作**: 支持专业化智能体的协同工作
- **📊 智能诊断**: 自动化根因分析和影响评估
- **📋 行动规划**: 智能生成修复策略和执行计划
- **📈 持续学习**: 从运维经验中学习和优化
- **📄 报告生成**: 自动化生成运维报告和知识积累

### 技术架构
```
┌─────────────────────────────────────────────────────────────────┐
│                   智能运维智能体系统                              │
├─────────────────────────────────────────────────────────────────┤
│  LangGraph 工作流编排层                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  监控数据采集  →  告警处理  →  故障诊断  →  自动化执行       │ │
│  │       ↓           ↓           ↓           ↓                 │ │
│  │  状态管理    →  决策节点   →  执行节点  →  反馈学习          │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  DSPy 模块化推理层                                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  AlertAnalyzer  │  DiagnosticAgent  │  ActionPlanner        │ │
│  │       ↓              ↓                    ↓                 │ │
│  │  推理优化    │   根因分析        │   策略生成                │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  基础设施层                                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  监控系统    │  知识库        │  执行引擎      │  日志系统    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- LangGraph >= 0.2.0
- DSPy >= 2.4.0
- LangChain >= 0.2.0

### 安装依赖
```bash
# 克隆项目
git clone <repository-url>
cd intelligent-ops-agent

# 安装依赖
pip install -r requirements.txt
```

### 基础使用示例
```python
import asyncio
from src.agents.intelligent_ops_agent import IntelligentOpsAgent, AgentConfig
from src.dspy_modules.alert_analyzer import AlertInfo

# 创建智能体配置
config = AgentConfig(
    agent_id="ops_agent_001",
    agent_type="general",
    specialization="system_performance",
    enable_learning=True,
    enable_reporting=True
)

# 创建智能体
agent = IntelligentOpsAgent(config)

# 创建告警信息
alert = AlertInfo(
    alert_id="cpu_spike_001",
    timestamp="2024-01-01T12:00:00Z",
    severity="high",
    source="system_monitor",
    message="CPU usage exceeded 90% threshold",
    metrics={"cpu_usage": 0.95},
    tags=["cpu", "performance"]
)

# 处理告警
async def main():
    result = await agent.process_alert(alert)
    print(f"处理结果: {result['status']}")

asyncio.run(main())
```

## 📖 详细文档

### 项目结构
```
intelligent-ops-agent/
├── src/
│   ├── dspy_modules/          # DSPy 模块化推理组件
│   │   ├── alert_analyzer.py     # 告警分析器
│   │   ├── diagnostic_agent.py   # 诊断智能体
│   │   ├── action_planner.py     # 行动规划器
│   │   └── report_generator.py   # 报告生成器
│   ├── langgraph_workflow/    # LangGraph 工作流
│   │   ├── ops_workflow.py       # 主工作流
│   │   ├── workflow_nodes.py     # 工作流节点
│   │   └── state_manager.py      # 状态管理器
│   └── agents/                # 智能体实现
│       └── intelligent_ops_agent.py  # 主智能体类
├── examples/                  # 使用示例
│   ├── basic_usage.py            # 基础使用示例
│   ├── multi_agent_scenario.py  # 多智能体场景
│   └── complete_demo.py          # 完整功能演示
├── tests/                     # 测试文件
├── docs/                      # 文档
│   └── architecture.md           # 架构设计文档
└── requirements.txt           # 依赖列表
```

### 核心组件说明

#### 1. DSPy 模块化推理组件

**AlertAnalyzer (告警分析器)**
- 告警信息解析和分类
- 紧急程度评估
- 告警关联分析
- 根因提示生成

**DiagnosticAgent (诊断智能体)**
- 根因分析
- 影响范围评估
- 历史案例检索
- 诊断报告生成

**ActionPlanner (行动规划器)**
- 修复策略生成
- 风险评估
- 执行步骤规划
- 回滚计划制定

**ReportGenerator (报告生成器)**
- 事件报告生成
- 性能分析报告
- 趋势分析
- 优化建议

#### 2. LangGraph 工作流编排

**工作流阶段**
- `monitoring`: 监控数据采集
- `alerting`: 告警处理
- `diagnosis`: 故障诊断
- `planning`: 行动规划
- `execution`: 自动化执行
- `reporting`: 报告生成

**状态管理**
- 工作流状态跟踪
- 错误处理和重试
- 状态持久化
- 历史记录管理

#### 3. 智能体系统

**单智能体模式**
- 独立处理运维任务
- 完整的工作流执行
- 自主学习和优化

**多智能体协作模式**
- 专业化智能体分工
- 协作处理复杂问题
- 知识共享和整合

## 💡 使用场景

### 1. 系统监控和告警处理
```python
# 处理CPU使用率告警
cpu_alert = AlertInfo(
    alert_id="cpu_high_001",
    severity="high",
    message="CPU usage exceeding threshold",
    metrics={"cpu_usage": 0.95}
)

result = await agent.process_alert(cpu_alert)
```

### 2. 故障诊断和根因分析
```python
# 基于症状进行诊断
symptoms = [
    "API response time increased",
    "Database connection timeouts",
    "Memory usage gradually increasing"
]

diagnosis = await agent.diagnose_issue(symptoms, context)
```

### 3. 自动化行动规划
```python
# 生成修复计划
action_plan = await agent.plan_actions(diagnostic_result, system_context)
```

### 4. 多智能体协作
```python
# 创建专业化智能体
network_agent = IntelligentOpsAgent(AgentConfig(
    agent_id="network_specialist",
    agent_type="network",
    specialization="network_diagnostics"
))

security_agent = IntelligentOpsAgent(AgentConfig(
    agent_id="security_specialist", 
    agent_type="security",
    specialization="security_response"
))
```

## 🎯 运行示例

### 基础功能演示
```bash
python examples/basic_usage.py
```

### 多智能体协作演示
```bash
python examples/multi_agent_scenario.py
```

### 完整功能演示
```bash
python examples/complete_demo.py
```

## 🧪 测试

```bash
# 运行单元测试
python -m pytest tests/unit/

# 运行集成测试
python -m pytest tests/integration/

# 代码格式检查
python -m black .
python -m isort .
python -m flake8 .
```

## 🔧 配置

### 环境变量
```bash
# DSPy 配置
export DSPY_LANGUAGE_MODEL="your-llm-provider"
export DSPY_API_KEY="your-api-key"

# LangGraph 配置
export LANGGRAPH_CONFIG_PATH="path/to/config.yaml"

# 日志配置
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"
```

### 配置文件示例
```bash
# .env 配置文件
# DeepSeek LLM 配置（推荐 - 成本低，中文友好）
DEEPSEEK_API_KEY="your-deepseek-api-key-here"
LLM_PROVIDER="deepseek"
LLM_MODEL_NAME="deepseek-chat"
LLM_BASE_URL="https://api.deepseek.com/v1"
LLM_TEMPERATURE="0.1"
LLM_MAX_TOKENS="2000"

# LangGraph 配置
LANGGRAPH_MAX_ITERATIONS="100"
LANGGRAPH_TIMEOUT="300"
LANGGRAPH_ENABLE_CHECKPOINTS="true"

# 智能体配置
AGENT_ENABLE_LEARNING="true"
AGENT_ENABLE_REPORTING="true"
AGENT_AUTO_EXECUTION="false"
AGENT_MAX_RETRIES="3"
```

### DeepSeek 快速配置指南
1. **获取 API Key**: 访问 [DeepSeek 官网](https://www.deepseek.com/) 注册并获取 API Key
2. **配置环境**: 复制 `.env.example` 为 `.env` 并填入您的 API Key
3. **测试连接**: 运行 `python examples/deepseek_test.py` 验证配置

详细的 DeepSeek 集成说明请参考 [DeepSeek 集成指南](docs/deepseek_integration.md)。

## 📈 性能监控

### 指标收集
- 智能体响应时间
- 诊断准确率
- 自动化成功率
- 用户满意度

### 监控仪表板
- 实时工作流状态
- 性能趋势分析
- 错误率统计
- 资源使用情况

## 🔒 安全考虑

### 访问控制
- 基于角色的权限管理
- API 密钥认证
- 操作审计日志

### 数据保护
- 敏感信息脱敏
- 加密存储
- 安全传输

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排框架
- [DSPy](https://github.com/stanfordnlp/dspy) - 模块化推理框架
- [LangChain](https://github.com/langchain-ai/langchain) - 基础框架支持

## 📞 联系我们

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 文档: [Documentation Site]

---

**智能运维智能体** - 让运维更智能，让系统更可靠！ 🚀