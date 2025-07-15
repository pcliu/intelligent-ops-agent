# LangGraph Studio 测试示例

## 🚀 在 LangGraph Studio 中测试智能运维代理

### 启动服务器
使用启动脚本：
```bash
./scripts/start_server.sh
```

或手动启动：
```bash
source .venv/bin/activate
langgraph dev --port 2024 --host 127.0.0.1 --no-browser
```

### 访问 Studio UI
Studio URL: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

### 可用的图
现在有两个独立的图可供选择：
- **`ops-workflow`** - 底层工作流图（完整的运维流程）
- **`ops-agent`** - 高层智能体图（任务导向的智能处理）

推荐使用 `ops-agent` 进行交互式测试。

## 📋 测试示例

### 🚀 新功能：自然语言交互 + DSPy 智能路由

现在智能体支持**自然语言交互**和**自动任务判断**！您可以：
- 🗣️ 直接用自然语言描述问题
- 🧠 自动理解和提取结构化信息
- 🎯 智能判断应该执行哪种任务
- 📊 完整的运维处理流程

### 🧠 自然语言理解流程
1. **理解输入** (`understand_input`) - 解析自然语言，提取告警信息、症状、上下文
2. **智能路由** (`route_task`) - 基于理解结果自动选择处理路径
3. **连续处理** - 自动执行完整的运维流程链

### 1. 自然语言输入测试 🆕

**输入 (字符串)**:
```json
"我们的web服务器CPU使用率达到95%，已经持续了10分钟，用户开始报告网站访问缓慢"
```

**预期流程**: `initialize` → `understand_input` → `route_task` → `process_alert` → `diagnose_issue` → `plan_actions`
**预期输出**: 
- 🧠 自然语言理解提取告警信息
- 📊 告警分析和分类
- 🔍 问题诊断和根因分析
- 📋 行动计划建议

**更多自然语言示例**:
```json
"数据库连接超时，响应很慢，可能影响用户体验"
```

```json
"生产环境磁盘空间不足，剩余不到10%，需要紧急处理"
```

---

### 2. 告警处理测试 (简化格式)

**输入 (AlertInfo 格式)**:
```json
{
  "alert_info": {
    "alert_id": "cpu_high_001",
    "timestamp": "2025-01-09T10:30:00Z",
    "severity": "high",
    "source": "monitoring_system",
    "message": "CPU usage exceeds 85% for 5 minutes",
    "metrics": {
      "cpu_usage": 0.87,
      "memory_usage": 0.65,
      "load_average": 3.2
    },
    "tags": ["cpu", "performance", "critical"]
  }
}
```

**预期路由**: `process_alert` → `diagnose_issue` → `plan_actions` (连续工作流)
**预期输出**: 完整的运维处理流程，包括告警分析、问题诊断和行动规划

---


### 3. 🎯 Graph 模式 JSON 输入格式

当使用 LangGraph Studio 的 Graph 模式时，如果只需要输入 messages，按照以下格式拼接 JSON：


### 智能运维场景示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "系统告警：CPU使用率超过90%，内存使用率85%"
    }
  ]
}
```

### 4. 症状列表测试

**输入 (症状数组)**:
```json
[
  "数据库连接超时",
  "API响应缓慢", 
  "内存使用率持续上升"
]
```

**预期路由**: `diagnose_issue`
**预期输出**: 基于症状的根因分析和诊断结果

---

### 5. 报告生成测试

**输入**:
```json
{
  "incident_id": "test_001",
  "description": "测试智能体报告生成功能",
  "test_data": {
    "timestamp": "2025-01-09T10:30:00Z",
    "source": "studio_test"
  }
}
```

**预期路由**: `generate_report`
**预期输出**: 智能体生成的事件报告，包含分析、诊断、行动计划等信息

---

### 6. 问题诊断测试 (带上下文)

**输入**:
```json
{
  "symptoms": [
    "数据库连接超时",
    "API响应缓慢",
    "内存使用率持续上升"
  ],
  "context": {
    "system_metrics": {
      "cpu_usage": 0.75,
      "memory_usage": 0.85,
      "disk_io": "high"
    },
    "log_entries": [
      "Connection pool exhausted",
      "Slow query detected",
      "Memory leak in process X"
    ],
    "topology_info": {
      "services": ["web", "api", "database"],
      "dependencies": {
        "web": ["api"],
        "api": ["database"]
      }
    }
  }
}
```

**预期路由**: `diagnose_issue`
**预期输出**: 详细的诊断结果，包括根因分析、置信度、影响评估、受影响组件、业务影响等

---

### 7. 行动规划测试

**输入**:
```json
{
  "diagnostic_result": {
    "root_cause": "数据库连接池耗尽",
    "confidence_score": 0.85,
    "impact_assessment": "high",
    "business_impact": "用户无法访问服务",
    "recovery_estimate": "15分钟",
    "affected_components": ["database", "api"],
    "evidence": ["连接池监控指标", "错误日志"]
  },
  "context": {
    "environment": "production",
    "maintenance_window": false,
    "auto_scaling": true
  }
}
```

**预期路由**: `plan_actions`
**预期输出**: 详细的行动计划，包括执行步骤、风险评估、时间预估、回滚计划等

---

### 🌟 自然语言交互专项测试

#### 场景 1: 性能问题描述
**输入**:
```json
"我们的API响应时间从平常的200ms变成了2秒，用户投诉很多，需要快速定位问题"
```

**预期 NLU 提取**:
- 意图: `diagnose_issue`
- 症状: ["API响应时间变慢", "用户投诉增多"]
- 上下文: {"performance_degradation": true, "user_impact": true}

#### 场景 2: 告警级联故障
**输入**:
```json
"刚刚收到告警：数据库CPU 90%，接着web服务开始报500错误，现在负载均衡器也显示异常"
```

**预期 NLU 提取**:
- 意图: `process_alert`
- 告警信息: 数据库CPU高使用率告警
- 症状: ["web服务500错误", "负载均衡器异常"]
- 上下文: {"cascade_failure": true}

#### 场景 3: 容量规划问题
**输入**:
```json
"生产环境磁盘使用率已经达到85%，而且每天增长约2%，担心很快会满"
```

**预期 NLU 提取**:
- 意图: `plan_actions`
- 告警信息: 磁盘空间不足
- 上下文: {"environment": "production", "growth_trend": "2% daily"}

#### 场景 4: 混合中英文输入
**输入**:
```json
"Production环境的Redis连接数过高，已经超过max_connections，应用开始抛出connection timeout异常"
```

**预期 NLU 提取**:
- 意图: `process_alert`
- 告警信息: Redis连接数过高
- 症状: ["connection timeout异常"]
- 上下文: {"environment": "production"}

### 7. 反馈学习测试

**输入**:
```json
{
  "feedback": "处理效果很好，问题解决及时",
  "rating": 5,
  "comments": "建议增加自动通知功能",
  "incident_id": "test_001"
}
```

**预期路由**: `learn_feedback`
**预期输出**: 反馈学习结果和系统改进建议

---

## 🧠 DSPy 智能路由说明

### 路由工作原理
1. **输入分析**: DSPy 分析用户输入的内容和结构
2. **意图识别**: 使用 Chain-of-Thought 推理判断用户意图
3. **置信度评估**: 返回 0-1 的置信度分数
4. **智能回退**: 低置信度时使用规则路由

### 路由决策示例
- 包含 `alert_info` → `process_alert`
- 包含 `symptoms` → `diagnose_issue`
- 包含 `diagnostic_result` → `plan_actions`
- 包含 `action_plan` → `execute_actions`
- 包含 `incident_id` 或 `description` → `generate_report`
- 包含 `feedback` 或 `rating` → `learn_feedback`
- 自然语言描述 → 基于内容智能判断

### 置信度阈值
- **> 0.6**: 使用 DSPy 路由结果
- **≤ 0.6**: 回退到规则路由
- **异常**: 使用默认路由 (`process_alert`)

## 🔧 高级测试场景

## ⚡ 人工干预和中断场景测试

基于 LangGraph 的 `interrupt()` 机制，以下是实际可测试的人工干预场景：

### 场景 1: 低置信度诊断请求人工输入 🤔

**背景**: 当诊断置信度低于 0.7 时，系统会主动请求运维人员提供额外信息

**输入 (症状诊断)**:
```json
{
  "messages": [
    {
      "role": "user", 
      "content": "系统有些异常但不确定具体问题，响应有点慢"
    }
  ]
}
```

**预期中断**: 
- 在 `_diagnose_issue_node` 中触发 `request_operator_input()` 
- 显示中断数据：诊断置信度、初步判断、请求额外信息
- **Studio 行为**: 暂停执行，等待人工输入

**Resume 输入示例** (⚠️ **已修复** - 直接输入文本，无需JSON格式):
```
我发现数据库连接池使用率很高，而且有很多慢查询日志
```

**预期恢复**: 基于额外信息重新诊断，提高置信度

---

### 场景 2: 意图澄清中断 ❓

**背景**: 当自然语言理解置信度低于 0.5 时，请求用户澄清意图

**输入 (模糊表述)**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "有点问题，帮忙看看"
    }
  ]
}
```

**预期中断**:
- 在 `_understand_and_route_node` 中触发 `request_clarification()`
- 显示低置信度NLU结果和澄清请求
- **Studio 行为**: 等待澄清输入

**Resume 输入示例** (⚠️ **已修复** - 直接输入文本):
```
Web服务器CPU使用率过高，用户访问很慢
```

**预期恢复**: 重新进行NLU并继续正常流程

---

### 场景 3: 高风险操作审批中断 ⚠️

**背景**: 执行包含重启、删除等高风险操作时，自动请求审批

**输入 (触发高风险执行)**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "数据库完全卡死了，需要立即重启数据库服务"
    }
  ]
}
```

**预期中断**:
- 完成诊断和规划后，在 `_execute_actions_node` 中触发 `request_execution_approval()`
- 显示行动计划和风险评估
- **Studio 行为**: 等待审批决策

**Resume 输入选项** (⚠️ **已修复** - 直接输入决策文本):

审批通过:
```
approved
```

审批拒绝:
```
rejected
```

修改请求:
```
modified
```

或带说明的回复:
```
同意重启，已通知相关团队
```

**预期恢复**: 根据审批决策继续或修改执行计划

---

### 场景 4: 自定义运维人员交互 💬

**背景**: 在任何需要人工确认或输入的场景中使用

**输入 (复杂故障)**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "生产环境出现了奇怪的间歇性故障，不知道从哪里开始排查"
    }
  ]
}
```

**可能的中断场景**:
1. **信息收集阶段**: 请求更多故障细节
2. **诊断阶段**: 请求确认某些假设
3. **规划阶段**: 请求对建议方案的反馈

**Resume 输入示例** (⚠️ **已修复** - 直接输入文本):
```
故障主要发生在晚上7-9点，影响订单支付功能，错误日志显示超时
```

---

## 🛠️ 实际测试步骤

### 在 LangGraph Studio 中测试中断

1. **启动服务器**:
   ```bash
   ./scripts/start_server.sh
   ```

2. **访问 Studio**: 
   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

3. **选择图**: `ops-agent`

4. **测试低置信度诊断中断**:
   
   **Graph 模式输入**:
   ```json
   {
     "messages": [
       {
         "role": "user",
         "content": "系统有些问题，不太确定"
       }
     ]
   }
   ```
   
   **观察**: 系统在诊断节点暂停，显示中断信息
   
   **Resume 操作**: 点击Resume按钮，输入详细信息

5. **测试执行审批中断**:
   
   **Chat 模式输入**:
   ```
   数据库连接池耗尽，需要重启数据库服务
   ```
   
   **观察**: 系统在执行节点暂停，请求审批
   
   **Resume 操作**: 提供审批决策

---

## 🔄 中断和恢复的状态管理

### 中断点识别
代码中的自动中断点：
- `_understand_and_route_node`: NLU置信度 < 0.5
- `_diagnose_issue_node`: 诊断置信度 < 0.7  
- `_execute_actions_node`: 检测到高风险操作

### 静态中断点设置
编译时设置的中断点：
```python
compile_kwargs["interrupt_before"] = ["execute_actions"]
```

### 状态保持
中断时保持的状态：
- `ChatState` 中的所有字段
- `messages` 消息历史
- `context` 上下文信息
- 中间处理结果 (`analysis_result`, `diagnostic_result` 等)

---

## 🧪 验证要点

### 中断机制验证
- ✅ 中断正确触发（在预期条件下）
- ✅ 中断数据完整传递
- ✅ Studio UI 正确显示中断状态

### 恢复机制验证  
- ✅ Resume 操作成功恢复执行
- ✅ 状态完整保持
- ✅ 基于人工输入正确调整后续处理

### 交互体验验证
- ✅ 中断提示信息清晰有用
- ✅ Resume 输入格式简单明确
- ✅ 整个流程自然流畅

### 错误处理验证
- ✅ 无效的Resume输入得到正确处理
- ✅ 超时或取消操作有合理响应
- ✅ 多次Resume操作状态一致

通过这些基于实际代码的中断测试，可以验证智能运维系统的人机协作能力！

### 场景 1: 完整的事件处理流程

1. **步骤 1**: 提交告警
```
{"alert_id": "web_down_001", "severity": "critical", "message": "Web服务无响应"}
```

2. **步骤 2**: 获取状态检查结果
```
status
```

3. **步骤 3**: 进行问题诊断
```json
{
  "symptoms": ["Web服务无响应", "负载均衡器报错"],
  "context": {"environment": "production"}
}
```

4. **步骤 4**: 制定行动计划
```json
{
  "diagnostic_result": {
    "root_cause": "Web服务器进程崩溃",
    "confidence_score": 0.9,
    "impact_assessment": "critical"
  },
  "system_context": {"environment": "production"}
}
```

### 场景 2: 性能问题分析

**输入**:
```json
{
  "alert_id": "perf_degradation_001",
  "severity": "medium",
  "source": "apm_system",
  "message": "API响应时间增加300%",
  "metrics": {
    "response_time": 1500,
    "throughput": 50,
    "error_rate": 0.05
  },
  "tags": ["performance", "api", "latency"]
}
```

### 场景 3: 资源不足告警

**输入**:
```json
{
  "alert_id": "resource_low_001", 
  "severity": "high",
  "source": "infrastructure_monitoring",
  "message": "磁盘空间不足，剩余空间小于10%",
  "metrics": {
    "disk_usage": 0.92,
    "free_space": "2.1GB",
    "inode_usage": 0.87
  },
  "tags": ["disk", "storage", "capacity"]
}
```

## 📊 预期的图执行流程

在 LangGraph Studio 中，您将看到以下节点执行：

### ops-agent 图（推荐测试）:
1. **initialize** - 初始化智能体状态
2. **understand_input** - 🧠 **自然语言理解** - 解析输入，提取结构化信息
3. **route_task** - ✨ **DSPy 智能路由** - 自动判断任务类型
4. **连续工作流执行** - 告警处理的完整流程链：
   - `process_alert` → `diagnose_issue` → `plan_actions` → (`execute_actions`) → (`generate_report`)
   - 或独立任务：`generate_report`, `learn_feedback`
5. **finalize** - 完成任务并返回结果

### 🔄 新的连续工作流逻辑：
- **告警输入** → 自动执行完整的运维流程链
- **症状输入** → 直接进入诊断阶段，然后继续后续流程
- **诊断结果输入** → 直接进入行动规划
- **其他输入** → 执行对应的独立任务

### 🔍 观察要点
- **understand_input 节点**: 🧠 查看自然语言理解过程和信息提取结果
- **route_task 节点**: ✨ 查看 DSPy 路由决策过程和意图识别
- **任务节点**: 📊 查看 DSPy 模块的推理输出和分析结果
- **状态变化**: 🔄 观察 AgentState 在各节点间的演变，特别是 NLU 字段的变化

### ops-workflow 图（完整流程）:
1. **monitor_collect** - 监控数据收集
2. **alert_process** - 告警处理
3. **fault_diagnose** - 故障诊断
4. **action_plan** - 行动规划
5. **action_execute** - 执行行动
6. **report_generate** - 生成报告
7. **feedback_learn** - 反馈学习

## 🐛 故障排除

### 常见问题

1. **Studio 连接失败**
   - 确认服务器在端口 2024 上运行
   - 检查浏览器是否允许 HTTPS 访问 HTTP 本地服务器
   - 尝试直接访问 API 文档: http://127.0.0.1:2024/docs

2. **智能体未初始化**
   - 检查 `.env` 文件中的 DEEPSEEK_API_KEY 配置
   - 确保网络连接正常

3. **JSON 解析错误**
   - 确保 JSON 格式正确
   - 使用在线 JSON 验证工具检查

4. **模块导入错误**
   - 确保项目已安装: `uv pip install -e .`
   - 检查依赖是否完整

5. **图选择错误**
   - 确认选择的图存在（ops-agent 或 ops-workflow）
   - 如果看到多余的图（如 studio-ops-agent），请重启服务器

### 调试技巧

1. **查看详细日志**: 在 Studio 中观察每个节点的状态变化
2. **分步测试**: 先测试简单的报告生成，再测试复杂功能
3. **检查错误消息**: 注意返回的 `errors` 字段和状态信息
4. **使用 API 文档**: 直接在 http://127.0.0.1:2024/docs 中测试

## 🎯 测试目标

通过这些测试，您可以验证：

### 双层架构验证
- ✅ **ops-agent 图**: 任务导向的智能体行为
- ✅ **ops-workflow 图**: 完整的运维流程编排
- ✅ 两个图的独立性和功能完整性

### 核心功能验证
- ✅ 智能体初始化和状态管理
- ✅ 告警处理和分析能力
- ✅ 问题诊断和根因分析
- ✅ 行动规划和风险评估
- ✅ 报告生成和学习反馈
- ✅ 错误处理和恢复机制

### 集成测试
- ✅ DeepSeek LLM 集成正常
- ✅ DSPy 模块功能正确
- ✅ LangGraph 工作流执行稳定
- ✅ Studio 界面交互流畅

成功通过这些测试后，您就可以确信智能运维系统已经准备好在生产环境中使用了！

## 🚀 快速开始

### 方式1: 聊天模式（推荐）🆕
1. 启动服务器：`./scripts/start_server.sh`
2. 访问 Studio：https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
3. 选择 `ops-agent` 图 
4. 切换到 **Chat** 模式
5. 直接输入自然语言：`我们的web服务器CPU使用率达到95%，持续了10分钟`
6. 享受聊天式的智能运维体验！

> 🎉 **新架构优势**: 现在 `ops-agent` 图直接来自 `src/agents/intelligent_ops_agent.py`，无需单独的包装文件！

### 方式2: 图模式（详细调试）
1. 启动服务器：`./scripts/start_server.sh`
2. 访问 Studio：https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
3. 选择 `ops-agent` 图
4. 保持 **Graph** 模式
5. 使用下方的测试示例进行输入
6. 观察详细的节点执行过程

## 🎯 聊天模式特点

- **🗣️ 自然对话**: 直接用中文描述问题
- **🤖 智能回复**: 结构化的运维建议和分析
- **📊 实时处理**: 即时的问题分析和解决方案
- **🔍 上下文理解**: 理解运维场景和业务影响
- **⚡ 简单易用**: 无需复杂的 JSON 输入

## 📝 聊天模式示例

### 简单告警描述
```
CPU使用率过高，需要分析一下
```

### 复杂问题描述
```
我们的web服务器CPU使用率达到95%，已经持续了10分钟，用户开始报告网站访问缓慢
```

### 多症状描述
```
数据库连接超时，API响应很慢，用户投诉增多，可能是级联故障
```

### 容量规划问题
```
生产环境磁盘使用率已经达到85%，而且每天增长约2%，担心很快会满
```

---

## 🔄 中断恢复功能修复说明

### ⚠️ 重要更新：中断恢复输入格式已修复

**问题**: 之前的中断恢复需要JSON格式输入，导致用户体验不佳。

**修复**: 现在支持直接文本输入，无需JSON包装。

### 🛠️ 修复的技术细节

1. **中断函数增强**:
   - `request_operator_input()` - 运维人员输入请求
   - `request_execution_approval()` - 执行审批请求  
   - `request_clarification()` - 意图澄清请求

2. **返回值处理**:
   ```python
   # 自动处理不同类型的返回值
   if isinstance(human_response, dict):
       return human_response.get("response", "")
   elif isinstance(human_response, str):
       return human_response  # 直接返回字符串
   else:
       return str(human_response) if human_response else ""
   ```

3. **消息历史保存**:
   - 中断前自动添加 AI 请求消息
   - 恢复后自动添加用户回复消息
   - 保持完整的对话上下文

### 🎯 现在的使用方式

#### 在 LangGraph Studio 中恢复中断：

1. **触发中断** - 系统自动暂停并显示中断信息
2. **点击 Resume** - 在 Studio 界面点击恢复按钮
3. **直接输入文本** - 无需JSON，直接输入您的回复
4. **点击 Submit** - 系统自动继续执行

#### 输入示例：

**诊断补充**:
```
我发现数据库连接池使用率很高，而且有很多慢查询日志
```

**执行审批**:
```
approved
```
或
```
同意重启，已通知团队
```

**意图澄清**:
```
帮我分析数据库性能问题，查询响应时间很慢
```

### ✅ 验证修复效果

运行测试确认修复成功：
```bash
source .venv/bin/activate
python test_interrupt_fix.py
```

**测试结果**: 
- ✅ 中断函数正确处理字符串返回值
- ✅ AI 消息格式清晰易读
- ✅ 消息历史完整保存
- ✅ 兼容不同输入格式

### 🎉 现在可以享受流畅的中断恢复体验了！