# LangGraph Studio 测试示例

## 🚀 在 LangGraph Studio 中测试智能运维智能体

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
- **`ops-agent`** - 统一的智能运维智能体（推荐使用）

## 🏗️ 统一架构说明

### 核心特性
- **🔄 状态驱动**: 基于 `ChatState` 的统一状态管理
- **🧠 智能路由**: `IntelligentRouter` 模块分析完整状态并决定下一步
- **🎯 统一实现**: 所有功能集中在 `IntelligentOpsAgent` 类中
- **📈 中断恢复**: 支持 `collect_info` 节点的人工干预

### 工作流程
```
initialize → router → [business_nodes | collect_info] → router → finalize
```

**业务节点执行顺序**: 
`process_alert` → `diagnose_issue` → `plan_actions` → `execute_actions` → `generate_report`

**信息收集模式**: 
缺少前置条件 → `collect_info` → 更新状态 → `router` → 重新路由

## 📋 测试示例

### 🎯 快速开始（推荐）

#### 聊天模式
1. 启动服务器：`./scripts/start_server.sh`
2. 访问 Studio 并选择 `ops-agent` 图
3. 切换到 **Chat** 模式
4. 直接输入自然语言描述

**示例输入**:
```
我们的web服务器CPU使用率达到95%，已经持续了10分钟，用户开始报告网站访问缓慢
```

### 1. 完整告警处理测试

**输入 (ChatState 格式)**:
```json
{
  "alert_info": {
    "alert_id": "cpu_high_001",
    "timestamp": "2025-01-09T10:30:00Z",
    "severity": "high",
    "source": "prometheus",
    "message": "CPU usage exceeds 90% for 5 minutes",
    "metrics": {
      "cpu_usage": 0.95,
      "memory_usage": 0.65,
      "load_average": 3.2
    },
    "tags": ["cpu", "performance", "critical"]
  }
}
```

**预期流程**: `initialize` → `router` → `process_alert` → `router` → `diagnose_issue` → `router` → `plan_actions` → `router` → `finalize`

**预期输出**: 完整的运维处理报告，包括告警分析、诊断结果、行动计划

---

### 2. 症状诊断测试

**输入**:
```json
{
  "symptoms": [
    "API响应时间增加到5秒",
    "数据库连接池耗尽", 
    "内存使用率持续上升",
    "错误日志显示OutOfMemoryError"
  ],
  "context": {
    "system_type": "微服务架构",
    "peak_traffic": true,
    "recent_deployments": ["user-service-v2.1", "payment-service-v1.3"]
  }
}
```

**预期流程**: `initialize` → `router` → `diagnose_issue` → `router` → `plan_actions` → `router` → `finalize`

---

### 3. 仅消息输入测试 (Graph 模式)

**输入**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "生产环境数据库CPU使用率90%，web服务开始报500错误"
    }
  ]
}
```

**预期流程**: `initialize` → `router` → `collect_info` (从消息提取信息) → `router` → 业务流程

---

## ⚡ 中断恢复测试

### 🔧 中断机制说明

当路由器决定需要收集更多信息时，会自动路由到 `collect_info` 节点：
- **触发条件**: 业务节点缺少必要的前置数据
- **中断方式**: `collect_info` 节点调用 `request_operator_input()` 函数
- **恢复方式**: 用户提供信息后，自动返回 `router` 重新分析

### 场景 1: 信息不完整触发中断

**输入 (不完整信息)**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "系统有些问题，不太确定具体情况"
    }
  ]
}
```

**预期行为**:
1. `router` 分析发现信息不足
2. 路由到 `collect_info` 节点
3. **中断**: 显示"📝 需要补充信息"提示
4. **等待用户输入**: 在 Studio 中暂停

**Resume 输入示例** (直接文本):
```
发现数据库连接池使用率很高，而且有很多慢查询日志，API响应时间变慢
```

**预期恢复**: 
- 将用户输入添加到消息历史
- 返回 `router` 重新分析
- 基于补充信息路由到合适的业务节点

### 场景 2: 缺少告警详情触发中断

**输入**:
```json
{
  "messages": [
    {
      "role": "user", 
      "content": "CPU告警"
    }
  ]
}
```

**预期行为**:
1. `router` 尝试路由到 `process_alert`
2. 发现 `alert_info` 为空
3. 路由到 `collect_info` 收集告警详情
4. **中断**: 请求提供详细的告警信息

**Resume 输入示例**:
```
服务器CPU使用率95%，持续10分钟，告警ID是cpu_spike_001，来源是prometheus监控
```

---

## 🧠 智能路由测试

### Router 决策逻辑

`IntelligentRouter` 基于完整的 `ChatState` 分析并决定下一步：

1. **有 alert_info** → `process_alert`
2. **有 symptoms** → `diagnose_issue`  
3. **有 diagnostic_result** → `plan_actions`
4. **有 action_plan** → `execute_actions`
5. **缺少前置条件** → `collect_info`
6. **所有处理完成** → `generate_report` → `finalize`

### 测试路由决策

**测试 1: 直接告警路由**
```json
{
  "alert_info": {
    "alert_id": "test_001",
    "severity": "high", 
    "message": "Database connection timeout"
  }
}
```
**预期**: 直接路由到 `process_alert`

**测试 2: 诊断路由**
```json
{
  "symptoms": ["slow response", "high CPU"],
  "alert_info": {
    "alert_id": "processed_001"
  }
}
```
**预期**: 跳过 `process_alert`，直接路由到 `diagnose_issue`

---

## 🎛️ 高级测试场景

### 1. 完整工作流测试

**目标**: 测试从告警到报告的完整流程

**输入**:
```json
{
  "alert_info": {
    "alert_id": "full_test_001",
    "timestamp": "2025-01-09T15:00:00Z",
    "severity": "critical",
    "source": "monitoring",
    "message": "Web service completely down",
    "metrics": {"availability": 0.0},
    "tags": ["web", "outage", "critical"]
  }
}
```

**观察点**:
- ✅ Router 的路由决策日志
- ✅ 每个业务节点的处理结果
- ✅ 状态在节点间的传递
- ✅ 最终报告的生成

### 2. 错误处理测试

**输入 (无效数据)**:
```json
{
  "invalid_field": "test",
  "messages": [
    {
      "role": "user",
      "content": "invalid input test"
    }
  ]
}
```

**预期**: 
- Router 处理无效输入
- 路由到 `collect_info` 请求澄清
- 错误处理不会导致工作流崩溃

---

## 🛠️ 实际测试步骤

### 在 LangGraph Studio 中测试

1. **启动服务**:
   ```bash
   ./scripts/start_server.sh
   ```

2. **访问 Studio**: 
   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

3. **选择图**: `ops-agent`

4. **测试完整流程**:
   - **Graph 模式**: 使用上述 JSON 输入
   - **Chat 模式**: 直接文本对话

5. **测试中断恢复**:
   - 输入不完整信息
   - 观察中断触发
   - 使用 Resume 功能提供补充信息
   - 验证工作流正确恢复

---

## 🔍 观察要点

### 状态管理
- ✅ `ChatState` 字段的正确更新
- ✅ 消息历史的完整保存
- ✅ 业务数据的正确传递

### 路由逻辑
- ✅ Router 的决策准确性
- ✅ 条件检查的正确性
- ✅ 目标节点的设置

### 中断处理
- ✅ 中断触发的及时性
- ✅ 中断信息的清晰度
- ✅ 恢复后的正确执行

### 业务逻辑
- ✅ DSPy 模块的推理质量
- ✅ 告警分析的准确性
- ✅ 诊断结果的可信度
- ✅ 行动计划的合理性

---

## 🐛 故障排除

### 常见问题

1. **中断不触发**
   - 检查输入是否真的缺少必要信息
   - 确认 `collect_info` 节点的路由条件
   - 检查 `request_operator_input` 函数

2. **路由决策错误**
   - 检查 `IntelligentRouter` 的分析逻辑
   - 确认 `ChatState` 字段的正确设置
   - 查看 Router 的调试日志

3. **状态不一致**
   - 检查节点间的状态传递
   - 确认状态更新的原子性
   - 验证状态验证逻辑

### 调试技巧

1. **使用详细日志**: 观察每个节点的输入输出
2. **分步测试**: 逐个测试各个功能模块
3. **状态检查**: 在 Studio 中查看 `ChatState` 变化
4. **错误捕获**: 注意 `errors` 字段的内容

---

## 🎯 验证目标

通过这些测试，验证：

### 核心功能
- ✅ 状态驱动的工作流执行
- ✅ 智能路由的准确性
- ✅ 中断恢复的可靠性
- ✅ 业务逻辑的正确性

### 用户体验
- ✅ 自然语言交互的流畅性
- ✅ 中断提示的清晰度
- ✅ 错误处理的友好性
- ✅ 整体流程的直观性

### 技术集成
- ✅ DeepSeek LLM 的稳定性
- ✅ DSPy 模块的可靠性
- ✅ LangGraph 工作流的正确性
- ✅ Studio 界面的响应性

---

## 🚀 推荐测试流程

### 1. 快速验证
```bash
# 聊天模式测试
输入: "CPU使用率过高，需要分析"
预期: 完整的运维处理流程
```

### 2. 中断测试
```bash
# 不完整信息测试
输入: "系统有问题"
预期: 中断并请求更多信息
```

### 3. 完整流程测试
```bash
# 复杂场景测试
输入: 结构化的告警信息
预期: 完整的告警→诊断→规划→报告流程
```

成功通过这些测试后，您的智能运维系统就已经准备好处理真实的运维场景了！ 🎉