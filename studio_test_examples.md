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

### 1. 生成报告测试

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

**预期输出**: 智能体生成的事件报告，包含分析、诊断、行动计划等信息。

---

### 2. 告警处理测试

**输入 (AlertInfo 格式)**:
```json
{
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
```

**预期输出**: 告警分析结果，包括优先级、分类、紧急程度评分、根因提示和建议行动。

---

### 3. 问题诊断测试

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

**预期输出**: 详细的诊断结果，包括根因分析、置信度、影响评估、受影响组件、业务影响等。

---

### 4. 行动规划测试

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

**预期输出**: 详细的行动计划，包括执行步骤、风险评估、时间预估、回滚计划等。

## 🔧 高级测试场景

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
2. **route_task** - 根据任务类型路由
3. **任务执行节点** - 执行具体功能
   - `process_alert` - 告警处理
   - `diagnose_issue` - 问题诊断  
   - `plan_actions` - 行动规划
   - `execute_actions` - 执行行动
   - `generate_report` - 生成报告
   - `learn_feedback` - 学习反馈
4. **finalize** - 完成任务并返回结果

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

1. 启动服务器：`./scripts/start_server.sh`
2. 访问 Studio：https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
3. 选择 `ops-agent` 图
4. 从报告生成测试开始
5. 逐步尝试更复杂的功能