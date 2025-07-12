# LangGraph Server 设置与 Agent Chat UI 集成指南

## 概述

本指南详细说明如何将智能运维项目设置为 LangGraph 服务器，并与 `agent-chat-ui` 集成，实现基于 Web 的聊天界面。

## 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source .venv/bin/activate

# 确保所有依赖已安装
uv sync

# 安装项目到虚拟环境
uv pip install -e .
```

### 2. 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 文件，至少配置以下内容：
DEEPSEEK_API_KEY=your-deepseek-api-key-here
LLM_PROVIDER=deepseek
LLM_MODEL_NAME=deepseek-chat
```

### 3. 启动 LangGraph 服务器

```bash
# 启动开发服务器
langgraph dev --port 2024 --host 0.0.0.0 --no-browser

# 或者指定配置文件
langgraph dev --config langgraph.json --port 2024
```

服务器启动后，您将看到：
- 🚀 API: `http://0.0.0.0:2024`
- 🎨 Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024`
- 📚 API Docs: `http://0.0.0.0:2024/docs`

## 与 Agent Chat UI 集成

### 1. 安装 Agent Chat UI

```bash
# 方式1：使用 create-agent-chat-app
npx create-agent-chat-app my-ops-chat

# 方式2：克隆仓库
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
pnpm install
```

### 2. 配置 Agent Chat UI

创建或编辑 `.env.local` 文件：

```bash
# Next.js 应用配置
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NEXT_PUBLIC_ASSISTANT_ID=ops-agent

# LangGraph 服务器配置
LANGGRAPH_API_URL=http://localhost:2024
LANGGRAPH_API_KEY=  # 开发环境可以留空
```

### 3. 启动 Chat UI

```bash
# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev
```

Chat UI 将在 `http://localhost:3000` 启动。

## 架构说明

### 数据流

```
用户输入 -> Agent Chat UI -> LangGraph Server -> 智能运维工作流 -> 响应返回
```

## 支持的交互功能

### 📊 告警分析
```
用户：系统CPU使用率过高，请分析
助手：正在分析告警信息...
      📊 告警分析结果
      告警等级: high
      分类: performance
      影响评估: 系统性能受到影响...
```

### 🔍 故障诊断
```
用户：数据库连接异常，请诊断
助手：🔍 故障诊断结果
      根因分析: 数据库连接池耗尽
      修复建议: 重启数据库服务...
```

### ⚡ 修复建议
```
用户：请提供修复方案
助手：⚡ 行动计划
      1. 重启数据库服务
      2. 清理连接池
      3. 增加连接超时时间
```

### 📈 系统监控
```
用户：显示系统状态
助手：📈 系统监控状态
      CPU使用率: 75%
      内存使用率: 60%
      工作流状态: active
```

## 生产部署

### 1. 使用 Docker 部署

```bash
# 构建 Docker 镜像
langgraph build

# 运行容器
docker run -p 2024:2024 --env-file .env langgraph-ops-agent
```

### 2. 使用 LangGraph Cloud

```bash
# 部署到 LangGraph Cloud
langgraph deploy --config langgraph.json
```

### 3. 环境变量配置

生产环境需要配置：
- `DATABASE_URL`: PostgreSQL 连接字符串
- `REDIS_URL`: Redis 连接字符串
- `LANGSMITH_API_KEY`: LangSmith API 密钥

## 故障排除

### 常见问题

1. **服务器启动失败**
   ```bash
   # 检查依赖安装
   uv pip install -e .
   
   # 检查 langgraph.json 配置
   cat langgraph.json
   ```

2. **Chat UI 连接失败**
   ```bash
   # 检查 LANGGRAPH_API_URL 配置
   echo $LANGGRAPH_API_URL
   
   # 测试 API 连接
   curl http://localhost:2024/health
   ```

### 调试技巧

1. **启用详细日志**
   ```bash
   langgraph dev --server-log-level DEBUG
   ```

2. **使用 Studio UI 调试**
   访问 `https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024`

3. **API 文档查看**
   访问 `http://0.0.0.0:2024/docs`

## 下一步

1. 自定义聊天界面样式
2. 添加更多运维功能
3. 集成更多监控数据源
4. 实现多用户会话管理
5. 添加安全认证机制

## 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [Agent Chat UI 仓库](https://github.com/langchain-ai/agent-chat-ui)
- [LangGraph Studio 使用指南](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)