# DeepSeek LLM 集成指南

本文档介绍如何在智能运维智能体中配置和使用 DeepSeek 大语言模型。

## 🌟 为什么选择 DeepSeek

### 优势
- **高性价比**: API 调用成本远低于 GPT-4
- **中文优化**: 对中文理解和生成效果优秀
- **代码能力**: DeepSeek-Coder 在代码生成任务上表现突出
- **开源透明**: 模型架构和训练方法公开
- **API 兼容**: 支持 OpenAI 格式的 API 调用

### 适用场景
- 运维日志分析（中文日志友好）
- 故障诊断报告生成
- 自动化脚本生成
- 运维知识问答
- 成本敏感的大规模部署

## 🚀 快速配置

### 1. 获取 API Key
1. 访问 [DeepSeek 官网](https://www.deepseek.com/)
2. 注册账户并获取 API Key
3. 充值账户（支持微信、支付宝）

### 2. 环境配置
创建 `.env` 文件：
```bash
# 复制示例配置文件
cp .env.example .env

# 编辑配置文件
vim .env
```

设置必要的环境变量：
```bash
# DeepSeek 配置
DEEPSEEK_API_KEY="your-deepseek-api-key-here"
LLM_PROVIDER="deepseek"
LLM_MODEL_NAME="deepseek-chat"
LLM_BASE_URL="https://api.deepseek.com/v1"
LLM_TEMPERATURE="0.1"
LLM_MAX_TOKENS="2000"
```

### 3. 测试连接
```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行 DeepSeek 测试
python examples/deepseek_test.py
```

## 📋 配置选项

### 支持的模型
| 模型名称 | 描述 | 适用场景 | 建议参数 |
|---------|------|----------|----------|
| `deepseek-chat` | 通用对话模型 | 告警分析、诊断报告 | temp=0.1, max_tokens=2000 |
| `deepseek-coder` | 代码专用模型 | 脚本生成、代码分析 | temp=0.1, max_tokens=4000 |

### 参数调优建议
```python
# 保守配置（高准确性）
LLMConfig(
    provider="deepseek",
    model_name="deepseek-chat",
    temperature=0.1,        # 低温度保证稳定性
    max_tokens=2000,        # 适中的输出长度
    timeout=60              # 合理的超时时间
)

# 创意配置（更多样化输出）
LLMConfig(
    provider="deepseek",
    model_name="deepseek-chat",
    temperature=0.3,        # 稍高温度增加创意
    max_tokens=4000,        # 更长的输出
    timeout=120             # 更长的超时时间
)
```

## 🔧 代码集成

### 基础使用
```python
from src.utils.llm_config import setup_deepseek_llm, LLMConfig

# 配置 DeepSeek
config = LLMConfig(
    provider="deepseek",
    model_name="deepseek-chat",
    api_key="your-api-key"
)

# 初始化 LLM
dspy_lm, langchain_llm = setup_deepseek_llm(config)

# DSPy 使用
response = dspy_lm("请分析这个告警：CPU使用率90%")

# LangChain 使用
response = langchain_llm.invoke("生成运维检查清单")
```

### 智能体集成
```python
from src.agents.intelligent_ops_agent import IntelligentOpsAgent, AgentConfig

# 创建智能体（自动使用 DeepSeek）
agent_config = AgentConfig(
    agent_id="deepseek_ops_agent",
    agent_type="general",
    specialization="deepseek_powered"
)

agent = IntelligentOpsAgent(agent_config)

# 处理告警
alert = {
    "alert_id": "cpu_high_001",
    "severity": "high",
    "message": "CPU使用率持续超过85%",
    "metrics": {"cpu_usage": 0.90}
}

result = await agent.process_alert(alert)
```

## 🧪 测试验证

### 连接测试
```bash
# 测试基本连接
python -c "
from src.utils.llm_config import test_llm_connection
test_llm_connection()
"
```

### 功能测试
```bash
# 运行完整的 DeepSeek 测试套件
python examples/deepseek_test.py
```

### 性能测试
```python
import time
from src.utils.llm_config import setup_deepseek_llm, get_llm_config_from_env

# 测试响应时间
config = get_llm_config_from_env()
dspy_lm, _ = setup_deepseek_llm(config)

start_time = time.time()
response = dspy_lm("测试响应时间")
end_time = time.time()

print(f"响应时间: {end_time - start_time:.2f} 秒")
```

## 🔄 多模型切换

### 配置多个提供商
```python
from src.utils.llm_config import LLMConfig, DEEPSEEK_CONFIGS, OPENAI_CONFIGS

# DeepSeek 配置
deepseek_config = DEEPSEEK_CONFIGS["deepseek-chat"]
deepseek_config.api_key = "your-deepseek-key"

# OpenAI 备用配置
openai_config = OPENAI_CONFIGS["gpt-4"]
openai_config.api_key = "your-openai-key"

# 根据需要切换
current_config = deepseek_config  # 或 openai_config
```

### 环境变量切换
```bash
# 切换到 DeepSeek
export LLM_PROVIDER="deepseek"
export DEEPSEEK_API_KEY="your-key"

# 切换到 OpenAI
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-key"

# 切换到本地 Ollama
export LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"
```

## 📊 成本优化

### 成本控制策略
```python
# 设置合理的令牌限制
config = LLMConfig(
    max_tokens=1000,        # 限制输出长度
    temperature=0.1,        # 减少重复尝试
    timeout=30              # 避免长时间等待
)

# 使用缓存减少重复调用
import functools

@functools.lru_cache(maxsize=128)
def cached_llm_call(prompt: str) -> str:
    return dspy_lm(prompt)
```

### 监控使用情况
```python
class UsageTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0
    
    def track_usage(self, response):
        if hasattr(response, 'usage'):
            self.total_tokens += response.usage.total_tokens
            self.total_requests += 1
    
    def get_cost_estimate(self):
        # DeepSeek 价格约为 $0.0014/1K tokens
        return self.total_tokens * 0.0014 / 1000

tracker = UsageTracker()
```

## 🛠️ 故障排除

### 常见问题

#### 1. API Key 错误
```
错误: DeepSeek API key is required
解决: 设置 DEEPSEEK_API_KEY 环境变量
```

#### 2. 连接超时
```python
# 增加超时时间
config = LLMConfig(timeout=120)
```

#### 3. 速率限制
```python
import time
import random

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded")
```

#### 4. 中文编码问题
```python
# 确保正确的编码设置
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

### 日志调试
```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 在 LLM 调用中添加日志
logger.debug(f"发送请求: {prompt}")
response = dspy_lm(prompt)
logger.debug(f"收到响应: {response}")
```

## 📈 性能优化

### 批量处理
```python
async def batch_process_alerts(alerts):
    tasks = []
    for alert in alerts:
        task = agent.process_alert(alert)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 并发控制
```python
import asyncio

# 限制并发数量
semaphore = asyncio.Semaphore(5)

async def limited_llm_call(prompt):
    async with semaphore:
        return await llm_call(prompt)
```

## 🔒 安全最佳实践

### API Key 管理
```bash
# 使用环境变量而不是硬编码
export DEEPSEEK_API_KEY="sk-..."

# 或使用密钥管理服务
# AWS Secrets Manager, Azure Key Vault 等
```

### 敏感数据处理
```python
import re

def sanitize_prompt(prompt: str) -> str:
    # 移除敏感信息
    prompt = re.sub(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[CARD_MASKED]', prompt)
    prompt = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_MASKED]', prompt)
    return prompt
```

## 📚 参考资源

- [DeepSeek 官方文档](https://api-docs.deepseek.com/)
- [DSPy 文档](https://dspy-docs.vercel.app/)
- [LangChain 文档](https://python.langchain.com/docs/get_started/introduction)
- [智能运维最佳实践](./ops_best_practices.md)

## 🎯 下一步

1. **生产部署**: 配置负载均衡和故障转移
2. **监控告警**: 设置 API 使用量和成本监控
3. **模型微调**: 基于运维数据优化模型表现
4. **多模态集成**: 结合图像识别分析运维图表