"""
智能体 LangGraph Studio 集成模块
导出编译好的智能体图供 Studio 使用
"""

from src.agents.intelligent_ops_agent import IntelligentOpsAgent, AgentConfig

# 创建默认的智能体配置
default_config = AgentConfig(
    agent_id="ops_agent_studio",
    agent_type="general",
    specialization="studio_demo",
    enable_learning=True,
    enable_reporting=True,
    auto_execution=False
)

# 创建智能体实例
_agent_instance = IntelligentOpsAgent(default_config)

# 编译智能体图 - 这是 Studio 需要的主要对象
graph = _agent_instance.compile()

# 同时导出智能体实例供其他用途
agent = _agent_instance

# 导出供 LangGraph Studio 使用
__all__ = ["graph", "agent"]