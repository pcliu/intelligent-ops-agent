"""
LangGraph Studio 集成模块
直接导出编译好的图供 Studio 使用
"""
from src.langgraph_workflow.ops_workflow import OpsWorkflow

# 创建工作流实例
_workflow_instance = OpsWorkflow()

# 编译图 - 这是 Studio 需要的主要对象
graph = _workflow_instance.compile()

# 同时导出工作流实例供其他用途
workflow = _workflow_instance

# 导出供 LangGraph Studio 使用
__all__ = ["graph", "workflow"]