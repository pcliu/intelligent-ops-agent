import asyncio
from typing import Dict, Any, Optional, Callable, List
from langgraph.graph import StateGraph, END
from .state_manager import OpsState, StateManager
from .workflow_nodes import WorkflowNodes


class OpsWorkflow:
    """智能运维工作流
    
    整合 LangGraph 和 DSPy，实现智能化运维流程：
    1. 监控数据采集
    2. 告警处理和分析
    3. 故障诊断
    4. 行动规划
    5. 自动化执行
    6. 报告生成
    7. 反馈学习
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state_manager = StateManager()
        self.workflow_nodes = WorkflowNodes()
        self.graph = self._build_graph()
        self.compiled_graph = None
    
    def _build_graph(self) -> StateGraph:
        """构建工作流图"""
        # 创建状态图
        workflow = StateGraph(OpsState)
        
        # 添加节点
        nodes = self.workflow_nodes.get_node_router()
        for node_name, node_func in nodes.items():
            workflow.add_node(node_name, node_func)
        
        # 设置入口点
        workflow.set_entry_point("monitor_collect")
        
        # 添加边
        self._add_edges(workflow)
        
        return workflow
    
    def _add_edges(self, workflow: StateGraph):
        """添加工作流边"""
        # 获取条件边配置
        conditional_edges = self.workflow_nodes.get_conditional_edges()
        
        # 添加条件边
        for node_name, condition_func in conditional_edges.items():
            if node_name == "monitor_collect":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "alert_process": "alert_process",
                        "monitor_collect": "monitor_collect"
                    }
                )
            elif node_name == "alert_process":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "diagnosis": "diagnosis",
                        "error_handling": "error_handling"
                    }
                )
            elif node_name == "diagnosis":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "action_plan": "action_plan",
                        "error_handling": "error_handling"
                    }
                )
            elif node_name == "action_plan":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "action_execute": "action_execute",
                        "error_handling": "error_handling"
                    }
                )
            elif node_name == "action_execute":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "report_generate": "report_generate",
                        "error_handling": "error_handling"
                    }
                )
            elif node_name == "report_generate":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "feedback_learn": "feedback_learn",
                        "error_handling": "error_handling"
                    }
                )
            elif node_name == "feedback_learn":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "monitor_collect": "monitor_collect"
                    }
                )
            elif node_name == "error_handling":
                workflow.add_conditional_edges(
                    node_name,
                    condition_func,
                    {
                        "monitor_collect": "monitor_collect",
                        "alerting": "alert_process",
                        "diagnosis": "diagnosis",
                        "planning": "action_plan",
                        "execution": "action_execute",
                        "reporting": "report_generate",
                        "END": END
                    }
                )
    
    def compile(self):
        """编译工作流图"""
        self.compiled_graph = self.graph.compile()
        return self.compiled_graph
    
    async def run(self, initial_state: Optional[OpsState] = None, 
                  max_iterations: int = 100) -> OpsState:
        """运行工作流
        
        Args:
            initial_state: 初始状态
            max_iterations: 最大迭代次数
            
        Returns:
            OpsState: 最终状态
        """
        if not self.compiled_graph:
            self.compile()
        
        # 初始化状态
        if initial_state is None:
            initial_state = self.state_manager.initialize_state(
                workflow_id=f"ops_{asyncio.get_event_loop().time()}"
            )
        
        # 运行工作流
        result = await self.compiled_graph.ainvoke(
            initial_state,
            config={"recursion_limit": max_iterations}
        )
        
        return result
    
    async def stream_run(self, initial_state: Optional[OpsState] = None,
                        max_iterations: int = 100):
        """流式运行工作流
        
        Args:
            initial_state: 初始状态
            max_iterations: 最大迭代次数
            
        Yields:
            Dict: 每个步骤的状态更新
        """
        if not self.compiled_graph:
            self.compile()
        
        # 初始化状态
        if initial_state is None:
            initial_state = self.state_manager.initialize_state(
                workflow_id=f"ops_{asyncio.get_event_loop().time()}"
            )
        
        # 流式运行工作流
        async for step in self.compiled_graph.astream(
            initial_state,
            config={"recursion_limit": max_iterations}
        ):
            yield step
    
    def get_state_snapshot(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取状态快照
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            Dict: 状态快照
        """
        # 这里应该从持久化存储中获取状态
        # 当前实现返回当前状态的导出
        if self.state_manager.current_state:
            return self.state_manager.export_state(self.state_manager.current_state)
        return None
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """暂停工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            bool: 是否成功暂停
        """
        try:
            if self.state_manager.current_state:
                self.state_manager.update_workflow_status(
                    self.state_manager.current_state, 
                    "paused"
                )
                return True
            return False
        except Exception:
            return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """恢复工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            bool: 是否成功恢复
        """
        try:
            if self.state_manager.current_state:
                self.state_manager.update_workflow_status(
                    self.state_manager.current_state, 
                    "active"
                )
                return True
            return False
        except Exception:
            return False
    
    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流指标
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            Dict: 工作流指标
        """
        if not self.state_manager.current_state:
            return {}
        
        state = self.state_manager.current_state
        
        # 计算执行时间
        execution_time = (state["last_update"] - state["start_time"]).total_seconds()
        
        # 计算成功率
        total_errors = len(state["errors"])
        success_rate = max(0, 1 - (total_errors / 10))  # 假设最多10个错误
        
        return {
            "workflow_id": workflow_id,
            "execution_time_seconds": execution_time,
            "current_stage": state["workflow_stage"],
            "status": state["workflow_status"],
            "total_errors": total_errors,
            "retry_count": state["retry_count"],
            "success_rate": success_rate,
            "has_alert": state["current_alert"] is not None,
            "has_diagnosis": state["diagnostic_result"] is not None,
            "has_action_plan": state["action_plan"] is not None,
            "has_execution_result": state["execution_result"] is not None,
            "has_report": state["incident_report"] is not None
        }
    
    def validate_workflow(self) -> List[str]:
        """验证工作流配置
        
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        # 验证节点配置
        try:
            nodes = self.workflow_nodes.get_node_router()
            if not nodes:
                errors.append("No workflow nodes configured")
            
            required_nodes = [
                "monitor_collect", "alert_process", "diagnosis",
                "action_plan", "action_execute", "report_generate",
                "feedback_learn", "error_handling"
            ]
            
            for node in required_nodes:
                if node not in nodes:
                    errors.append(f"Missing required node: {node}")
        except Exception as e:
            errors.append(f"Node configuration error: {str(e)}")
        
        # 验证条件边配置
        try:
            conditional_edges = self.workflow_nodes.get_conditional_edges()
            if not conditional_edges:
                errors.append("No conditional edges configured")
        except Exception as e:
            errors.append(f"Conditional edges configuration error: {str(e)}")
        
        return errors
    
    def get_workflow_definition(self) -> Dict[str, Any]:
        """获取工作流定义
        
        Returns:
            Dict: 工作流定义
        """
        return {
            "name": "Intelligent Operations Workflow",
            "description": "智能运维工作流，整合LangGraph和DSPy实现智能化运维",
            "version": "1.0.0",
            "nodes": list(self.workflow_nodes.get_node_router().keys()),
            "entry_point": "monitor_collect",
            "state_schema": {
                "current_alert": "AlertInfo",
                "alert_analysis": "AlertAnalysisResult",
                "diagnostic_result": "DiagnosticResult",
                "action_plan": "ActionPlan",
                "execution_result": "ExecutionResult",
                "incident_report": "IncidentReport",
                "workflow_stage": "str",
                "workflow_status": "str",
                "system_metrics": "Dict[str, Any]"
            },
            "stages": [
                "monitoring",
                "alerting", 
                "diagnosis",
                "planning",
                "execution",
                "reporting"
            ],
            "error_handling": "Built-in error handling with retry mechanism",
            "feedback_loop": "Continuous learning from execution results"
        }


class WorkflowFactory:
    """工作流工厂"""
    
    @staticmethod
    def create_ops_workflow(config: Optional[Dict[str, Any]] = None) -> OpsWorkflow:
        """创建运维工作流
        
        Args:
            config: 配置参数
            
        Returns:
            OpsWorkflow: 运维工作流实例
        """
        return OpsWorkflow(config)
    
    @staticmethod
    def create_custom_workflow(nodes: Dict[str, Callable], 
                              edges: Dict[str, Any],
                              config: Optional[Dict[str, Any]] = None) -> OpsWorkflow:
        """创建自定义工作流
        
        Args:
            nodes: 节点配置
            edges: 边配置
            config: 配置参数
            
        Returns:
            OpsWorkflow: 自定义工作流实例
        """
        workflow = OpsWorkflow(config)
        
        # 这里可以添加自定义节点和边的逻辑
        # 由于复杂性，这里简化实现
        
        return workflow