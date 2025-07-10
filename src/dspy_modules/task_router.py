"""
DSPy 任务路由模块

基于 DSPy 的智能任务路由器，能够根据用户输入自动判断应该执行哪种任务类型。
"""

from typing import Any, Dict, List, Union
from dataclasses import dataclass

import dspy


class TaskRouterSignature(dspy.Signature):
    """智能任务路由签名"""
    user_input: str = dspy.InputField(desc="用户输入的原始数据或描述，包含告警、症状、问题描述等信息")
    available_tasks: str = dspy.InputField(desc="可用的任务类型列表及其描述")
    
    task_type: str = dspy.OutputField(desc="推荐的任务类型，必须是 available_tasks 中的一个")
    confidence: float = dspy.OutputField(desc="置信度，范围 0.0 到 1.0，表示对任务类型判断的确信程度")
    reasoning: str = dspy.OutputField(desc="推理过程，解释为什么选择这个任务类型")


@dataclass
class TaskRouterResult:
    """任务路由结果"""
    task_type: str
    confidence: float
    reasoning: str
    input_summary: str = ""
    fallback_reason: str = ""


class TaskRouter(dspy.Module):
    """智能任务路由器
    
    基于 DSPy 的模块化推理，能够根据用户输入自动判断应该执行哪种任务类型。
    支持多模态输入：告警信息、症状描述、自然语言等。
    """
    
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(TaskRouterSignature)
        
        # 定义可用的任务类型
        self.available_tasks = {
            "process_alert": "处理告警和监控事件，分析告警信息并生成处理建议",
            "diagnose_issue": "诊断问题和根因分析，基于症状和上下文信息进行深入分析",
            "plan_actions": "制定修复行动计划，基于诊断结果生成具体的解决方案",
            "execute_actions": "执行修复操作，按照行动计划进行系统修复",
            "generate_report": "生成事件报告，汇总整个事件处理过程",
            "learn_feedback": "学习用户反馈，从用户评价中改进系统性能"
        }
    
    def forward(self, user_input: Any) -> TaskRouterResult:
        """执行智能任务路由
        
        Args:
            user_input: 用户输入，可以是字典、字符串或其他格式
            
        Returns:
            TaskRouterResult: 路由结果，包含任务类型、置信度和推理过程
        """
        try:
            # 将输入转换为文本描述
            input_description = self._describe_input(user_input)
            
            # 构建任务类型描述
            tasks_description = self._format_available_tasks()
            
            # 执行路由推理
            result = self.router(
                user_input=input_description,
                available_tasks=tasks_description
            )
            
            # 验证和清理结果
            task_type = self._validate_task_type(result.task_type)
            confidence = self._validate_confidence(result.confidence)
            
            return TaskRouterResult(
                task_type=task_type,
                confidence=confidence,
                reasoning=result.reasoning,
                input_summary=input_description[:200] + "..." if len(input_description) > 200 else input_description
            )
            
        except Exception as e:
            # 路由失败，返回默认结果
            return TaskRouterResult(
                task_type="process_alert",  # 默认任务类型
                confidence=0.3,
                reasoning=f"DSPy 路由失败，使用默认任务类型。错误: {str(e)}",
                input_summary=str(user_input)[:100],
                fallback_reason=str(e)
            )
    
    def _describe_input(self, user_input: Any) -> str:
        """将输入转换为文本描述"""
        descriptions = []
        
        if isinstance(user_input, dict):
            # 字典格式输入
            descriptions.extend(self._parse_dict_input(user_input))
        elif isinstance(user_input, str):
            # 字符串格式输入
            descriptions.append(f"用户描述: {user_input}")
        elif isinstance(user_input, list):
            # 列表格式输入
            descriptions.append(f"列表数据: {', '.join(str(item) for item in user_input)}")
        else:
            # 其他格式
            descriptions.append(f"输入数据: {str(user_input)}")
        
        return "\n".join(descriptions) if descriptions else "无明确输入信息"
    
    def _parse_dict_input(self, data: Dict[str, Any]) -> List[str]:
        """解析字典格式输入"""
        descriptions = []
        
        # 告警信息
        if "alert_info" in data:
            alert = data["alert_info"]
            if isinstance(alert, dict):
                message = alert.get("message", "")
                severity = alert.get("severity", "")
                source = alert.get("source", "")
                descriptions.append(f"告警信息: {message} (严重程度: {severity}, 来源: {source})")
                
                # 添加指标信息
                if "metrics" in alert:
                    metrics_str = ", ".join(f"{k}: {v}" for k, v in alert["metrics"].items())
                    descriptions.append(f"监控指标: {metrics_str}")
        
        # 症状描述
        if "symptoms" in data:
            symptoms = data["symptoms"]
            if isinstance(symptoms, list):
                descriptions.append(f"症状描述: {', '.join(symptoms)}")
        
        # 上下文信息
        if "context" in data:
            context = data["context"]
            if isinstance(context, dict):
                descriptions.append(f"环境信息: {self._summarize_context(context)}")
        
        # 诊断结果
        if "diagnostic_result" in data:
            diag = data["diagnostic_result"]
            if isinstance(diag, dict):
                root_cause = diag.get("root_cause", "")
                impact = diag.get("impact_assessment", "")
                descriptions.append(f"诊断结果: 根因 - {root_cause}, 影响 - {impact}")
        
        # 行动计划
        if "action_plan" in data:
            plan = data["action_plan"]
            if isinstance(plan, dict):
                plan_id = plan.get("plan_id", "")
                priority = plan.get("priority", "")
                descriptions.append(f"行动计划: {plan_id} (优先级: {priority})")
        
        # 自然语言描述
        for field in ["message", "description", "text", "content"]:
            if field in data and isinstance(data[field], str):
                descriptions.append(f"用户描述: {data[field]}")
        
        # 事件信息
        if "incident_id" in data:
            descriptions.append(f"事件ID: {data['incident_id']}")
        
        # 反馈信息
        if any(field in data for field in ["feedback", "rating", "comments"]):
            feedback_info = []
            if "feedback" in data:
                feedback_info.append(f"反馈: {data['feedback']}")
            if "rating" in data:
                feedback_info.append(f"评分: {data['rating']}")
            if "comments" in data:
                feedback_info.append(f"评论: {data['comments']}")
            descriptions.append("用户反馈信息: " + ", ".join(feedback_info))
        
        return descriptions
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """总结上下文信息"""
        summary_parts = []
        
        if "system_metrics" in context:
            metrics = context["system_metrics"]
            if isinstance(metrics, dict):
                summary_parts.append(f"系统指标({len(metrics)}项)")
        
        if "log_entries" in context:
            logs = context["log_entries"]
            if isinstance(logs, list):
                summary_parts.append(f"日志条目({len(logs)}条)")
        
        if "topology_info" in context:
            summary_parts.append("拓扑信息")
        
        if "environment" in context:
            summary_parts.append(f"环境: {context['environment']}")
        
        return ", ".join(summary_parts) if summary_parts else "无特定上下文"
    
    def _format_available_tasks(self) -> str:
        """格式化可用任务类型"""
        task_lines = []
        for task_type, description in self.available_tasks.items():
            task_lines.append(f"- {task_type}: {description}")
        return "\n".join(task_lines)
    
    def _validate_task_type(self, task_type: str) -> str:
        """验证任务类型"""
        if task_type in self.available_tasks:
            return task_type
        
        # 尝试模糊匹配
        task_lower = task_type.lower()
        for valid_task in self.available_tasks.keys():
            if task_lower in valid_task.lower() or valid_task.lower() in task_lower:
                return valid_task
        
        # 默认返回第一个任务类型
        return "process_alert"
    
    def _validate_confidence(self, confidence: Union[str, float]) -> float:
        """验证置信度"""
        try:
            conf_value = float(confidence)
            return max(0.0, min(1.0, conf_value))
        except (ValueError, TypeError):
            return 0.5  # 默认置信度
    
    def get_task_description(self, task_type: str) -> str:
        """获取任务类型描述"""
        return self.available_tasks.get(task_type, "未知任务类型")
    
    def list_available_tasks(self) -> Dict[str, str]:
        """列出所有可用任务类型"""
        return self.available_tasks.copy()


# 用于测试的示例函数
def test_task_router():
    """测试任务路由器"""
    router = TaskRouter()
    
    # 测试用例
    test_cases = [
        # 告警输入
        {
            "alert_info": {
                "alert_id": "cpu_high_001",
                "message": "CPU usage exceeds 85%",
                "severity": "high",
                "source": "monitoring"
            }
        },
        # 症状输入
        {
            "symptoms": ["数据库连接超时", "API响应缓慢"],
            "context": {"environment": "production"}
        },
        # 诊断结果输入
        {
            "diagnostic_result": {
                "root_cause": "内存泄漏",
                "impact_assessment": "high"
            }
        },
        # 自然语言输入
        "服务器响应很慢，用户反馈页面加载不出来",
        # 反馈输入
        {
            "feedback": "处理效果很好",
            "rating": 5,
            "comments": "问题解决及时"
        }
    ]
    
    for i, test_input in enumerate(test_cases):
        print(f"\n=== 测试用例 {i+1} ===")
        print(f"输入: {test_input}")
        
        result = router.forward(test_input)
        print(f"任务类型: {result.task_type}")
        print(f"置信度: {result.confidence}")
        print(f"推理过程: {result.reasoning}")
        print(f"输入总结: {result.input_summary}")


if __name__ == "__main__":
    test_task_router()