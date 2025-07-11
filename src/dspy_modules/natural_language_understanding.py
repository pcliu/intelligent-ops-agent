"""
自然语言理解模块

基于 DSPy 的自然语言理解，用于处理用户的自然语言输入
"""

import dspy
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class IntentClassification(dspy.Signature):
    """意图分类签名"""
    input_text: str = dspy.InputField(desc="用户输入的自然语言文本")
    intent: str = dspy.OutputField(desc="识别的意图，可选值：process_alert, diagnose_issue, plan_actions, execute_actions, generate_report, learn_feedback")
    confidence: float = dspy.OutputField(desc="置信度，0-1之间的数值")
    reasoning: str = dspy.OutputField(desc="分类推理过程")


class InformationExtraction(dspy.Signature):
    """信息提取签名"""
    input_text: str = dspy.InputField(desc="用户输入的自然语言文本")
    intent: str = dspy.InputField(desc="已识别的意图")
    extracted_info: str = dspy.OutputField(desc="提取的结构化信息，JSON格式字符串")
    reasoning: str = dspy.OutputField(desc="信息提取的推理过程")


class AlertParsing(dspy.Signature):
    """告警信息解析签名"""
    input_text: str = dspy.InputField(desc="包含告警信息的自然语言文本")
    alert_severity: str = dspy.OutputField(desc="告警严重程度：critical, high, medium, low")
    alert_source: str = dspy.OutputField(desc="告警来源系统或组件")
    alert_message: str = dspy.OutputField(desc="规范化的告警消息")
    metrics: str = dspy.OutputField(desc="提取的指标信息，JSON格式字符串")
    tags: str = dspy.OutputField(desc="相关标签，逗号分隔")
    reasoning: str = dspy.OutputField(desc="告警解析的推理过程")


class NLUResult(BaseModel):
    """自然语言理解结果"""
    intent: str
    confidence: float
    extracted_info: Dict[str, Any]
    alert_info: Optional[Dict[str, Any]] = None
    symptoms: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    reasoning: str
    raw_input: str
    
    class Config:
        extra = "allow"


class NaturalLanguageUnderstanding(dspy.Module):
    """自然语言理解模块
    
    使用 DSPy 进行自然语言理解，包括意图分类、信息提取和告警解析
    """
    
    def __init__(self):
        super().__init__()
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.info_extractor = dspy.ChainOfThought(InformationExtraction)
        self.alert_parser = dspy.ChainOfThought(AlertParsing)
        
        # 预定义的意图映射
        self.intent_mapping = {
            "process_alert": "处理告警",
            "diagnose_issue": "诊断问题",
            "plan_actions": "制定行动计划",
            "execute_actions": "执行行动",
            "generate_report": "生成报告",
            "learn_feedback": "学习反馈"
        }
        
        # 告警相关关键词
        self.alert_keywords = {
            "cpu", "内存", "磁盘", "网络", "服务", "数据库", "响应", "延迟", "错误", "异常",
            "超时", "连接", "故障", "宕机", "告警", "监控", "指标", "阈值", "性能",
            "memory", "disk", "network", "service", "database", "response", "latency",
            "error", "exception", "timeout", "connection", "failure", "alert", "monitoring"
        }
        
        # 严重程度关键词
        self.severity_keywords = {
            "critical": ["严重", "紧急", "关键", "崩溃", "宕机", "无响应", "critical", "urgent", "down"],
            "high": ["高", "重要", "异常", "超出", "超过", "high", "important", "exceed"],
            "medium": ["中等", "一般", "轻微", "medium", "moderate", "minor"],
            "low": ["低", "提醒", "注意", "low", "notice", "warning"]
        }
    
    def forward(self, input_text: str) -> NLUResult:
        """处理自然语言输入"""
        try:
            # 步骤1：意图分类
            intent_result = self.intent_classifier(input_text=input_text)
            
            # 步骤2：信息提取
            info_result = self.info_extractor(
                input_text=input_text,
                intent=intent_result.intent
            )
            
            # 解析提取的信息
            extracted_info = self._parse_extracted_info(info_result.extracted_info)
            
            # 步骤3：如果是告警相关，进行告警解析
            alert_info = None
            if self._is_alert_related(input_text, intent_result.intent):
                alert_result = self.alert_parser(input_text=input_text)
                alert_info = self._build_alert_info(alert_result, input_text)
            
            # 步骤4：提取症状和上下文
            symptoms = self._extract_symptoms(input_text, extracted_info)
            context = self._extract_context(input_text, extracted_info)
            
            # 构建结果
            return NLUResult(
                intent=intent_result.intent,
                confidence=intent_result.confidence,
                extracted_info=extracted_info,
                alert_info=alert_info,
                symptoms=symptoms,
                context=context,
                reasoning=f"意图分类: {intent_result.reasoning} | 信息提取: {info_result.reasoning}",
                raw_input=input_text
            )
            
        except Exception as e:
            # 错误处理：返回默认结果
            return NLUResult(
                intent="process_alert",
                confidence=0.3,
                extracted_info={"message": input_text},
                reasoning=f"NLU处理异常: {str(e)}，使用默认解析",
                raw_input=input_text
            )
    
    def _parse_extracted_info(self, extracted_info_str: str) -> Dict[str, Any]:
        """解析提取的信息字符串"""
        try:
            import json
            return json.loads(extracted_info_str)
        except:
            return {"message": extracted_info_str}
    
    def _is_alert_related(self, input_text: str, intent: str) -> bool:
        """判断是否为告警相关"""
        if intent == "process_alert":
            return True
        
        # 检查是否包含告警关键词
        text_lower = input_text.lower()
        return any(keyword in text_lower for keyword in self.alert_keywords)
    
    def _build_alert_info(self, alert_result, input_text: str) -> Dict[str, Any]:
        """构建告警信息"""
        try:
            # 解析指标信息
            metrics = {}
            try:
                import json
                metrics = json.loads(alert_result.metrics)
            except:
                metrics = self._extract_metrics_from_text(input_text)
            
            # 解析标签
            tags = [tag.strip() for tag in alert_result.tags.split(",") if tag.strip()]
            
            return {
                "alert_id": f"nl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": alert_result.alert_severity,
                "source": alert_result.alert_source or "natural_language_input",
                "message": alert_result.alert_message,
                "metrics": metrics,
                "tags": tags,
                "raw_input": input_text
            }
        except Exception as e:
            # 构建基本告警信息
            return {
                "alert_id": f"nl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "severity": self._detect_severity(input_text),
                "source": "natural_language_input",
                "message": input_text,
                "metrics": self._extract_metrics_from_text(input_text),
                "tags": ["natural_language"],
                "raw_input": input_text
            }
    
    def _extract_metrics_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取指标信息"""
        import re
        metrics = {}
        
        # 提取百分比
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(percentage_pattern, text)
        if percentages:
            metrics["percentage_values"] = [float(p) for p in percentages]
        
        # 提取时间
        time_pattern = r'(\d+)\s*分钟'
        times = re.findall(time_pattern, text)
        if times:
            metrics["duration_minutes"] = [int(t) for t in times]
        
        # 提取数值
        number_pattern = r'(\d+(?:\.\d+)?)'
        numbers = re.findall(number_pattern, text)
        if numbers:
            metrics["numeric_values"] = [float(n) for n in numbers[:3]]  # 限制数量
        
        return metrics
    
    def _detect_severity(self, text: str) -> str:
        """检测严重程度"""
        text_lower = text.lower()
        
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return severity
        
        # 默认中等严重程度
        return "medium"
    
    def _extract_symptoms(self, text: str, extracted_info: Dict[str, Any]) -> Optional[List[str]]:
        """提取症状信息"""
        symptoms = []
        
        # 从提取的信息中获取症状
        if "symptoms" in extracted_info:
            symptoms.extend(extracted_info["symptoms"])
        
        # 从文本中识别症状关键词
        symptom_keywords = [
            "缓慢", "超时", "错误", "异常", "失败", "无响应", "延迟", "卡顿",
            "slow", "timeout", "error", "exception", "failure", "unresponsive", "lag"
        ]
        
        for keyword in symptom_keywords:
            if keyword in text.lower():
                symptoms.append(f"检测到{keyword}问题")
        
        return symptoms if symptoms else None
    
    def _extract_context(self, text: str, extracted_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取上下文信息"""
        context = {}
        
        # 从提取的信息中获取上下文
        if "context" in extracted_info:
            context.update(extracted_info["context"])
        
        # 检测环境信息
        if "生产" in text or "production" in text.lower():
            context["environment"] = "production"
        elif "测试" in text or "test" in text.lower():
            context["environment"] = "testing"
        elif "开发" in text or "development" in text.lower():
            context["environment"] = "development"
        
        # 检测时间信息
        if "持续" in text:
            context["duration_mentioned"] = True
        
        # 检测影响范围
        if "用户" in text or "user" in text.lower():
            context["user_impact"] = True
        
        return context if context else None