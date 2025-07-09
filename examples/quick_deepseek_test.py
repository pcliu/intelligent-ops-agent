#!/usr/bin/env python3
"""
快速 DeepSeek 模型验证
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.utils.llm_config import test_llm_connection, get_llm_config_from_env, setup_deepseek_llm

def quick_test():
    """快速测试"""
    print("🚀 快速 DeepSeek 模型验证")
    print("=" * 40)
    
    # 1. 检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...")
    
    # 2. 测试连接
    print("\n🔗 测试连接...")
    if not test_llm_connection():
        print("❌ 连接失败")
        return False
    
    # 3. 测试 DSPy 模块
    print("\n🧠 测试 DSPy 告警分析...")
    try:
        from src.dspy_modules.alert_analyzer import AlertAnalyzer, AlertInfo
        from src.utils.llm_config import setup_deepseek_llm, get_llm_config_from_env
        
        # 设置 LLM
        config = get_llm_config_from_env()
        dspy_lm, langchain_llm = setup_deepseek_llm(config)
        print("✅ LLM 配置完成")
        
        # 创建告警分析器
        analyzer = AlertAnalyzer()
        
        # 测试告警
        test_alert = AlertInfo(
            alert_id="quick_test_001",
            timestamp="2024-01-09T10:30:00Z",
            severity="high",
            source="test_monitor",
            message="CPU 使用率超过 85%",
            metrics={"cpu_usage": 0.87},
            tags=["cpu", "performance"]
        )
        
        print("🔄 分析告警...")
        result = analyzer.forward(alert_info=test_alert, historical_alerts=[])
        
        print("✅ 分析完成！")
        print(f"   分类: {result.category}")
        print(f"   优先级: {result.priority}")
        print(f"   紧急程度: {result.urgency_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ DSPy 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 DeepSeek 模型验证成功！")
        print("💡 可以在 LangGraph Studio 中使用工作流了")
    else:
        print("\n❌ DeepSeek 模型验证失败")
        print("💡 请检查 API Key 和网络连接")