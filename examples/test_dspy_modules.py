#!/usr/bin/env python3
"""
测试 DSPy 模块功能（基础功能测试，不调用 LLM）
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dspy_modules.smart_router import SmartIntelligentRouter
from dspy_modules.memory_retrieval import IntelligentMemoryAnalyzer, IntelligentQueryGenerator, MemoryQualityEvaluator
from dspy_modules.intelligent_info_collector import IntelligentInfoCollector
from dspy_modules.alert_analyzer import AlertInfo


def test_dspy_modules_basic():
    """测试 DSPy 模块基础功能"""
    
    print("🧪 开始测试 DSPy 模块基础功能...")
    
    try:
        # 准备测试数据
        test_alert = AlertInfo(
            alert_id="TEST002",
            timestamp="2025-01-25T11:00:00Z",
            severity="critical",
            source="database",
            message="数据库连接池耗尽",
            metrics={"connection_count": 100, "response_time": 15.0},
            tags=["database", "performance"]
        )
        
        test_state = {
            "alert_info": test_alert,
            "symptoms": ["查询超时", "连接拒绝"],
            "analysis_result": {"category": "database", "priority": "critical"},
            "diagnostic_result": {
                "root_cause": "连接池配置不当",
                "confidence_score": 0.3,  # 低置信度
                "potential_causes": ["连接泄漏", "并发过高", "配置错误"]
            },
            "context": {"environment": "production", "database_type": "mysql"},
            "memory_queries": None,
            "memory_context": None,
            "info_collection_queries": None,
            "info_collection_context": None,
            "errors": ["连接超时"]
        }
        
        # 1. 测试智能路由器
        print("\n🧠 测试智能路由器...")
        smart_router = SmartIntelligentRouter()
        
        # 测试状态提取
        business_data = smart_router._extract_business_data(test_state)
        print(f"✅ 业务数据提取: {business_data[:80]}...")
        
        completion_status = smart_router._check_completion_status(test_state)
        print(f"✅ 完成状态: {completion_status}")
        
        # 测试回退决策
        fallback = smart_router._create_fallback_decision(test_state, "测试")
        print(f"✅ 回退决策: {fallback.next_node} (置信度: {fallback.confidence})")
        
        # 2. 测试记忆分析器
        print("\n🧠 测试记忆分析器...")
        memory_analyzer = IntelligentMemoryAnalyzer()
        
        # 测试上下文构建
        context_summary = memory_analyzer._build_context_summary(test_state)
        print(f"✅ 上下文摘要: {context_summary[:80]}...")
        
        data_quality = memory_analyzer._assess_data_quality(test_state)
        print(f"✅ 数据质量评估: {data_quality}")
        
        # 测试回退分析
        fallback_analysis = memory_analyzer._create_fallback_analysis(test_state, "diagnosis", "测试错误")
        print(f"✅ 回退分析: {fallback_analysis.necessity} - {fallback_analysis.focus}")
        
        # 3. 测试查询生成器
        print("\n🔍 测试查询生成器...")
        query_generator = IntelligentQueryGenerator()
        
        # 测试关键词提取
        keywords = query_generator._extract_business_keywords(test_state)
        print(f"✅ 业务关键词: {keywords}")
        
        # 测试查询解析
        test_queries = "数据库故障案例, 连接池优化, MySQL性能调优"
        parsed = query_generator._parse_queries(test_queries)
        print(f"✅ 查询解析: {parsed}")
        
        # 测试回退查询
        fallback_queries = query_generator._generate_fallback_queries(
            fallback_analysis, test_state, "测试错误"
        )
        print(f"✅ 回退查询: {fallback_queries.primary_queries}")
        
        # 4. 测试质量评估器
        print("\n📊 测试质量评估器...")
        quality_evaluator = MemoryQualityEvaluator()
        
        # 测试结果摘要
        mock_results = [
            {"content": "数据库告警处理经验：连接池配置应设置为100"},
            {"content": "MySQL故障诊断：检查慢查询日志"},
            {"content": "性能优化解决方案：增加连接数限制"}
        ]
        
        results_summary = quality_evaluator._summarize_results(mock_results)
        print(f"✅ 结果摘要: {results_summary}")
        
        business_relevance = quality_evaluator._assess_business_relevance(test_state)
        print(f"✅ 业务相关性: {business_relevance}")
        
        # 测试回退评估
        fallback_eval = quality_evaluator._create_fallback_evaluation(mock_results, "测试错误")
        print(f"✅ 回退评估: 质量{fallback_eval.quality_score} - {fallback_eval.relevance}")
        
        # 5. 测试信息收集器
        print("\n📝 测试信息收集器...")
        info_collector = IntelligentInfoCollector()
        
        # 测试缺失信息识别
        incomplete_state = {**test_state}
        del incomplete_state["symptoms"]  # 移除症状信息
        
        missing_info = info_collector._identify_missing_info(incomplete_state)
        print(f"✅ 缺失信息识别: {missing_info}")
        
        # 测试状态摘要构建
        state_summary = info_collector._build_state_summary(test_state)
        print(f"✅ 状态摘要: {state_summary[:80]}...")
        
        # 测试业务阶段确定
        stage = info_collector._determine_business_stage(test_state)
        print(f"✅ 业务阶段: {stage}")
        
        # 测试提取上下文构建
        extraction_context = info_collector._build_extraction_context("symptoms")
        print(f"✅ 提取上下文: {extraction_context[:80]}...")
        
        # 测试回退分析
        fallback_info = info_collector._create_fallback_analysis(test_state, "测试错误")
        print(f"✅ 信息收集回退: {fallback_info['missing_type']} - {fallback_info['priority']}")
        
        print(f"\n🎉 DSPy 模块基础功能测试全部通过!")
        print(f"   - SmartIntelligentRouter: ✅")
        print(f"   - IntelligentMemoryAnalyzer: ✅")
        print(f"   - IntelligentQueryGenerator: ✅")
        print(f"   - MemoryQualityEvaluator: ✅")
        print(f"   - IntelligentInfoCollector: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ DSPy 模块基础功能测试失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        return False


def main():
    """主函数"""
    success = test_dspy_modules_basic()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()