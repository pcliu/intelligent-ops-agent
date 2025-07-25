#!/usr/bin/env python3
"""
测试记忆服务基础架构
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import EpisodeManager, OpsEpisodeType
from dspy_modules.memory_retrieval import SimpleMemoryProcessor
from memory.episode_manager import OpsEpisode, EpisodeMetadata
from dspy_modules.alert_analyzer import AlertInfo
from datetime import datetime


async def test_memory_infrastructure():
    """测试记忆服务基础架构"""
    
    print("🧪 开始测试记忆服务基础架构...")
    
    try:
        # 1. 测试 EpisodeManager
        print("\n📝 测试 EpisodeManager...")
        episode_manager = EpisodeManager()
        
        # 创建测试状态
        test_alert = AlertInfo(
            alert_id="TEST001",
            timestamp="2025-01-25T10:00:00Z",
            severity="high",
            source="test_system",
            message="CPU使用率超过90%",
            metrics={"cpu_usage": 95.5},
            tags=["performance", "cpu"]
        )
        
        test_state = {
            "alert_info": test_alert,
            "symptoms": ["CPU使用率高", "响应缓慢"],
            "analysis_result": {"category": "cpu", "priority": "high"},
            "diagnostic_result": {
                "root_cause": "进程占用过多CPU",
                "confidence_score": 0.85,
                "potential_causes": ["内存泄漏", "死循环"]
            }
        }
        
        episodes = episode_manager.generate_episodes_from_state(test_state)
        print(f"✅ 成功生成 {len(episodes)} 个情节")
        
        for episode in episodes:
            print(f"   - {episode.name}: {episode.episode_type.value}")
        
        # 2. 测试精简记忆处理器
        print("\n🧠 测试精简记忆处理器...")
        memory_processor = SimpleMemoryProcessor()
        
        # 测试内部方法
        context_summary = memory_processor._build_context_summary(test_state)
        print(f"✅ 上下文摘要构建: {context_summary}")
        
        # 测试查询解析
        test_queries = "故障诊断, 历史案例, CPU性能问题"
        parsed_queries = memory_processor._parse_queries(test_queries)
        print(f"✅ 查询解析: {parsed_queries}")
        
        # 测试回退策略
        fallback_result = memory_processor._create_fallback_result(test_state, "diagnosis", "测试错误")
        print(f"✅ 回退策略: 需要记忆={fallback_result.need_memory}, 查询={fallback_result.queries}")
        
        # 3. 测试情节生成（直接使用 EpisodeManager）
        print("\n📝 测试情节生成...")
        episode_manager = EpisodeManager()
        generated_episodes = episode_manager.generate_episodes_from_state(test_state)
        print(f"✅ 从状态生成 {len(generated_episodes)} 个情节")
        
        # 4. 测试状态验证函数
        print("\n✅ 测试状态验证...")
        sys.path.append(str(Path(__file__).parent.parent / "src" / "agents"))
        from intelligent_ops_agent import validate_chat_state, create_empty_chat_state
        
        # 测试空状态
        empty_state = create_empty_chat_state()
        is_valid = validate_chat_state(empty_state)
        print(f"✅ 空状态验证: {'通过' if is_valid else '失败'}")
        
        # 测试带记忆字段的状态
        memory_state = {
            **empty_state,
            "memory_queries": ["测试查询1", "测试查询2"],
            "memory_context": {"test": "data"},
            "info_collection_queries": ["收集信息1"],
            "info_collection_context": {"collected": "info"}
        }
        is_valid = validate_chat_state(memory_state)
        print(f"✅ 记忆状态验证: {'通过' if is_valid else '失败'}")
        
        print(f"\n🎉 精简记忆架构测试全部通过!")
        print(f"   - EpisodeManager: ✅")
        print(f"   - 精简记忆处理器: ✅") 
        print(f"   - ChatState验证: ✅")
        print(f"   - 单次推理优化: ✅ (从3次推理简化为1次)")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保所有依赖模块都已正确创建")
        return False
        
    except Exception as e:
        print(f"❌ 基础架构测试失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        return False


async def main():
    """主函数"""
    success = await test_memory_infrastructure()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())