#!/usr/bin/env python3
"""
测试 Graphiti 连接和基础功能
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphiti_core import Graphiti


async def test_graphiti_connection():
    """测试 Graphiti 连接和基础功能"""
    
    print("🧪 开始测试 Graphiti 连接...")
    
    try:
        # 1. 检查环境变量
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        print(f"✅ 配置检查:")
        print(f"   Neo4j URI: {neo4j_uri}")
        print(f"   Neo4j User: {neo4j_user}")
        print(f"   OpenAI API Key: {'已设置' if openai_api_key else '未设置'}")
        
        if not openai_api_key:
            print("❌ 请在 .env 文件中设置 OPENAI_API_KEY")
            return False
        
        # 2. 创建 Graphiti 实例（使用默认的 OpenAI 客户端）
        graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
        print("✅ Graphiti 实例创建成功")
        
        # 3. 测试数据库连接
        print("\n🔗 测试 Neo4j 数据库连接...")
        await graphiti.build_indices_and_constraints()
        print("✅ Neo4j 连接成功，索引和约束构建完成")
        
        # 4. 测试基础功能 - 添加测试情节
        print("\n📝 测试添加情节功能...")
        from graphiti_core.nodes import EpisodeType
        from datetime import datetime
        
        await graphiti.add_episode(
            name="Test Episode",
            episode_body="这是一个测试情节，用于验证 Graphiti 的基础功能。服务器 CPU 使用率达到 95%，需要立即处理。",
            source_description="Graphiti 连接测试",
            reference_time=datetime.now(),
            source=EpisodeType.text
        )
        print("✅ 测试情节添加成功")
        
        # 5. 测试搜索功能
        print("\n🔍 测试搜索功能...")
        search_results = await graphiti.search(
            query="CPU 使用率",
            num_results=5
        )
        
        print(f"✅ 搜索完成，找到 {len(search_results)} 个结果:")
        for i, result in enumerate(search_results[:3], 1):
            # EntityEdge 对象的属性可能不同，我们尝试打印其基本信息
            print(f"   {i}. 边类型: {type(result).__name__}")
        
        # 6. 测试完成
        print(f"\n🎉 Graphiti 连接测试全部通过!")
        print(f"   - Neo4j 连接: ✅")
        print(f"   - LLM 推理: ✅") 
        print(f"   - 嵌入模型: ✅")
        print(f"   - 情节存储: ✅")
        print(f"   - 混合搜索: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Graphiti 连接测试失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        
        # 提供故障排除建议
        print(f"\n🔧 故障排除建议:")
        print(f"   1. 确保 Neo4j Desktop 正在运行")
        print(f"   2. 检查 Neo4j 连接参数 (URI, 用户名, 密码)")
        print(f"   3. 验证 API 密钥是否正确设置")
        print(f"   4. 检查网络连接")
        
        return False


async def main():
    """主函数"""
    success = await test_graphiti_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())