"""
Memory Tools for DSPy ReAct Integration

提供记忆相关的工具函数，供 DSPy ReAct 模块使用
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

logger = logging.getLogger(__name__)

# 全局记忆工具实例
_memory_tool_instance: Optional['MemoryTool'] = None


class MemoryTool:
    """记忆工具类 - 封装 Graphiti 功能"""
    
    def __init__(self):
        """初始化记忆工具"""
        self.graphiti: Optional[Graphiti] = None
        self._initialized = False
        
        # 从环境变量读取配置
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
        
    async def initialize(self) -> None:
        """初始化 Graphiti 客户端"""
        if self._initialized:
            return
            
        try:
            # 创建 Graphiti 实例 - 使用默认的 OpenAI 配置
            # Graphiti 会自动从环境变量 OPENAI_API_KEY 读取配置
            self.graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user, 
                password=self.neo4j_password
                # llm_client 和 embedder 参数留空，让 Graphiti 使用默认配置
            )
            
            # 构建索引和约束
            await self.graphiti.build_indices_and_constraints()
            
            self._initialized = True
            logger.info("Memory tool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory tool: {e}")
            raise
    
    async def search_memory(self, query: str, case_type: str = "general", 
                           limit: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆
        
        Args:
            query: 搜索查询
            case_type: 案例类型
            limit: 结果数量限制
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # 构建搜索查询
            enhanced_query = f"{case_type} {query}" if case_type != "general" else query
            
            # 执行搜索
            results = await self.graphiti.search(
                query=enhanced_query,
                num_results=limit
            )
            
            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_result = {
                    "content": getattr(result, 'fact', ''),
                    "score": getattr(result, 'score', 0.0),
                    "source": getattr(result, 'source_node_name', ''),
                    "target": getattr(result, 'target_node_name', ''),
                    "edge_type": getattr(result, 'fact_type', ''),
                    "created_at": getattr(result, 'created_at', ''),
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Memory search completed: {len(formatted_results)} results for '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    async def add_episode(self, name: str, content: str, episode_type: str = "general", 
                         source_description: str = "智能运维系统") -> str:
        """添加记忆情节
        
        Args:
            name: 情节名称
            content: 情节内容
            episode_type: 情节类型
            source_description: 来源描述
            
        Returns:
            str: 情节UUID
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.graphiti.add_episode(
                name=name,
                episode_body=content,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=datetime.now(),
                group_id=episode_type
            )
            
            episode_uuid = getattr(result, 'episode_uuid', 'unknown')
            logger.info(f"Added episode: {name} (UUID: {episode_uuid})")
            return episode_uuid
            
        except Exception as e:
            logger.error(f"Failed to add episode {name}: {e}")
            return ""
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # 执行简单搜索测试连接
            await self.graphiti.search(query="test", num_results=1)
            return True
            
        except Exception as e:
            logger.error(f"Memory tool health check failed: {e}")
            return False


def get_memory_tool() -> MemoryTool:
    """获取全局记忆工具实例"""
    global _memory_tool_instance
    
    if _memory_tool_instance is None:
        _memory_tool_instance = MemoryTool()
    
    return _memory_tool_instance


# ==================== DSPy Tool Functions ====================

def _run_async_in_sync(coro):
    """在同步函数中运行异步代码的辅助函数"""
    import asyncio
    import threading
    
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_running_loop()
        # 如果在异步上下文中，在新线程中运行
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                result = asyncio.run(coro)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
        
    except RuntimeError:
        # 没有运行中的事件循环，直接运行
        return asyncio.run(coro)


def search_historical_cases(query: str, case_type: str = "diagnosis") -> str:
    """搜索历史案例 - DSPy Tool 函数
    
    Args:
        query: 搜索查询，描述要查找的问题或症状
        case_type: 案例类型，如 diagnosis, solution, alert 等
        
    Returns:
        str: 格式化的历史案例信息
    """
    memory_tool = get_memory_tool()
    
    try:
        results = _run_async_in_sync(memory_tool.search_memory(query, case_type, 3))
        
        if not results:
            return f"没有相关历史信息"
        
        # 格式化返回结果
        formatted_cases = []
        for i, case in enumerate(results, 1):
            content = case.get('content', '').strip()
            score = case.get('score', 0.0)
            formatted_cases.append(f"案例{i} (相似度{score:.2f}): {content}")
        
        return f"找到{len(results)}个相关案例:\n" + "\n".join(formatted_cases)
        
    except Exception as e:
        logger.error(f"Search historical cases failed: {e}")
        # 返回模拟结果，避免工具调用失败
        return f"历史案例搜索暂时不可用，建议基于以下常见模式分析: 1. 检查系统资源使用情况 2. 查看近期配置变更 3. 分析错误日志模式"


def search_solution_patterns(problem_description: str) -> str:
    """搜索解决方案模式 - DSPy Tool 函数
    
    Args:
        problem_description: 问题描述
        
    Returns:
        str: 相关的解决方案和最佳实践
    """
    memory_tool = get_memory_tool()
    
    try:
        results = _run_async_in_sync(memory_tool.search_memory(
            f"解决方案 {problem_description}",
            "solution",
            2
        ))
        
        if not results:
            return f"没有相关历史信息"
        
        solutions = []
        for i, solution in enumerate(results, 1):
            content = solution.get('content', '').strip()
            score = solution.get('score', 0.0)
            solutions.append(f"方案{i} (相似度{score:.2f}): {content}")
        
        return f"找到{len(results)}个解决方案:\n" + "\n".join(solutions)
        
    except Exception as e:
        logger.error(f"Search solution patterns failed: {e}")
        # 返回模拟结果
        return f"解决方案搜索暂时不可用，建议尝试以下通用解决方案: 1. 重启相关服务 2. 检查配置文件 3. 查看系统日志 4. 联系技术支持"


def search_alert_patterns(alert_info: str) -> str:
    """搜索告警模式 - DSPy Tool 函数
    
    Args:
        alert_info: 告警信息描述
        
    Returns:
        str: 相关的告警模式和关联性分析
    """
    memory_tool = get_memory_tool()
    
    try:
        results = _run_async_in_sync(memory_tool.search_memory(
            f"告警 {alert_info}",
            "alert",
            3
        ))
        
        if not results:
            return f"没有相关历史信息"
        
        patterns = []
        for i, pattern in enumerate(results, 1):
            content = pattern.get('content', '').strip()
            score = pattern.get('score', 0.0)
            patterns.append(f"模式{i} (相似度{score:.2f}): {content}")
        
        return f"找到{len(results)}个相关告警模式:\n" + "\n".join(patterns)
        
    except Exception as e:
        logger.error(f"Search alert patterns failed: {e}")
        return f"搜索告警模式时出错: {str(e)}"


def search_action_history(action_context: str) -> str:
    """搜索行动历史 - DSPy Tool 函数
    
    Args:
        action_context: 行动上下文描述
        
    Returns:
        str: 相关的历史执行经验和建议
    """
    memory_tool = get_memory_tool()
    
    try:
        results = _run_async_in_sync(memory_tool.search_memory(
            f"执行 行动 {action_context}",
            "action",
            2
        ))
        
        if not results:
            return f"没有相关历史信息"
        
        experiences = []
        for i, exp in enumerate(results, 1):
            content = exp.get('content', '').strip()
            score = exp.get('score', 0.0)
            experiences.append(f"经验{i} (相似度{score:.2f}): {content}")
        
        return f"找到{len(results)}个相关执行经验:\n" + "\n".join(experiences)
        
    except Exception as e:
        logger.error(f"Search action history failed: {e}")
        return f"搜索执行经验时出错: {str(e)}"