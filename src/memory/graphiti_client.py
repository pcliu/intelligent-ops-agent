"""
Graphiti 客户端封装

提供对 Graphiti 核心功能的封装和扩展
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from config.graphiti_config import GraphitiConfig
from .episode_manager import OpsEpisode


logger = logging.getLogger(__name__)


class GraphitiClientWrapper:
    """Graphiti 客户端封装类"""
    
    def __init__(self, config: Optional[GraphitiConfig] = None):
        """
        初始化 Graphiti 客户端
        
        Args:
            config: Graphiti 配置，如果为 None 则从环境变量加载
        """
        self.config = config or GraphitiConfig.from_env()
        self.client: Optional[Graphiti] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化 Graphiti 客户端"""
        if self._initialized:
            return
            
        try:
            # 创建 Graphiti 实例（使用默认的 OpenAI 客户端）
            self.client = Graphiti(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password
            )
            
            # 构建索引和约束
            await self.client.build_indices_and_constraints()
            
            self._initialized = True
            logger.info("Graphiti 客户端初始化成功")
            
        except Exception as e:
            logger.error(f"Graphiti 客户端初始化失败: {e}")
            raise
    
    async def add_episode(self, episode: OpsEpisode) -> str:
        """
        添加运维情节到知识图谱
        
        Args:
            episode: 运维情节对象
            
        Returns:
            str: 情节的 UUID
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.client.add_episode(
                name=episode.name,
                episode_body=episode.episode_body,
                source_description=episode.source_description,
                reference_time=episode.reference_time,
                source=EpisodeType.text,  # 统一使用 text 类型
                # 可以通过 group_id 来分组不同类型的情节
                group_id=episode.episode_type.value
            )
            
            logger.info(f"成功添加情节: {episode.name}")
            return result.episode_uuid if hasattr(result, 'episode_uuid') else "unknown"
            
        except Exception as e:
            logger.error(f"添加情节失败 {episode.name}: {e}")
            raise
    
    async def batch_add_episodes(self, episodes: List[OpsEpisode]) -> List[str]:
        """
        批量添加情节
        
        Args:
            episodes: 情节列表
            
        Returns:
            List[str]: 情节 UUID 列表
        """
        if not episodes:
            return []
        
        uuids = []
        for episode in episodes:
            try:
                uuid = await self.add_episode(episode)
                uuids.append(uuid)
            except Exception as e:
                logger.error(f"批量添加情节失败 {episode.name}: {e}")
                uuids.append("")  # 保持索引一致性
        
        return uuids
    
    async def search(self, query: str, num_results: int = 10, 
                    center_node_uuid: Optional[str] = None,
                    group_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        执行混合搜索
        
        Args:
            query: 搜索查询
            num_results: 结果数量限制
            center_node_uuid: 中心节点 UUID（用于重新排序）
            group_ids: 图分区 ID 列表
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            search_results = await self.client.search(
                query=query,
                num_results=num_results,
                center_node_uuid=center_node_uuid,
                group_ids=group_ids
            )
            
            # 转换结果为更友好的格式
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "content": getattr(result, 'fact', ''),
                    "source_node": getattr(result, 'source_node_name', ''),
                    "target_node": getattr(result, 'target_node_name', ''),
                    "edge_type": getattr(result, 'fact_type', ''),
                    "score": getattr(result, 'score', 0.0),
                    "metadata": {
                        "source_uuid": getattr(result, 'source_node_uuid', ''),
                        "target_uuid": getattr(result, 'target_node_uuid', ''),
                        "edge_uuid": getattr(result, 'uuid', '')
                    }
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"搜索完成，找到 {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    async def get_node_neighbors(self, node_uuid: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        获取节点的邻居（用于关系分析）
        
        Args:
            node_uuid: 节点 UUID
            max_depth: 最大深度
            
        Returns:
            List[Dict[str, Any]]: 邻居节点列表
        """
        # 这个功能需要根据 Graphiti 的具体 API 实现
        # 暂时返回空列表，待 Graphiti 提供相应接口后补充
        logger.warning("get_node_neighbors 功能待实现")
        return []
    
    async def close(self) -> None:
        """关闭客户端连接"""
        if self.client and hasattr(self.client, 'close'):
            await self.client.close()
        self._initialized = False
        logger.info("Graphiti 客户端已关闭")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._initialized and self.client is not None
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # 执行一个简单的搜索来测试连接
            await self.client.search(query="test", num_results=1)
            return True
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False