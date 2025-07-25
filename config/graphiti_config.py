"""
Graphiti 客户端配置文件
"""
import os
from typing import Optional
from pydantic import BaseModel, Field


class GraphitiConfig(BaseModel):
    """Graphiti 配置模型"""
    
    # Neo4j 连接配置
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j 连接 URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j 用户名")
    neo4j_password: str = Field(default="password", description="Neo4j 密码")
    neo4j_database: str = Field(default="neo4j", description="Neo4j 数据库名")
    
    # LLM 配置 (用于 Graphiti 的推理功能)
    llm_provider: str = Field(default="deepseek", description="LLM 提供商")
    llm_model_name: str = Field(default="deepseek-chat", description="LLM 模型名称")
    llm_base_url: Optional[str] = Field(default="https://api.deepseek.com/v1", description="LLM API 基础 URL")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API 密钥")
    llm_temperature: float = Field(default=0.1, description="LLM 温度参数")
    llm_max_tokens: int = Field(default=2000, description="LLM 最大令牌数")
    
    # Embedder 配置
    embedder_provider: str = Field(default="openai", description="嵌入模型提供商")
    embedder_model: str = Field(default="text-embedding-3-small", description="嵌入模型名称")
    embedder_api_key: Optional[str] = Field(default=None, description="嵌入模型 API 密钥")
    
    # Graphiti 运行配置
    max_search_results: int = Field(default=10, description="最大搜索结果数")
    search_timeout: int = Field(default=30, description="搜索超时时间(秒)")
    enable_logging: bool = Field(default=True, description="是否启用日志")
    log_level: str = Field(default="INFO", description="日志级别")
    
    @classmethod
    def from_env(cls) -> "GraphitiConfig":
        """从环境变量创建配置"""
        return cls(
            # Neo4j 配置
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            
            # LLM 配置
            llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
            llm_model_name=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            
            # Embedder 配置
            embedder_provider=os.getenv("EMBEDDER_PROVIDER", "openai"),
            embedder_model=os.getenv("EMBEDDER_MODEL", "text-embedding-3-small"),
            embedder_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            
            # 运行配置
            max_search_results=int(os.getenv("GRAPHITI_MAX_SEARCH_RESULTS", "10")),
            search_timeout=int(os.getenv("GRAPHITI_SEARCH_TIMEOUT", "30")),
            enable_logging=os.getenv("GRAPHITI_ENABLE_LOGGING", "true").lower() == "true",
            log_level=os.getenv("GRAPHITI_LOG_LEVEL", "INFO")
        )


# 默认配置实例
default_config = GraphitiConfig.from_env()