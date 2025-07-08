"""
LLM 配置模块
支持 DeepSeek、OpenAI 等多种语言模型
"""

import os
import dspy
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 配置类"""
    provider: str = "deepseek"  # deepseek, openai, ollama
    model_name: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 60


class DeepSeekLLM(dspy.LM):
    """DeepSeek LLM 适配器，用于 DSPy"""
    
    def __init__(self, 
                 model: str = "deepseek-chat",
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com/v1",
                 **kwargs):
        
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.kwargs = kwargs
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable.")
        
        # 初始化 OpenAI 兼容的客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Please install openai package: uv add openai")
        
        super().__init__(model=model)
    
    def basic_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """基础请求方法"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 2000),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            return {
                "choices": [{
                    "text": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            return {
                "choices": [{"text": f"Error: {str(e)}", "finish_reason": "error"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """调用方法"""
        response = self.basic_request(prompt, **kwargs)
        return response["choices"][0]["text"]


def setup_deepseek_llm(config: Optional[LLMConfig] = None) -> tuple:
    """
    设置 DeepSeek LLM 用于 DSPy 和 LangChain
    
    Args:
        config: LLM 配置
        
    Returns:
        tuple: (dspy_lm, langchain_llm)
    """
    if config is None:
        config = LLMConfig()
    
    # 设置环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
    
    # DSPy LLM 配置
    if config.provider.lower() == "deepseek":
        dspy_lm = DeepSeekLLM(
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url or "https://api.deepseek.com/v1",
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # LangChain LLM 配置 (使用 OpenAI 兼容接口)
        langchain_llm = ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.api_key or os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=config.base_url or "https://api.deepseek.com/v1",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
    
    elif config.provider.lower() == "openai":
        # OpenAI 配置
        dspy_lm = dspy.OpenAI(
            model=config.model_name,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        langchain_llm = ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    elif config.provider.lower() == "ollama":
        # Ollama 本地配置
        dspy_lm = dspy.OllamaLocal(
            model=config.model_name,
            base_url=config.base_url or "http://localhost:11434"
        )
        
        langchain_llm = Ollama(
            model=config.model_name,
            base_url=config.base_url or "http://localhost:11434"
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    # 设置 DSPy 默认 LM
    dspy.settings.configure(lm=dspy_lm)
    
    return dspy_lm, langchain_llm


def get_llm_config_from_env() -> LLMConfig:
    """从环境变量获取 LLM 配置"""
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "deepseek"),
        model_name=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
        timeout=int(os.getenv("LLM_TIMEOUT", "60"))
    )


def validate_llm_config(config: LLMConfig) -> bool:
    """验证 LLM 配置"""
    if config.provider.lower() == "deepseek":
        if not config.api_key and not os.getenv("DEEPSEEK_API_KEY"):
            print("❌ DeepSeek API key is required")
            return False
    elif config.provider.lower() == "openai":
        if not config.api_key and not os.getenv("OPENAI_API_KEY"):
            print("❌ OpenAI API key is required")
            return False
    
    if config.temperature < 0 or config.temperature > 2:
        print("❌ Temperature should be between 0 and 2")
        return False
    
    if config.max_tokens <= 0:
        print("❌ Max tokens should be positive")
        return False
    
    return True


def test_llm_connection(config: Optional[LLMConfig] = None) -> bool:
    """测试 LLM 连接"""
    try:
        if config is None:
            config = get_llm_config_from_env()
        
        print(f"🔄 Testing {config.provider} connection...")
        
        dspy_lm, langchain_llm = setup_deepseek_llm(config)
        
        # 测试 DSPy
        test_prompt = "Hello, please respond with 'Connection successful'"
        response = dspy_lm(test_prompt)
        
        print(f"✅ {config.provider} connection successful")
        print(f"   Model: {config.model_name}")
        print(f"   Response: {response[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ {config.provider if config else 'LLM'} connection failed: {str(e)}")
        return False


# 预设配置
DEEPSEEK_CONFIGS = {
    "deepseek-chat": LLMConfig(
        provider="deepseek",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=2000
    ),
    "deepseek-coder": LLMConfig(
        provider="deepseek",
        model_name="deepseek-coder",
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=4000
    )
}

OPENAI_CONFIGS = {
    "gpt-4": LLMConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=2000
    ),
    "gpt-3.5-turbo": LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=2000
    )
}