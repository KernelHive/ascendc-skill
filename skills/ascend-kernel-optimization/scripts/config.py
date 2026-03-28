"""
Configuration module for ascend-kernel-optimization skill.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class PathConfig:
    tiling_resource_base: str = field(
        default_factory=lambda: _get_env("TILING_RESOURCE_BASE", "")
    )
    log_base: str = field(
        default_factory=lambda: _get_env("LOG_BASE", "/root/wjh/logs/kernel_optimization")
    )
    output_base: str = field(
        default_factory=lambda: _get_env("OUTPUT_BASE", "/root/wjh/codex/output_ops")
    )
    knowledge_base_path: str = field(
        default_factory=lambda: _get_env(
            "KNOWLEDGE_BASE_PATH", 
            os.path.join(os.path.dirname(__file__), "rag_db")
        )
    )


@dataclass
class LLMConfig:
    base_url: str = _get_env("LLM_BASE_URL", "https://api.modelarts-maas.com/v1")
    api_key: str = _get_env("LLM_API_KEY", "")
    model: str = _get_env("LLM_MODEL", "DeepSeek-V3.2")


@dataclass
class OptimizerConfig:
    max_iterations: int = int(_get_env("MAX_ITERATIONS", "3"))
    max_bottlenecks_per_iter: int = 1


_path_config: Optional[PathConfig] = None
_llm_config: Optional[LLMConfig] = None
_optimizer_config: Optional[OptimizerConfig] = None


def get_path_config() -> PathConfig:
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def get_llm_config() -> LLMConfig:
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig()
    return _llm_config


def get_optimizer_config() -> OptimizerConfig:
    global _optimizer_config
    if _optimizer_config is None:
        _optimizer_config = OptimizerConfig()
    return _optimizer_config

