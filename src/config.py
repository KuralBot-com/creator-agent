import os
from dataclasses import dataclass


@dataclass
class Config:
    api_key: str
    agent_id: str
    arena_base_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    poll_interval: int
    prompt_dir: str
    llm_temperature: float
    llm_max_tokens: int
    llm_max_retries: int

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("AGENT_API_KEY", "")
        if not api_key:
            raise RuntimeError("AGENT_API_KEY environment variable is required")

        agent_id = os.environ.get("AGENT_ID", "")
        if not agent_id:
            raise RuntimeError("AGENT_ID environment variable is required")

        return cls(
            api_key=api_key,
            agent_id=agent_id,
            arena_base_url=os.environ.get("ARENA_BASE_URL", "https://api.kuralbot.com"),
            llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            llm_api_key=os.environ.get("LLM_API_KEY", "ollama"),
            llm_model=os.environ.get("LLM_MODEL", "local-model"),
            poll_interval=int(os.environ.get("POLL_INTERVAL", "30")),
            prompt_dir=os.environ.get("PROMPT_DIR", "prompts"),
            llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
            llm_max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
        )
