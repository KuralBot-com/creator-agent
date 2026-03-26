"""Tests for Config dataclass and env var parsing."""

import pytest

from src.config import Config


class TestConfigFromEnv:
    def test_required_fields(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "test-key-123")
        monkeypatch.setenv("AGENT_ID", "agent-uuid-456")

        config = Config.from_env()

        assert config.api_key == "test-key-123"
        assert config.agent_id == "agent-uuid-456"

    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("AGENT_ID", "a")

        config = Config.from_env()

        assert config.arena_base_url == "https://api.kuralbot.com"
        assert config.llm_base_url == "http://localhost:11434/v1"
        assert config.llm_api_key == "ollama"
        assert config.llm_model == "local-model"
        assert config.poll_interval == 30
        assert config.prompt_dir == "prompts"
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 256
        assert config.llm_max_retries == 3

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("AGENT_ID", "a")
        monkeypatch.setenv("ARENA_BASE_URL", "http://localhost:3000")
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:5000/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
        monkeypatch.setenv("LLM_MODEL", "my-model")
        monkeypatch.setenv("POLL_INTERVAL", "60")
        monkeypatch.setenv("PROMPT_DIR", "/tmp/prompts")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLM_MAX_TOKENS", "512")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")

        config = Config.from_env()

        assert config.arena_base_url == "http://localhost:3000"
        assert config.llm_base_url == "http://localhost:5000/v1"
        assert config.llm_api_key == "sk-test-key"
        assert config.llm_model == "my-model"
        assert config.poll_interval == 60
        assert config.prompt_dir == "/tmp/prompts"
        assert config.llm_temperature == 0.5
        assert config.llm_max_tokens == 512
        assert config.llm_max_retries == 5

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        monkeypatch.setenv("AGENT_ID", "a")

        with pytest.raises(RuntimeError, match="AGENT_API_KEY"):
            Config.from_env()

    def test_missing_agent_id_raises(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.delenv("AGENT_ID", raising=False)

        with pytest.raises(RuntimeError, match="AGENT_ID"):
            Config.from_env()

    def test_empty_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "")
        monkeypatch.setenv("AGENT_ID", "a")

        with pytest.raises(RuntimeError, match="AGENT_API_KEY"):
            Config.from_env()

    def test_empty_agent_id_raises(self, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("AGENT_ID", "")

        with pytest.raises(RuntimeError, match="AGENT_ID"):
            Config.from_env()
