"""Tests for PromptLoader."""

import os
import pytest

from src.prompt_loader import PromptLoader


@pytest.fixture
def prompt_dir(tmp_path):
    """Create a temp prompt directory with valid system.txt and user.txt."""
    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("You are a Tamil poet.", encoding="utf-8")
    user_file.write_text("Write a kural about: {prompt}", encoding="utf-8")
    return str(tmp_path)


class TestPromptLoader:
    def test_loads_prompts(self, prompt_dir):
        loader = PromptLoader(prompt_dir)

        assert loader.get_system_prompt() == "You are a Tamil poet."

    def test_user_prompt_formatting(self, prompt_dir):
        loader = PromptLoader(prompt_dir)

        result = loader.get_user_prompt("அறம்")
        assert result == "Write a kural about: அறம்"

    def test_strips_whitespace(self, tmp_path):
        (tmp_path / "system.txt").write_text("  system content  \n\n", encoding="utf-8")
        (tmp_path / "user.txt").write_text("  {prompt} topic  \n", encoding="utf-8")
        loader = PromptLoader(str(tmp_path))

        assert loader.get_system_prompt() == "system content"
        assert loader.get_user_prompt("test") == "test topic"

    def test_missing_system_txt_raises(self, tmp_path):
        (tmp_path / "user.txt").write_text("{prompt}", encoding="utf-8")

        with pytest.raises(RuntimeError, match="System prompt not found"):
            PromptLoader(str(tmp_path))

    def test_missing_user_txt_raises(self, tmp_path):
        (tmp_path / "system.txt").write_text("system", encoding="utf-8")

        with pytest.raises(RuntimeError, match="User prompt template not found"):
            PromptLoader(str(tmp_path))

    def test_empty_system_txt_raises(self, tmp_path):
        (tmp_path / "system.txt").write_text("", encoding="utf-8")
        (tmp_path / "user.txt").write_text("{prompt}", encoding="utf-8")

        with pytest.raises(RuntimeError, match="System prompt is empty"):
            PromptLoader(str(tmp_path))

    def test_empty_user_txt_raises(self, tmp_path):
        (tmp_path / "system.txt").write_text("system", encoding="utf-8")
        (tmp_path / "user.txt").write_text("   \n  ", encoding="utf-8")

        with pytest.raises(RuntimeError, match="User prompt template is empty"):
            PromptLoader(str(tmp_path))

    def test_loads_real_prompts(self):
        """Smoke test: load actual prompt files from the repo."""
        prompts_path = os.path.join(os.path.dirname(__file__), "..", "prompts")
        if not os.path.isdir(prompts_path):
            pytest.skip("prompts directory not found")

        loader = PromptLoader(prompts_path)
        assert len(loader.get_system_prompt()) > 0
        result = loader.get_user_prompt("அன்பு")
        assert "அன்பு" in result
