"""Tests for LLMClient — output cleaning, structural validation, and generation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm_client import LLMClient


@pytest.fixture
def llm():
    return LLMClient(
        base_url="http://localhost:1234/v1",
        model="test-model",
        temperature=0.7,
        max_tokens=256,
        max_retries=3,
        api_key="test-key",
    )


class TestCleanOutput:
    def test_plain_two_lines(self, llm):
        text = "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
        assert llm._clean_output(text) == text

    def test_strips_markdown_code_fences(self, llm):
        text = "```\nஅகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு\n```"
        assert llm._clean_output(text) == "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"

    def test_strips_language_code_fences(self, llm):
        text = "```tamil\nஅகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு\n```"
        assert llm._clean_output(text) == "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"

    def test_strips_surrounding_quotes(self, llm):
        text = '"அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"'
        result = llm._clean_output(text)
        assert result == "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"

    def test_strips_tamil_preamble_kural(self, llm):
        text = "குறள்: அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
        result = llm._clean_output(text)
        assert result == "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"

    def test_strips_english_preamble(self, llm):
        text = "Here is your kural:\nஅகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
        result = llm._clean_output(text)
        assert result == "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"

    def test_strips_venba_preamble(self, llm):
        text = "வெண்பா: அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
        result = llm._clean_output(text)
        assert result == "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"

    def test_keeps_only_first_two_content_lines(self, llm):
        text = "line1 w2 w3 w4\nline2 w2 w3\nextra line\nanother extra"
        result = llm._clean_output(text)
        assert result == "line1 w2 w3 w4\nline2 w2 w3"

    def test_skips_blank_lines(self, llm):
        text = "\n\nline1 w2 w3 w4\n\nline2 w2 w3\n\n"
        result = llm._clean_output(text)
        assert result == "line1 w2 w3 w4\nline2 w2 w3"

    def test_returns_none_for_empty(self, llm):
        assert llm._clean_output("") is None
        assert llm._clean_output("   \n  ") is None

    def test_single_line_kept(self, llm):
        text = "ஒரே வரி மட்டும்"
        result = llm._clean_output(text)
        assert result == "ஒரே வரி மட்டும்"


class TestValidateStructure:
    def test_valid_4_3(self, llm):
        text = "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
        assert llm._validate_structure(text) is True

    def test_invalid_3_3(self, llm):
        assert llm._validate_structure("w1 w2 w3\nw1 w2 w3") is False

    def test_invalid_4_4(self, llm):
        assert llm._validate_structure("w1 w2 w3 w4\nw1 w2 w3 w4") is False

    def test_invalid_single_line(self, llm):
        assert llm._validate_structure("w1 w2 w3 w4") is False

    def test_invalid_three_lines(self, llm):
        assert llm._validate_structure("w1 w2 w3 w4\nw1 w2 w3\nextra") is False

    def test_empty_string(self, llm):
        assert llm._validate_structure("") is False


def _mock_completion(content: str):
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestGenerateKural:
    @pytest.mark.asyncio
    async def test_returns_valid_kural_on_first_try(self, llm):
        valid = "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
        llm._client = AsyncMock()
        llm._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(valid)
        )

        result = await llm.generate_kural("system", "user")

        assert result == valid
        assert llm._client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_invalid_structure(self, llm):
        invalid = "w1 w2 w3\nw1 w2 w3"
        valid = "w1 w2 w3 w4\nw1 w2 w3"
        llm._client = AsyncMock()
        llm._client.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_completion(invalid),
                _mock_completion(valid),
            ]
        )

        result = await llm.generate_kural("system", "user")

        assert result == valid
        assert llm._client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_after_max_retries(self, llm):
        invalid = "w1 w2 w3\nw1 w2 w3"
        llm._client = AsyncMock()
        llm._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(invalid)
        )

        result = await llm.generate_kural("system", "user")

        assert result is None
        assert llm._client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_handles_empty_response(self, llm):
        llm._client = AsyncMock()
        llm._client.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_completion(""),
                _mock_completion(""),
                _mock_completion(""),
            ]
        )

        result = await llm.generate_kural("system", "user")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_llm_exception(self, llm):
        llm._client = AsyncMock()
        llm._client.chat.completions.create = AsyncMock(
            side_effect=Exception("connection refused")
        )

        result = await llm.generate_kural("system", "user")
        assert result is None

    @pytest.mark.asyncio
    async def test_cleans_before_validating(self, llm):
        wrapped = '```\nகுறள்: w1 w2 w3 w4\nw1 w2 w3\n```'
        llm._client = AsyncMock()
        llm._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(wrapped)
        )

        result = await llm.generate_kural("system", "user")
        assert result == "w1 w2 w3 w4\nw1 w2 w3"

    @pytest.mark.asyncio
    async def test_passes_correct_params(self, llm):
        valid = "w1 w2 w3 w4\nw1 w2 w3"
        llm._client = AsyncMock()
        llm._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(valid)
        )

        await llm.generate_kural("sys prompt", "usr prompt")

        llm._client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "usr prompt"},
            ],
            temperature=0.7,
            max_tokens=256,
        )
