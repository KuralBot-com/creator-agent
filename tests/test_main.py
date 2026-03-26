"""Tests for the create_cycle function in main."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.main import create_cycle


def _make_api(pages):
    """Create a mock ArenaClient that returns pages of requests.

    pages: list of (data, next_cursor) tuples
    """
    api = AsyncMock()
    responses = [
        {"data": data, "next_cursor": cursor}
        for data, cursor in pages
    ]
    api.list_open_requests = AsyncMock(side_effect=responses)
    api.submit_response = AsyncMock(return_value={"id": "resp-1"})
    return api


def _make_llm(outputs):
    """Create a mock LLMClient that returns outputs in sequence."""
    llm = AsyncMock()
    llm.generate_kural = AsyncMock(side_effect=outputs)
    return llm


def _make_prompts():
    prompts = MagicMock()
    prompts.get_system_prompt.return_value = "system"
    prompts.get_user_prompt.side_effect = lambda p: f"user: {p}"
    return prompts


class TestCreateCycle:
    @pytest.mark.asyncio
    async def test_no_requests(self):
        api = _make_api([([], None)])
        llm = _make_llm([])
        prompts = _make_prompts()

        count = await create_cycle(api, llm, prompts, "agent-1")

        assert count == 0
        llm.generate_kural.assert_not_called()
        api.submit_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_request_success(self):
        api = _make_api([
            ([{"id": "req-1", "prompt": "அறம்"}], None),
        ])
        kural = "w1 w2 w3 w4\nw1 w2 w3"
        llm = _make_llm([kural])
        prompts = _make_prompts()

        count = await create_cycle(api, llm, prompts, "agent-1")

        assert count == 1
        llm.generate_kural.assert_called_once_with("system", "user: அறம்")
        api.submit_response.assert_called_once_with("req-1", kural)

    @pytest.mark.asyncio
    async def test_multiple_requests(self):
        api = _make_api([
            ([
                {"id": "req-1", "prompt": "p1"},
                {"id": "req-2", "prompt": "p2"},
                {"id": "req-3", "prompt": "p3"},
            ], None),
        ])
        llm = _make_llm(["kural1", "kural2", "kural3"])
        prompts = _make_prompts()

        count = await create_cycle(api, llm, prompts, "agent-1")

        assert count == 3
        assert api.submit_response.call_count == 3

    @pytest.mark.asyncio
    async def test_pagination(self):
        api = _make_api([
            ([{"id": "req-1", "prompt": "p1"}], "cursor-abc"),
            ([{"id": "req-2", "prompt": "p2"}], None),
        ])
        llm = _make_llm(["k1", "k2"])
        prompts = _make_prompts()

        count = await create_cycle(api, llm, prompts, "agent-1")

        assert count == 2
        calls = api.list_open_requests.call_args_list
        assert calls[0].kwargs.get("cursor") is None or calls[0][1].get("cursor") is None
        # Second call should use the cursor
        assert calls[1].kwargs.get("cursor") == "cursor-abc" or calls[1][1].get("cursor") == "cursor-abc"

    @pytest.mark.asyncio
    async def test_skips_when_llm_returns_none(self):
        api = _make_api([
            ([
                {"id": "req-1", "prompt": "p1"},
                {"id": "req-2", "prompt": "p2"},
            ], None),
        ])
        llm = _make_llm([None, "kural2"])
        prompts = _make_prompts()

        count = await create_cycle(api, llm, prompts, "agent-1")

        assert count == 1
        api.submit_response.assert_called_once_with("req-2", "kural2")

    @pytest.mark.asyncio
    async def test_handles_submit_failure(self):
        api = _make_api([
            ([
                {"id": "req-1", "prompt": "p1"},
                {"id": "req-2", "prompt": "p2"},
            ], None),
        ])
        llm = _make_llm(["k1", "k2"])
        prompts = _make_prompts()
        api.submit_response = AsyncMock(
            side_effect=[Exception("server error"), {"id": "resp-2"}]
        )

        count = await create_cycle(api, llm, prompts, "agent-1")

        # First submit fails, second succeeds
        assert count == 1
