"""Tests for ArenaClient — HTTP interactions with the Arena server."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from src.arena_client import ArenaClient


@pytest.fixture
def client():
    return ArenaClient("http://localhost:3000", "test-api-key")


@pytest.fixture
def _cleanup(client):
    yield
    import asyncio
    asyncio.get_event_loop().run_until_complete(client.close())


class TestArenaClientInit:
    def test_strips_trailing_slash(self):
        c = ArenaClient("http://localhost:3000/", "key")
        assert c.base_url == "http://localhost:3000"

    def test_sets_auth_header(self, client):
        assert client._client.headers["Authorization"] == "Bearer test-api-key"


class TestListOpenRequests:
    @pytest.mark.asyncio
    async def test_basic_call(self, client):
        mock_response = httpx.Response(
            200,
            json={"data": [{"id": "req-1", "prompt": "test"}], "next_cursor": None},
            request=httpx.Request("GET", "http://localhost:3000/requests"),
        )
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.list_open_requests(agent_id="agent-1")

        client._client.get.assert_called_once_with(
            "/requests",
            params={"status": "open", "not_responded_by": "agent-1", "limit": 100},
        )
        assert result["data"][0]["id"] == "req-1"

    @pytest.mark.asyncio
    async def test_with_cursor(self, client):
        mock_response = httpx.Response(
            200,
            json={"data": [], "next_cursor": None},
            request=httpx.Request("GET", "http://localhost:3000/requests"),
        )
        client._client.get = AsyncMock(return_value=mock_response)

        await client.list_open_requests(agent_id="a", limit=50, cursor="abc123")

        call_kwargs = client._client.get.call_args
        params = call_kwargs.kwargs["params"] if "params" in call_kwargs.kwargs else call_kwargs[1]["params"]
        assert params["cursor"] == "abc123"
        assert params["limit"] == 50

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, client):
        mock_response = httpx.Response(
            500,
            json={"error": "internal"},
            request=httpx.Request("GET", "http://localhost:3000/requests"),
        )
        client._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPStatusError):
            await client.list_open_requests(agent_id="a")


class TestSubmitResponse:
    @pytest.mark.asyncio
    async def test_submit(self, client):
        mock_response = httpx.Response(
            201,
            json={"id": "resp-1", "request_id": "req-1", "content": "kural text"},
            request=httpx.Request("POST", "http://localhost:3000/responses"),
        )
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.submit_response("req-1", "kural text")

        client._client.post.assert_called_once_with(
            "/responses",
            json={"request_id": "req-1", "content": "kural text"},
        )
        assert result["id"] == "resp-1"

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, client):
        mock_response = httpx.Response(
            409,
            json={"error": "duplicate"},
            request=httpx.Request("POST", "http://localhost:3000/responses"),
        )
        client._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPStatusError):
            await client.submit_response("req-1", "kural text")


class TestWaitForServer:
    @pytest.mark.asyncio
    async def test_succeeds_immediately(self, client):
        with patch("src.arena_client.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(
                return_value=httpx.Response(
                    200,
                    request=httpx.Request("GET", "http://localhost:3000/health/ready"),
                )
            )
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_ctx

            await client.wait_for_server(timeout=5)

    @pytest.mark.asyncio
    async def test_timeout_raises(self, client):
        with patch("src.arena_client.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_ctx

            with pytest.raises(TimeoutError, match="not ready after"):
                await client.wait_for_server(timeout=0.1)
