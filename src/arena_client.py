import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


class ArenaClient:
    """Async HTTP client for the Arena server API with Bearer token auth."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

    async def close(self):
        await self._client.aclose()

    async def wait_for_server(self, timeout: float = 120) -> None:
        """Poll the arena-server health endpoint until it's ready."""
        url = f"{self.base_url}/health/ready"
        deadline = asyncio.get_event_loop().time() + timeout
        delay = 1.0

        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=5)
                    if resp.status_code == 200:
                        logger.info("Arena server is ready")
                        return
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                pass
            logger.info("Waiting for arena server at %s (retry in %.0fs)", url, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)

        raise TimeoutError(f"Arena server not ready after {timeout}s")

    async def list_open_requests(
        self,
        agent_id: str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Fetch open requests that this agent has not yet responded to."""
        params: dict = {
            "status": "open",
            "not_responded_by": agent_id,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        resp = await self._client.get("/requests", params=params)
        resp.raise_for_status()
        return resp.json()

    async def submit_response(self, request_id: str, content: str) -> dict:
        """Submit a generated kural response for a request."""
        resp = await self._client.post(
            "/responses",
            json={"request_id": request_id, "content": content},
        )
        resp.raise_for_status()
        return resp.json()
