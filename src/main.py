"""
Creator Agent — generates kural venba responses for arena prompt requests.

Connects to the arena-server, fetches open requests that this agent hasn't
responded to, generates kural venba poetry using a local LLM (LM Studio),
and submits responses.

Usage:
    python -m src.main
"""

import asyncio
import logging
import random

from dotenv import load_dotenv

from src.arena_client import ArenaClient
from src.config import Config
from src.llm_client import LLMClient
from src.prompt_loader import PromptLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def create_cycle(
    api: ArenaClient,
    llm: LLMClient,
    prompts: PromptLoader,
    agent_id: str,
) -> int:
    """Fetch unanswered requests, generate kurals, and submit responses.

    Returns the number of successfully submitted responses.
    """
    system_prompt = prompts.get_system_prompt()
    cursor: str | None = None
    count = 0

    while True:
        page = await api.list_open_requests(
            agent_id=agent_id,
            limit=100,
            cursor=cursor,
        )
        requests = page.get("data", [])
        if not requests:
            break

        for req in requests:
            req_id = req["id"]
            short_id = req_id[:8]
            prompt_text = req["prompt"]

            user_prompt = prompts.get_user_prompt(prompt_text)
            kural = await llm.generate_kural(system_prompt, user_prompt)
            if not kural:
                logger.warning("LLM failed to generate for request %s, skipping", short_id)
                continue

            try:
                await api.submit_response(req_id, kural)
                logger.info("Submitted response for %s: %s", short_id, kural[:60])
                count += 1
            except Exception as e:
                logger.warning("Failed to submit response for %s: %s", short_id, e)

        cursor = page.get("next_cursor")
        if not cursor:
            break

    return count


async def main():
    load_dotenv()
    config = Config.from_env()
    prompts = PromptLoader(config.prompt_dir)

    api = ArenaClient(config.arena_base_url, config.api_key)
    llm = LLMClient(
        config.llm_base_url,
        config.llm_model,
        config.llm_temperature,
        config.llm_max_tokens,
        config.llm_max_retries,
        api_key=config.llm_api_key,
    )

    try:
        await api.wait_for_server()

        logger.info(
            "Creator agent started — agent=%s interval=%ds api=%s llm=%s model=%s prompts=%s",
            config.agent_id[:8],
            config.poll_interval,
            config.arena_base_url,
            config.llm_base_url,
            config.llm_model,
            config.prompt_dir,
        )

        while True:
            try:
                count = await create_cycle(api, llm, prompts, config.agent_id)
                if count > 0:
                    logger.info("Submitted %d responses this cycle", count)
            except Exception as e:
                logger.warning("Creator cycle failed: %s", e)

            jitter = config.poll_interval * 0.2
            await asyncio.sleep(
                config.poll_interval + random.uniform(-jitter, jitter)
            )
    finally:
        await api.close()
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
