import logging
import re

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around OpenAI SDK for LM Studio (OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int = 3,
        api_key: str = "ollama",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def close(self):
        await self._client.close()

    def _clean_output(self, text: str) -> str | None:
        """Strip common LLM wrappers from the generated kural text."""
        # Remove markdown code fences
        text = re.sub(r"```(?:\w*)\n?", "", text)
        # Remove leading/trailing quotes
        text = text.strip().strip("\"'""''«»")
        # Remove common preambles (Tamil and English)
        text = re.sub(
            r"^(?:குறள்\s*:?\s*|Here is (?:your |the )?kural\s*:?\s*|வெண்பா\s*:?\s*|கவிதை\s*:?\s*|Verse\s*:?\s*)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Keep only the first two non-empty lines
        lines = text.strip().splitlines()
        content_lines = [l.strip() for l in lines if l.strip()]
        if len(content_lines) >= 2:
            text = content_lines[0] + "\n" + content_lines[1]
        else:
            text = "\n".join(content_lines)
        text = text.strip()
        return text if text else None

    def _validate_structure(self, text: str) -> bool:
        """Check that text has exactly 2 lines with 4 and 3 words."""
        lines = text.strip().splitlines()
        if len(lines) != 2:
            return False
        words_line1 = lines[0].strip().split()
        words_line2 = lines[1].strip().split()
        return len(words_line1) == 4 and len(words_line2) == 3

    async def generate_kural(self, system_prompt: str, user_prompt: str) -> str | None:
        """Generate a structurally valid kural venba.

        Retries up to max_retries times if structure is incorrect.
        Returns the cleaned text or None on failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content
                if not text:
                    logger.debug("Attempt %d/%d: empty response", attempt, self.max_retries)
                    continue

                cleaned = self._clean_output(text)
                if not cleaned:
                    logger.debug("Attempt %d/%d: empty after cleaning", attempt, self.max_retries)
                    continue

                if self._validate_structure(cleaned):
                    if attempt > 1:
                        logger.info("Structural validation passed on attempt %d", attempt)
                    return cleaned

                lines = cleaned.strip().splitlines()
                word_counts = [len(l.split()) for l in lines]
                logger.info(
                    "Attempt %d/%d: structure invalid (lines=%d, words=%s), retrying",
                    attempt,
                    self.max_retries,
                    len(lines),
                    word_counts,
                )
            except Exception as e:
                logger.warning("Attempt %d/%d LLM call failed: %s", attempt, self.max_retries, e)

        logger.warning("All %d attempts failed structural validation", self.max_retries)
        return None
