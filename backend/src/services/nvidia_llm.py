import httpx
from typing import Optional, List, Dict, Any
import json
import re
import logging
import asyncio

logger = logging.getLogger(__name__)


class NVIDIALLMService:
    """
    Service for NVIDIA NIM API (build.nvidia.com).
    Compatible with OpenAI specification and optimized for high-performance
    inference using NVIDIA Nemotron and partner models.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "moonshotai/kimi-k2.5"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.api_key = api_key.strip()
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.default_model = default_model or self.DEFAULT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _call_nvidia(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 90.0,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Make a call to NVIDIA NIM API with retry backoff for rate limits."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient() as client:
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                        timeout=timeout,
                    )

                    if response.status_code == 200:
                        body = response.json()
                        choices = body.get("choices")
                        if not choices:
                            raise RuntimeError("NVIDIA API returned invalid response: no choices")

                        content = choices[0].get("message", {}).get("content", "")
                        if not content.strip():
                            raise RuntimeError(f"NVIDIA API returned empty content for model={model}")

                        return {"content": content, "model": model}

                    # Handle 429 Rate Limit with exponential backoff
                    if response.status_code == 429 and attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2.0
                        logger.warning(
                            "NVIDIA API rate limit (429), retrying in %.1fs (attempt %d/%d)...",
                            wait_time,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    # Other non-200 responses
                    try:
                        body = response.json()
                        error_msg = body.get("error", {}).get("message", str(body))
                    except Exception:
                        error_msg = response.text or f"HTTP {response.status_code}"

                    logger.error(
                        "NVIDIA API error: status=%s model=%s message=%s",
                        response.status_code,
                        model,
                        error_msg,
                    )
                    raise RuntimeError(
                        f"NVIDIA API Error {response.status_code} (model={model}): {error_msg}"
                    )

                except httpx.RequestError as exc:
                    last_error = exc
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 1.5
                        logger.warning(
                            "NVIDIA API connection error: %s. Retrying in %.1fs...",
                            exc,
                            wait_time,
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Failed to connect to NVIDIA API after {max_retries} attempts: {exc}") from exc

            if last_error:
                raise RuntimeError(f"NVIDIA API call failed: {last_error}") from last_error
            raise RuntimeError(f"NVIDIA API call failed after {max_retries} attempts")

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Safely extract and parse JSON from model output."""
        cleaned = content.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback for markdown-wrapped JSON blocks
            json_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
            if json_block_match:
                cleaned = json_block_match.group(1).strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

            # Fallback: find outermost curly braces
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass

            raise RuntimeError(f"Could not parse valid JSON from NVIDIA model output: {cleaned[:200]}...")

    async def analyze_video_content(
        self,
        transcript: str,
        video_metadata: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze video transcript using NVIDIA NIM model (e.g. Llama-3.1-Nemotron-70B-Instruct).
        """
        prompt = self._build_analysis_prompt(transcript, video_metadata)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a master viral video editor, director, and retention specialist. "
                    "Analyze the given video transcript with extreme precision. "
                    "Select only high-impact moments that work as standalone viral shorts (TikTok, Reels, Shorts). "
                    "Requirements:\n"
                    "- Only select continuous, coherent segments from the transcript.\n"
                    "- Every clip must have a strong hook within the first 3 seconds.\n"
                    "- Output strictly valid JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        target_model = model or self.default_model
        result = await self._call_nvidia(
            model=target_model,
            messages=messages,
        )
        return self._extract_json(result["content"])

    def _build_analysis_prompt(
        self,
        transcript: str,
        metadata: Dict[str, Any],
    ) -> str:
        duration = metadata.get("duration", "N/A")
        title = metadata.get("title", "N/A")

        return f"""Analyze this video transcript and identify 3-5 high-performing viral clip candidates:

Video Details:
- Title: {title}
- Total Duration: {duration}s

Transcript:
{transcript}

Rules:
1. Clip duration should be between 30 and 75 seconds.
2. start_time and end_time can be seconds (float) or MM:SS format matching the transcript timestamps.
3. virality_score must be a float between 0.0 and 1.0 based on hook strength, emotional resonance, and retention value.
4. hook_text should be a magnetic, curiosity-inducing headline under 12 words.

Return JSON in this exact structure:
{{
    "clips": [
        {{
            "start_time": 0.0,
            "end_time": 45.0,
            "hook_text": "Magnetic hook title",
            "virality_score": 0.92,
            "why_viral": "Explanation of retention, emotional trigger, and payout"
        }}
    ],
    "summary": "Brief summary of the video content",
    "suggested_hashtags": ["#shorts", "#viral", "#topic"]
}}"""
