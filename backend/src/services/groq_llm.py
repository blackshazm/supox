import httpx
from typing import Optional, List, Dict, Any
import json
import re
import logging

logger = logging.getLogger(__name__)

class GroqLLMService:
    """
    Service for Groq AI API.
    Highly optimized for latency.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str, default_model: Optional[str] = None):
        self.api_key = api_key
        self.default_model = default_model or "llama-3.3-70b-versatile"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _call_groq(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Make a call to Groq API (OpenAI-compatible)."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout,
            )

            body = response.json()
            if response.status_code != 200:
                error_msg = body.get("error", {}).get("message", str(body))
                logger.error(
                    "Groq API error: status=%s model=%s body=%s",
                    response.status_code,
                    model,
                    json.dumps(body),
                )
                raise RuntimeError(
                    f"Groq API Error {response.status_code} (model={model}): {error_msg}"
                )

            choices = body.get("choices")
            if not choices:
                raise RuntimeError(f"Groq returned invalid response: no choices")

            content = choices[0].get("message", {}).get("content", "")
            if not content.strip():
                raise RuntimeError(f"Groq returned empty content for model={model}")

            return {"content": content, "model": model}

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Small wrapper for JSON extraction since Groq supports json_object it should be cleaner."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback for manual cleaning if needed
            cleaned = content.strip()
            json_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
            if json_block_match:
                cleaned = json_block_match.group(1).strip()
            return json.loads(cleaned)

    async def analyze_video_content(
        self,
        transcript: str,
        video_metadata: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyses video transcript using Groq.
        """
        prompt = self._build_analysis_prompt(transcript, video_metadata)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in viral video editing. "
                    "Analyze the transcript and identify 3-5 best moments for shorts. "
                    "You MUST return the response in JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        target_model = model or self.default_model
        result = await self._call_groq(
            model=target_model,
            messages=messages,
        )
        return self._extract_json(result["content"])

    def _build_analysis_prompt(
        self,
        transcript: str,
        metadata: Dict[str, Any],
    ) -> str:
        return f"""
Analyze this video:
- Duration: {metadata.get('duration', 'N/A')}s
- Title: {metadata.get('title', 'N/A')}

Transcript:
{transcript}

Identify potential clips. Return JSON matching this schema:
{{
    "clips": [
        {{
            "start_time": float,
            "end_time": float,
            "hook_text": string,
            "virality_score": float (0-1),
            "why_viral": string
        }}
    ],
    "summary": string,
    "suggested_hashtags": [string]
}}
"""
