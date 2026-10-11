import httpx
from typing import Optional, List, Dict, Any
import json
import re
import logging

logger = logging.getLogger(__name__)

class ZAILLMService:
    """
    Serviço de LLM via Z.AI API.
    Provides access to the GLM-4.7-Flash model and others.
    """

    BASE_URL = "https://api.z.ai/api/paas/v4"

    def __init__(self, api_key: str, default_model: Optional[str] = None):
        self.api_key = api_key
        self.default_model = default_model or "glm-4.7"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _call_zai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        enable_thinking: bool = True,
    ) -> Dict[str, Any]:
        """Make a single call to Z.AI and return the parsed JSON response."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if enable_thinking:
            payload["thinking"] = {"type": "enabled"}

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
                    "Z.AI API error: status=%s model=%s body=%s",
                    response.status_code,
                    model,
                    json.dumps(body),
                )
                raise RuntimeError(
                    f"Z.AI API Error {response.status_code} (model={model}): {error_msg}"
                )

            choices = body.get("choices")
            if not choices:
                raise RuntimeError(f"Z.AI returned invalid response: no choices")

            content = choices[0].get("message", {}).get("content", "")
            if not content.strip():
                raise RuntimeError(f"Z.AI returned empty content for model={model}")

            return {"content": content, "model": model}

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Robustly extract JSON from LLM content."""
        cleaned = content.strip()
        
        # Markdown block extraction
        json_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if json_block_match:
            cleaned = json_block_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Brace matching fallback
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from Z.AI response: {content[:300]}...")

    async def analyze_video_content(
        self,
        transcript: str,
        video_metadata: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyses video transcript for viral clips using GLM-4.7 model.
        """
        prompt = self._build_analysis_prompt(transcript, video_metadata)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in viral video editing and content marketing. "
                    "Analyze the video and identify the best moments for creating viral shorts. "
                    "Consider hooks, educational/emotional value, and natural boundaries. "
                    "Return ONLY valid JSON. No preamble."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            target_model = model or self.default_model
            result = await self._call_zai(
                model=target_model,
                messages=messages,
                temperature=0.3,
            )
            parsed = self._extract_json(result["content"])
            logger.info("Z.AI success with model=%s", target_model)
            return parsed
        except Exception as e:
            logger.error("Z.AI analysis failed: %s", str(e))
            raise

    def _build_analysis_prompt(
        self,
        transcript: str,
        metadata: Dict[str, Any],
    ) -> str:
        return f"""
Analyze the following video and identify 3-5 best clips for social media:

VIDEO METADATA:
- Duration: {metadata.get('duration', 'N/A')} seconds
- Title: {metadata.get('title', 'N/A')}

TRANSCRIPT:
{transcript}

RESPONSE FORMAT (JSON only):
{{
    "clips": [
        {{
            "start_time": 45.5,
            "end_time": 89.2,
            "hook_text": "Clip hook label",
            "virality_score": 0.95,
            "why_viral": "Explanation"
        }}
    ],
    "summary": "Brief summary",
    "suggested_hashtags": ["#tag1", "#tag2"]
}}
"""
