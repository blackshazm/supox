import httpx
from typing import Optional, List, Dict, Any
import json
import re
import os
import logging

logger = logging.getLogger(__name__)

# Fallback model chain: try these in order when the primary model fails
FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "meta-llama/llama-3.3-70b-instruct",
    "nvidia/nemotron-3.5-lightning:free",
    "openrouter/free",
]


class OpenRouterService:
    """
    Serviço de LLM via OpenRouter - agregador de modelos.
    Includes retry logic and automatic model fallback.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, default_model: Optional[str] = None):
        self.api_key = api_key
        self.default_model = default_model or "nvidia/nemotron-3-ultra-550b-a55b:free"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "SupoClip",
        }

    async def _call_openrouter(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout: float = 90.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a single call to OpenRouter and return the parsed JSON body."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout,
            )

            body = response.json()
            if response.status_code != 200:
                error_detail = body.get("error", {})
                error_msg = error_detail.get("message", str(body)) if isinstance(error_detail, dict) else str(error_detail)
                logger.error(
                    "OpenRouter API error: status=%s model=%s body=%s",
                    response.status_code,
                    model,
                    json.dumps(body),
                )
                raise RuntimeError(
                    f"OpenRouter API Error {response.status_code} (model={model}): {error_msg}"
                )

            # Validate response structure
            choices = body.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                logger.error(
                    "OpenRouter returned empty choices: model=%s body=%s",
                    model,
                    str(body)[:500],
                )
                raise RuntimeError(
                    f"OpenRouter returned invalid response for model={model}: no choices"
                )

            content = choices[0].get("message", {}).get("content", "")
            if not content.strip():
                raise RuntimeError(
                    f"OpenRouter returned empty content for model={model}"
                )

            return {"content": content, "model": model}

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """
        Robustly extract JSON from LLM content that may contain
        markdown code blocks, preamble text, or trailing text.
        """
        # Strip markdown code fences
        cleaned = content.strip()

        # Try extracting from ```json ... ``` blocks
        json_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if json_block_match:
            cleaned = json_block_match.group(1).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try finding the first { ... } block (greedy)
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from LLM response: {cleaned[:300]}...")

    async def analyze_video_content(
        self,
        transcript: str,
        video_metadata: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analisa transcrição e identifica melhores momentos para clips.
        Retries with fallback models if the primary model fails.
        """
        duration_s = float(video_metadata.get('duration', 0) or 0)
        duration_m = int(duration_s) // 60
        duration_sec = int(duration_s) % 60
        duration_str = f"{duration_m:02d}:{duration_sec:02d}" if duration_s > 0 else "unknown"

        prompt = self._build_analysis_prompt(transcript, video_metadata, duration_s, duration_str)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an elite short-form video editor specialized in viral TikToks, YouTube Shorts, and Instagram Reels. "
                    "Your mission is to extract the 3 to 7 highest-impact, most viral segments from this podcast transcript.\n\n"
                    "RULES FOR VIRAL RETENTION & STORY STRUCTURE:\n"
                    "1. 3-ACT NARRATIVE (MANDATORY): Each clip MUST contain: (A) Strong Hook (0-7s) with immediate tension, shocking statement or provocative question; (B) Story context and development (8-45s); (C) Climax & clean conclusion/payoff (46-65s+). Never cut mid-sentence or leave the viewer hanging without a conclusion.\n"
                    "2. OPTIMAL DURATION: 50 to 75 seconds (allows full narrative arc). NEVER pick clips shorter than 45 seconds (they lack depth and context).\n"
                    "3. MULTI-PART SERIES (PARTE 1, 2, 3 / A, B, C): When an extraordinary story or debate lasts 1m30s to 3 minutes, DO NOT compress it into a weak 30s cut or discard it. Instead, extract it as 2 or 3 contiguous connected parts (50-70 seconds each). For each part in a series, set the same 'series_id' (e.g. 'serie-1'), 'part_number' (1, 2, or 3), 'total_parts' (2 or 3), and start 'hook_title' with '[Parte 1/2]' or '[Parte 1/3]'. Part 1 ends on suspense/tension, Part 2 on the climax, Part 3 on the lesson/payoff.\n"
                    "4. STRICT ANTI-SPONSOR & ANTI-INTRO FILTER: Strictly SKIP opening minutes containing greetings/sound checks, sponsor readings, discount codes ('cupom', 'patrocínio', 'merchan', 'link na descrição', 'superchat', 'manda o pix').\n"
                    "5. TIMESTAMPS: Strictly valid 'MM:SS' format found in transcript. start_time MUST be strictly less than end_time.\n\n"
                    "Return ONLY valid JSON matching the requested schema. No markdown, no backticks, no preamble."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Build ordered list of models to try
        primary = model or self.default_model
        models_to_try = [primary]
        for fallback in FALLBACK_MODELS:
            if fallback != primary and fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error: Optional[Exception] = None
        for i, try_model in enumerate(models_to_try):
            try:
                logger.info(
                    "OpenRouter attempt %d/%d with model=%s",
                    i + 1,
                    len(models_to_try),
                    try_model,
                )
                try:
                    result = await self._call_openrouter(
                        model=try_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4000,
                        timeout=90.0,
                        response_format={"type": "json_object"},
                    )
                except Exception as call_err:
                    # Some older models or endpoints might not accept response_format
                    logger.debug("Attempting without response_format due to: %s", call_err)
                    result = await self._call_openrouter(
                        model=try_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4000,
                        timeout=90.0,
                    )
                parsed = self._extract_json(result["content"])
                logger.info(
                    "OpenRouter success with model=%s, clips=%d",
                    try_model,
                    len(parsed.get("clips", [])),
                )
                return parsed
            except Exception as e:
                last_error = e
                logger.warning(
                    "OpenRouter model %s failed (attempt %d/%d): %s",
                    try_model,
                    i + 1,
                    len(models_to_try),
                    str(e),
                )
                continue

        # Fallback to direct NVIDIA API if NV_API_KEY is configured
        nv_key = os.getenv("NV_API_KEY")
        if nv_key:
            try:
                logger.info("OpenRouter models exhausted. Attempting fallback to direct NVIDIA API...")
                nv_content = await self._call_nvidia_api(nv_key, messages)
                if nv_content:
                    parsed = self._extract_json(nv_content)
                    if parsed and len(parsed.get("clips", [])) > 0:
                        logger.info(
                            "Direct NVIDIA API fallback succeeded with %d clips",
                            len(parsed["clips"]),
                        )
                        return parsed
            except Exception as nv_exc:
                logger.warning("Direct NVIDIA API fallback failed: %s", nv_exc)

        raise RuntimeError(
            f"All OpenRouter and fallback models failed. Last error: {last_error}"
        )

    async def _call_nvidia_api(
        self,
        nv_key: str,
        messages: List[Dict[str, str]],
        timeout: float = 60.0,
    ) -> Optional[str]:
        """Direct call to NVIDIA NIM API as a fallback."""
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {nv_key}",
            "Content-Type": "application/json",
        }
        for nv_model in [
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "google/gemma-4-31b-it",
            "nvidia/llama-3.1-nemotron-51b-instruct",
        ]:
            try:
                logger.info("Attempting NVIDIA NIM model=%s", nv_model)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        url,
                        headers=headers,
                        json={
                            "model": nv_model,
                            "messages": messages,
                            "temperature": 0.3,
                            "max_tokens": 2000,
                        },
                    )
                    if resp.status_code == 200:
                        body = resp.json()
                        choices = body.get("choices") or []
                        if choices:
                            return choices[0].get("message", {}).get("content", "")
                    else:
                        logger.warning(
                            "NVIDIA NIM model %s failed with status %d: %s",
                            nv_model,
                            resp.status_code,
                            resp.text[:200],
                        )
            except Exception as e:
                logger.warning("NVIDIA NIM model %s request failed: %s", nv_model, e)
        return None

    def _build_analysis_prompt(
        self,
        transcript: str,
        metadata: Dict[str, Any],
        duration_s: float = 0.0,
        duration_str: str = "unknown",
    ) -> str:
        return f"""Analyze the podcast transcript below and identify the 3 to 7 most viral, high-retention segments:

VIDEO METADATA:
- Duration: {duration_s:.0f} seconds ({duration_str} in MM:SS)
- Title: {metadata.get('title', 'N/A')}

CRITICAL VIRALITY RULES:
1. Strictly avoid the introductory minutes if they contain generic greetings, technical sound tests, or reading sponsors.
2. Completely discard segments mentioning promotions, coupon codes, channel subscriptions or social media plugs.
3. Prioritize moments of high emotion: intense storytelling, revelations, conflict, humor, and counter-intuitive insights.
4. Timestamps MUST be MM:SS format matching the transcript timestamps.

TRANSCRIPT WITH TIMESTAMPS:
{transcript}

RESPONSE FORMAT (JSON only, no markdown, no other text):
{{
    "clips": [
        {{
            "start_time": "01:15",
            "end_time": "02:18",
            "hook_title": "Ele revelou o que ninguém tinha coragem de dizer",
            "hook_text": "Strong opening hook line from transcript",
            "virality_score": 0.88,
            "category": "inspiration",
            "why_viral": "Complete 3-act story with strong hook, narrative build and surprising payoff",
            "series_id": null,
            "part_number": 1,
            "total_parts": 1
        }}
    ],
    "summary": "Brief video summary",
    "suggested_hashtags": ["#viral", "#podcast"]
}}
"""

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Retorna lista de modelos disponíveis via OpenRouter
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/models",
                headers=self.headers,
            )
            return response.json().get("data", [])
