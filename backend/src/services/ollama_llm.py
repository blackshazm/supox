import httpx
from typing import Optional, List, Dict, Any
import json
import logging
import time
import re
import sys

logger = logging.getLogger(__name__)

class OllamaLLMService:
    """
    Service for local Ollama API.
    Highly optimized for JSON outputs and stability on local hardware.
    """

    def __init__(self, base_url: Optional[str] = None, default_model: Optional[str] = "llama3.1:8b"):
        # Se nenhuma base_url for configurada, assuma o padrão do Ollama hostado localmente
        # Nós usamos a API nativa do Ollama ao invés do OpenAI Wrap para forçar "format: json"
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        if self.base_url.endswith("/v1"):
            # Corrige usuários que setaram URL pro OpenAI proxy mas queremos usar a nativa
            self.base_url = self.base_url[:-3]
            
        self.api_url = f"{self.base_url}/api/chat"
        self.default_model = default_model

    async def _call_ollama(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout: float = 3600.0,
    ) -> Dict[str, Any]:
        """Make a streaming call to native Ollama API, logging tokens in real-time."""
        payload = {
            "model": model,
            "messages": messages,
            "format": "json",
            "stream": True,  # streaming enabled for real-time visibility
            "options": {
                "temperature": temperature,
                "num_ctx": 12288
            }
        }

        async with httpx.AsyncClient() as client:
            start_time = time.time()
            full_content = ""
            token_count = 0
            last_log_at = 0

            logger.info("[Ollama] ⏳ Starting streaming request to model=%s ...", model)

            async with client.stream(
                "POST",
                self.api_url,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    try:
                        err = json.loads(body).get("error", body.decode())
                    except Exception:
                        err = body.decode(errors="replace")
                    logger.error(
                        "Ollama API error: status=%s model=%s error=%s",
                        response.status_code, model, err,
                    )
                    raise RuntimeError(
                        f"Ollama API Error {response.status_code} (model={model}): {err}"
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    full_content += token
                    token_count += len(token)

                    # Log progress every ~500 chars so you can see the IA working
                    if token_count - last_log_at >= 500:
                        last_log_at = token_count
                        elapsed = time.time() - start_time
                        preview = full_content[-120:].replace("\n", " ")
                        logger.info(
                            "[Ollama] 🤖 ~%d chars in %.1fs | ...%s",
                            token_count, elapsed, preview,
                        )

                    if chunk.get("done"):
                        break

            if not full_content.strip():
                raise RuntimeError(f"Ollama returned empty content for model={model}")

            elapsed_total = time.time() - start_time
            logger.info(
                "[Ollama] ✅ Done! model=%s | total=%d chars | time=%.2fs",
                model, len(full_content), elapsed_total,
            )
            return {"content": full_content, "model": model}

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """ Wrapper for JSON extraction with fallbacks. """
        # Remove eventuais tags de reasoning/think
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        try:
            # First attempt: direct parse
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Second attempt: handle markdown blocks
            json_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
            if json_block_match:
                cleaned_block = json_block_match.group(1).strip()
                try:
                    return json.loads(cleaned_block)
                except json.JSONDecodeError:
                    pass

            # Third attempt: extract first outer JSON object via regex
            json_obj_match = re.search(r'(\{[\s\S]*\})', cleaned)
            if json_obj_match:
                try:
                    return json.loads(json_obj_match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            
            # Fourth attempt: detect if it's a list but we want a dict with 'clips'
            try:
                possible_list = json.loads(cleaned)
                if isinstance(possible_list, list):
                    return {"clips": possible_list}
            except Exception:
                pass
            
            # Log the failure and the content for debugging
            logger.error("Failed to parse JSON from Ollama. Content: %s", content[:500])
            raise

    async def analyze_video_content(
        self,
        transcript: str,
        video_metadata: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyses video transcript using Local Ollama natively.
        Calculates dynamic clip quota based on total duration.
        """
        duration_s = float(video_metadata.get('duration', 0) or 0)
        if duration_s <= 0:
            # Fallback para estimativa por contagem de palavras (aprox 130 palavras/minuto)
            words = len(transcript.split())
            duration_s = (words / 130.0) * 60.0

        # Escala dinamica de quantidade de cortes por duracao:
        if duration_s <= 1500:  # Ate 25 minutos
            min_clips, max_clips = 4, 8
        elif duration_s <= 3600:  # 25 a 60 minutos
            min_clips, max_clips = 8, 14
        elif duration_s <= 7200:  # 1h a 2h
            min_clips, max_clips = 14, 22
        else:  # Mais de 2h
            min_clips, max_clips = 20, 30

        logger.info(
            "[Ollama] Video duration: %.1fs (%02d:%02d) -> Meta dinamica: %d a %d cortes",
            duration_s, int(duration_s) // 60, int(duration_s) % 60, min_clips, max_clips
        )

        prompt = self._build_analysis_prompt(transcript, video_metadata, min_clips, max_clips)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert viral video editor specialized in YouTube Shorts and TikTok. "
                    f"Analyze the transcript and identify between {min_clips} and {max_clips} best moments for short-form clips, well-distributed across the entire timeline.\n\n"
                    "RULES FOR VIRAL RETENTION & STORY STRUCTURE:\n"
                    "1. 3-ACT NARRATIVE: Each clip MUST contain: (A) Strong Hook (0-7s) with immediate tension or provocative question; (B) Story context and development (8-45s); (C) Climax & clean conclusion/payoff (46-65s+). Never cut mid-sentence.\n"
                    "2. OPTIMAL DURATION: 50 to 75 seconds. NEVER pick clips shorter than 45 seconds.\n"
                    "3. MULTI-PART SERIES (PARTE 1, 2, 3 / A, B, C): When an extraordinary story or debate lasts 1m30s to 3 minutes, DO NOT compress it into a weak 30s cut or discard it. Instead, extract it as 2 or 3 contiguous connected parts (50-70 seconds each). For each part in a series, set the same 'series_id' (e.g. 'serie-1'), 'part_number' (1, 2, or 3), 'total_parts' (2 or 3), and start 'hook_title' with '[Parte 1/2]' or '[Parte 1/3]'. Part 1 ends on suspense/tension, Part 2 on the climax, Part 3 on the lesson/payoff.\n"
                    "4. STRICT ANTI-SPONSOR & ANTI-INTRO: Skip greetings, sponsor readings, discount codes, channel plugs.\n"
                    "5. TIMESTAMPS: Strictly valid 'MM:SS' format. start_time MUST be strictly less than end_time.\n"
                    f"6. QUANTITY: Find between {min_clips} and {max_clips} distinct non-overlapping viral moments across the whole video.\n\n"
                    "You MUST return the response in strict JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        target_model = model or self.default_model
        result = await self._call_ollama(
            model=target_model,
            messages=messages,
        )
        return self._extract_json(result["content"])

    def _build_analysis_prompt(
        self,
        transcript: str,
        metadata: Dict[str, Any],
        min_clips: int = 4,
        max_clips: int = 8,
    ) -> str:
        duration_s = float(metadata.get('duration', 0) or 0)
        duration_m = int(duration_s) // 60
        duration_sec = int(duration_s) % 60
        duration_str = f"{duration_m:02d}:{duration_sec:02d}" if duration_s > 0 else "unknown"
        
        return f"""Analyze this video:
- Duration: {duration_s:.0f}s ({duration_str} in MM:SS format)
- Title: {metadata.get('title', 'N/A')}

CRITICAL RULES:
1. The video is exactly {duration_s:.0f} seconds long ({duration_str}).
2. ALL start_time and end_time values MUST be within the range 00:00 to {duration_str}.
3. start_time MUST be less than end_time.
4. Use ONLY the exact format MM:SS (e.g. "02:30").
5. Each clip must be between 50 and 75 seconds long (allowing up to 90s for deep stories). Clips shorter than 45s are strictly forbidden.
6. TARGET COUNT: Extract between {min_clips} and {max_clips} high-impact viral moments distributed across the entire duration.

Transcript:
{transcript}

Return JSON matching exactly this schema:
{{
    "clips": [
        {{
            "start_time": "MM:SS",
            "end_time": "MM:SS",
            "hook_title": "Short punchy headline for the clip",
            "hook_text": "Compelling hook text for this clip",
            "virality_score": 0.85,
            "why_viral": "Detailed reasoning why this segment is viral with 3-act narrative",
            "series_id": null,
            "part_number": 1,
            "total_parts": 1
        }}
    ],
    "summary": "Full transcript summary",
    "suggested_hashtags": ["hashtag1", "hashtag2"]
}}
"""
