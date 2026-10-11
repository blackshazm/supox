import pytest
from unittest.mock import AsyncMock, patch

from src.config import Config
from src.services.nvidia_llm import NVIDIALLMService
from src.ai import (
    _split_llm_name,
    _get_missing_llm_key_error,
    _build_transcript_model,
)
from pydantic_ai.models.openai import OpenAIModel


def test_nvidia_extract_json_clean():
    raw = '{"clips": [{"start_time": 10.0, "end_time": 45.0, "hook_text": "Segredo revelado", "virality_score": 0.9, "why_viral": "Curiosidade alta"}], "summary": "Resumo", "suggested_hashtags": ["#ia"]}'
    parsed = NVIDIALLMService._extract_json(raw)
    assert "clips" in parsed
    assert len(parsed["clips"]) == 1
    assert parsed["clips"][0]["hook_text"] == "Segredo revelado"


def test_nvidia_extract_json_markdown_wrapped():
    raw = 'Aqui está a resposta:\n```json\n{"clips": [{"start_time": 0.0, "end_time": 30.0, "hook_text": "Gancho", "virality_score": 0.85, "why_viral": "Bom"}], "summary": "Video", "suggested_hashtags": []}\n```\nEspero que ajude!'
    parsed = NVIDIALLMService._extract_json(raw)
    assert len(parsed["clips"]) == 1
    assert parsed["clips"][0]["hook_text"] == "Gancho"


@pytest.mark.asyncio
async def test_nvidia_analyze_video_content_mock():
    service = NVIDIALLMService(api_key="nvapi-test-key")
    mock_payload = {
        "clips": [
            {
                "start_time": 12.0,
                "end_time": 55.0,
                "hook_text": "Como a IA pensa",
                "virality_score": 0.95,
                "why_viral": "Fator novidade e alto engajamento",
            }
        ],
        "summary": "Discussão sobre modelos de IA",
        "suggested_hashtags": ["#ia", "#nemotron"],
    }

    with patch.object(service, "_call_nvidia", new_callable=AsyncMock) as mock_call:
        import json
        mock_call.return_value = {"content": json.dumps(mock_payload), "model": "nvidia/llama-3.1-nemotron-70b-instruct"}
        result = await service.analyze_video_content(
            transcript="[00:12 - 00:55] Como a inteligência artificial realmente pensa...",
            video_metadata={"duration": 60, "title": "IA e Futuro"},
        )

        assert "clips" in result
        assert result["clips"][0]["virality_score"] == 0.95
        assert result["clips"][0]["hook_text"] == "Como a IA pensa"
        mock_call.assert_called_once()


def test_split_llm_name_nvidia():
    provider, model = _split_llm_name("nvidia:nvidia/llama-3.1-nemotron-70b-instruct")
    assert provider == "nvidia"
    assert model == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_nvidia_missing_key_error():
    cfg = Config()
    cfg.nvidia_api_key = None
    err = _get_missing_llm_key_error("nvidia:nvidia/llama-3.1-nemotron-70b-instruct", cfg)
    assert err is not None
    assert "NVIDIA_API_KEY is not set" in err

    cfg.nvidia_api_key = "nvapi-fake-key"
    no_err = _get_missing_llm_key_error("nvidia:nvidia/llama-3.1-nemotron-70b-instruct", cfg)
    assert no_err is None


def test_build_transcript_model_nvidia():
    cfg = Config()
    cfg.llm = "nvidia:nvidia/llama-3.1-nemotron-70b-instruct"
    cfg.nvidia_api_key = "nvapi-test-key"
    cfg.nvidia_base_url = "https://integrate.api.nvidia.com/v1"

    model = _build_transcript_model(cfg)
    assert isinstance(model, OpenAIModel)
    assert model.model_name == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_config_infer_default_llm_nvidia():
    cfg = Config()
    cfg.nvidia_api_key = "nvapi-preset"
    cfg.openrouter_api_key = None
    cfg.google_api_key = None
    cfg.openai_api_key = None
    cfg.anthropic_api_key = None
    cfg.zai_api_key = None
    cfg.groq_api_key = None

    inferred = cfg._infer_default_llm()
    assert inferred == "nvidia:moonshotai/kimi-k2.5"
