import whisper
from faster_whisper import WhisperModel
import torch
from typing import Optional, Literal

class WhisperTranscriptionService:
    """
    Serviço de transcrição usando OpenAI Whisper ou Faster-Whisper
    """
    
    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "small",
        use_faster: bool = True,
        device: Optional[str] = None,
        compute_type: str = "int8"
    ):
        self.model_size = model_size
        self.use_faster = use_faster
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # CPU does not support float16 efficiently in ctranslate2
        if self.device == "cpu" and compute_type == "float16":
            self.compute_type = "int8"
        else:
            self.compute_type = compute_type
            
        if use_faster:
            # Faster-Whisper: 4x mais rápido com mesma precisão
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        else:
            # Whisper padrão OpenAI
            self.model = whisper.load_model(model_size).to(self.device)
    
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> dict:
        """
        Transcreve áudio com timestamps precisos para legendas
        """
        if self.use_faster:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=True,
                vad_filter=True  # Filtra silêncios
            )
            
            result_segments = []
            for segment in segments:
                words = []
                if segment.words:
                    for word in segment.words:
                        words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end
                        })
                
                result_segments.append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": words
                })
            
            return {
                "text": " ".join([str(s["text"]) for s in result_segments]),
                "segments": result_segments,
                "language": info.language,
                "duration": info.duration
            }
        else:
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task
            )
            return result
