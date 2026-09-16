import base64
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel


DEFAULT_API_URL = "https://openspeech.bytedance.com/api/v3/tts/create"
SUPPORTED_REFERENCE_SUFFIXES = {".wav", ".mp3", ".pcm", ".ogg", ".opus"}
OUTPUT_SUFFIXES = {
    "mp3": ".mp3",
    "ogg_opus": ".ogg",
    "opus": ".opus",
    "pcm": ".pcm",
    "wav": ".wav",
}


class SeedAudio(APIModel):
    """Volcengine Seed Audio HTTP API model.

    The rendered prompt determines the generation mode:
    - ``text``: text-to-speech
    - ``text`` + ``prompt_audio``: voice cloning
    - ``text`` + ``instruction``: instruction-following TTS
    - all three fields: voice cloning with an instruction
    """

    def __init__(
        self,
        model_name: str = "seed-audio-1.0",
        api_key: Optional[str] = None,
        api_url: str = DEFAULT_API_URL,
        request_timeout: int = 300,
        sample_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(is_chat=False, sample_params=sample_params)
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("SEED_AUDIO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Seed Audio API key is required. Set SEED_AUDIO_API_KEY or pass api_key."
            )
        self.api_url = api_url
        self.request_timeout = request_timeout
        self.session = requests.Session()

    @staticmethod
    def _prompt_to_payload(prompt: PromptStruct) -> Dict[str, Any]:
        if isinstance(prompt, dict):
            return dict(prompt)
        if isinstance(prompt, str):
            return {"text": prompt}
        raise TypeError("Seed Audio expects a string or dictionary prompt.")

    @staticmethod
    def _encode_reference_audio(audio_path: str) -> str:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        if path.suffix.lower() not in SUPPORTED_REFERENCE_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_REFERENCE_SUFFIXES))
            raise ValueError(
                f"Unsupported reference audio format {path.suffix!r}; "
                f"expected one of: {supported}"
            )
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Seed Audio reference audio must not exceed 10 MB.")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    @staticmethod
    def _compose_text_prompt(
        text: str, instruction: Optional[str], has_reference_audio: bool
    ) -> str:
        if instruction and has_reference_audio:
            return (
                f"参考@音频1的音色、口音和说话风格。{instruction.strip()}\n"
                f'朗读内容："{text}"'
            )
        if has_reference_audio:
            return f'请使用与@音频1相同的音色、口音和说话风格朗读："{text}"'
        if instruction:
            return f'{instruction.strip()}\n朗读内容："{text}"'
        return text

    def _build_request(
        self, prompt: PromptStruct, inference_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        values = self._prompt_to_payload(prompt)
        values.update(inference_params)

        text = str(values.pop("text", "")).strip()
        if not text:
            raise ValueError("Seed Audio requires non-empty `text`.")

        instruction = values.pop("instruction", None)
        prompt_audio = values.pop("prompt_audio", None)
        # The Seed Audio API does not require a transcript of the reference audio.
        values.pop("prompt_text", None)

        explicit_text_prompt = values.pop("text_prompt", None)
        references = values.pop("references", None)
        speaker = values.pop("speaker", None)

        if prompt_audio and references:
            raise ValueError(
                "Use either `prompt_audio` or `references`, not both, for Seed Audio."
            )
        if prompt_audio and speaker:
            raise ValueError(
                "Use either `prompt_audio` or `speaker`, not both, for Seed Audio."
            )

        has_reference_audio = bool(prompt_audio or references or speaker)
        request: Dict[str, Any] = {
            "model": self.model_name,
            "text_prompt": explicit_text_prompt
            or self._compose_text_prompt(text, instruction, has_reference_audio),
            **values,
        }

        if prompt_audio:
            request["references"] = [
                {"audio_data": self._encode_reference_audio(str(prompt_audio))}
            ]
        elif references:
            request["references"] = references
        elif speaker:
            request["references"] = [{"speaker": speaker}]

        return request

    @staticmethod
    def _output_suffix(request: Dict[str, Any]) -> str:
        audio_config = request.get("audio_config") or {}
        output_format = str(audio_config.get("format", "wav")).lower()
        return OUTPUT_SUFFIXES.get(output_format, f".{output_format}")

    def _write_audio_response(self, response_data: Dict[str, Any], suffix: str) -> str:
        audio_data = response_data.get("audio")
        audio_url = response_data.get("url")

        if audio_data:
            try:
                content = base64.b64decode(audio_data, validate=True)
            except (ValueError, TypeError) as error:
                raise RuntimeError(
                    "Seed Audio returned invalid base64 audio data."
                ) from error
        elif audio_url:
            audio_response = self.session.get(audio_url, timeout=self.request_timeout)
            audio_response.raise_for_status()
            content = audio_response.content
        else:
            raise RuntimeError(
                "Seed Audio response contains neither `audio` nor `url`."
            )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as output:
            output.write(content)
            return output.name

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        request = self._build_request(prompt, kwargs)
        response = self.session.post(
            self.api_url,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self.api_key,
                "X-Api-Request-Id": str(uuid.uuid4()),
            },
            json=request,
            timeout=self.request_timeout,
        )

        try:
            response_data = response.json()
        except ValueError as error:
            raise RuntimeError(
                f"Seed Audio returned a non-JSON response (HTTP {response.status_code})."
            ) from error

        if not response.ok:
            message = response_data.get("message", response.text)
            raise RuntimeError(
                f"Seed Audio request failed (HTTP {response.status_code}): {message}"
            )

        code = response_data.get("code")
        if code not in (None, 0):
            raise RuntimeError(
                f"Seed Audio request failed (code={code}): "
                f"{response_data.get('message', 'unknown error')}"
            )

        return self._write_audio_response(response_data, self._output_suffix(request))
