import base64
from unittest.mock import Mock

import pytest

from audio_evals.models.TTS.seed_audio import SeedAudio


@pytest.fixture
def model():
    return SeedAudio(
        model_name="seed-audio-1.0",
        api_key="test-key",
        sample_params={"audio_config": {"format": "wav", "sample_rate": 24000}},
    )


def test_builds_plain_tts_request(model):
    request = model._build_request(
        {"text": "Hello world"},
        {"audio_config": {"format": "wav", "sample_rate": 24000}},
    )

    assert request["model"] == "seed-audio-1.0"
    assert request["text_prompt"] == "Hello world"
    assert "references" not in request


def test_builds_voice_clone_request(model, tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")

    request = model._build_request(
        {
            "text": "你好，世界",
            "prompt_audio": str(reference),
            "prompt_text": "参考音频转写",
        },
        {},
    )

    assert "@音频1" in request["text_prompt"]
    assert "你好，世界" in request["text_prompt"]
    assert request["references"] == [
        {"audio_data": base64.b64encode(b"reference audio").decode("ascii")}
    ]
    assert "prompt_text" not in request


def test_builds_instruction_tts_requests(model, tmp_path):
    request = model._build_request(
        {"text": "Good morning", "instruction": "Speak cheerfully."}, {}
    )
    assert "Speak cheerfully." in request["text_prompt"]
    assert "Good morning" in request["text_prompt"]
    assert "references" not in request

    reference = tmp_path / "reference.mp3"
    reference.write_bytes(b"reference audio")
    cloned_request = model._build_request(
        {
            "text": "Good morning",
            "instruction": "Speak cheerfully.",
            "prompt_audio": str(reference),
        },
        {},
    )
    assert "@音频1" in cloned_request["text_prompt"]
    assert "Speak cheerfully." in cloned_request["text_prompt"]
    assert "references" in cloned_request


def test_inference_downloads_returned_audio(model, tmp_path):
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = {"url": "https://example.com/audio.wav"}

    audio_response = Mock()
    audio_response.content = b"generated audio"
    audio_response.raise_for_status.return_value = None

    model.session.post = Mock(return_value=response)
    model.session.get = Mock(return_value=audio_response)

    output_path = model.inference({"text": "Hello world"})

    with open(output_path, "rb") as output:
        assert output.read() == b"generated audio"

    post_call = model.session.post.call_args
    assert post_call.kwargs["headers"]["X-Api-Key"] == "test-key"
    assert post_call.kwargs["json"]["text_prompt"] == "Hello world"
    assert post_call.kwargs["json"]["audio_config"]["format"] == "wav"


def test_inference_decodes_inline_audio(model):
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = {
        "code": 0,
        "audio": base64.b64encode(b"generated audio").decode("ascii"),
    }
    model.session.post = Mock(return_value=response)

    output_path = model.inference({"text": "Hello world"})

    with open(output_path, "rb") as output:
        assert output.read() == b"generated audio"


def test_rejects_conflicting_voice_references(model, tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")

    with pytest.raises(ValueError, match="either `prompt_audio` or `speaker`"):
        model._build_request(
            {
                "text": "Hello",
                "prompt_audio": str(reference),
                "speaker": "speaker-id",
            },
            {},
        )
