import json
import pytest
import caption_agent

class DummyResponse:
    def __init__(self, text):
        self.text = text

class DummyModel:
    def generate_content(self, prompt, generation_config=None, safety_settings=None):
        payload = {"captions": ["one!", "two!"]}
        return DummyResponse(text=json.dumps(payload))

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    monkeypatch.setattr(caption_agent, "MODEL", DummyModel())

def test_generate_captions_for_trend():
    caps = caption_agent.generate_captions_for_trend("anything", n=2)
    assert caps == ["one!", "two!"]
