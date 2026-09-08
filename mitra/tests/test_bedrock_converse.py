"""GPT-5.6 Sol rejects inferenceConfig.temperature; Mode B must retry without it."""


def test_converse_retries_without_temperature(monkeypatch):
    import boto3

    from mitra.eval.bedrock_converse import converse_text

    calls = []

    class FakeClient:
        def converse(self, **kwargs):
            calls.append(dict(kwargs.get("inferenceConfig") or {}))
            cfg = kwargs.get("inferenceConfig") or {}
            if "temperature" in cfg:
                raise RuntimeError(
                    "This model doesn't support the temperature field. Remove temperature"
                )
            return {
                "output": {"message": {"content": [{"text": "अहं अत्र अस्मि।"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 5},
                "stopReason": "end_turn",
            }

    monkeypatch.setattr(boto3, "client", lambda *a, **k: FakeClient())

    out = converse_text(
        model_id="us.openai.gpt-5.6-sol",
        user="ping",
        system="sys",
        region="us-west-2",
        temperature=0.3,
        max_tokens=16,
    )
    assert out["text"] == "अहं अत्र अस्मि।"
    assert any("temperature" in c for c in calls)
    assert any("temperature" not in c for c in calls)
