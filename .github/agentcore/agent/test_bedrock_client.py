"""Unit tests for the provider-aware BedrockClient.

Runs offline: a fake ``bedrock-runtime`` client is injected so no AWS call is
made. Exercises model-id validation, provider dispatch, the OpenAI temperature
rule, and response parsing for both schemas.

    python -m unittest test_bedrock_client
"""

from __future__ import annotations

import io
import json
import unittest

import bedrock_client
from bedrock_client import BedrockClient


class _FakeBody:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw


class _FakeClient:
    """Captures the invoke_model request and returns a canned payload."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_kwargs: dict = {}

    def invoke_model(self, **kwargs):
        self.last_kwargs = kwargs
        return {"body": _FakeBody(self._payload)}


def _make_client(model_id: str, payload: dict) -> tuple[BedrockClient, _FakeClient]:
    client = BedrockClient.__new__(BedrockClient)  # bypass __init__/boto3
    client.model_id = model_id
    client.provider = bedrock_client._provider_of(model_id)
    client.region_name = "us-east-1"
    fake = _FakeClient(payload)
    client._client = fake
    return client, fake


class ModelValidationTests(unittest.TestCase):
    def test_default_is_claude_sonnet_5_5(self):
        self.assertEqual(
            bedrock_client.DEFAULT_MODEL_ID, "global.anthropic.claude-sonnet-5-5"
        )

    def test_unknown_model_rejected_naming_allowed_set(self):
        with self.assertRaises(ValueError) as ctx:
            BedrockClient(model_id="global.anthropic.claude-opus-5")
        msg = str(ctx.exception)
        self.assertIn("global.anthropic.claude-sonnet-5-5", msg)
        self.assertIn("global.openai.gpt-5.6-sol", msg)
        self.assertIn("global.openai.gpt-6-astra", msg)

    def test_provider_classification(self):
        self.assertEqual(
            bedrock_client._provider_of("global.anthropic.claude-sonnet-5-5"),
            "anthropic",
        )
        self.assertEqual(
            bedrock_client._provider_of("global.openai.gpt-5.6-sol"), "openai"
        )
        self.assertEqual(
            bedrock_client._provider_of("anthropic.claude-sonnet-5-5"), "anthropic"
        )


class AnthropicPathTests(unittest.TestCase):
    def test_request_and_response(self):
        payload = {"content": [{"type": "text", "text": "hello world"}]}
        client, fake = _make_client("global.anthropic.claude-sonnet-5-5", payload)
        out = client.invoke("hi", system="be terse", temperature=0.2)
        self.assertEqual(out, "hello world")
        body = json.loads(fake.last_kwargs["body"])
        self.assertEqual(body["anthropic_version"], bedrock_client.ANTHROPIC_VERSION)
        self.assertEqual(body["system"], "be terse")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_non_default_temperature_is_dropped(self):
        # Bedrock's newer Anthropic models reject a non-default temperature
        # ("`temperature` is deprecated for this model"); the client must omit
        # it rather than send the decomposer's 0.2 default.
        payload = {"content": [{"type": "text", "text": "x"}]}
        client, fake = _make_client("global.anthropic.claude-sonnet-5-5", payload)
        client.invoke("hi", temperature=0.2)
        body = json.loads(fake.last_kwargs["body"])
        self.assertNotIn("temperature", body)

    def test_default_temperature_is_forwarded(self):
        payload = {"content": [{"type": "text", "text": "x"}]}
        client, fake = _make_client("global.anthropic.claude-sonnet-5-5", payload)
        client.invoke("hi", temperature=1.0)
        body = json.loads(fake.last_kwargs["body"])
        self.assertEqual(body["temperature"], 1.0)


class OpenAIPathTests(unittest.TestCase):
    def _payload(self, text: str) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def test_response_parsing(self):
        client, _ = _make_client("global.openai.gpt-5.6-sol", self._payload("OK"))
        self.assertEqual(client.invoke("hi"), "OK")

    def test_uses_max_completion_tokens_and_system_role(self):
        client, fake = _make_client(
            "global.openai.gpt-6-astra", self._payload("x")
        )
        client.invoke("hi", system="sys", max_tokens=123)
        body = json.loads(fake.last_kwargs["body"])
        self.assertEqual(body["max_completion_tokens"], 123)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(
            body["messages"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_non_default_temperature_is_dropped(self):
        # Bedrock's OpenAI models reject non-default temperature; the client
        # must omit it rather than send 0.2 (the decomposer's default).
        client, fake = _make_client("global.openai.gpt-5.6-sol", self._payload("x"))
        client.invoke("hi", temperature=0.2)
        body = json.loads(fake.last_kwargs["body"])
        self.assertNotIn("temperature", body)

    def test_default_temperature_is_forwarded(self):
        client, fake = _make_client("global.openai.gpt-5.6-sol", self._payload("x"))
        client.invoke("hi", temperature=1.0)
        body = json.loads(fake.last_kwargs["body"])
        self.assertEqual(body["temperature"], 1.0)

    def test_content_parts_array_is_parsed(self):
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": [{"text": "a"}, {"text": "b"}]}}
            ]
        }
        client, _ = _make_client("global.openai.gpt-5.6-sol", payload)
        self.assertEqual(client.invoke("hi"), "ab")


class TruncationGuardTests(unittest.TestCase):
    """A response capped at max_tokens must fail loudly, not return a partial body.

    This is the issue #13 failure: an incomplete JSON object came back and the
    downstream json.loads died cryptically. The client now raises
    TruncatedResponseError with the real cause instead.
    """

    def test_default_max_tokens_is_well_above_old_8192(self):
        # Regression guard: the old 8192 default truncated large multi-file
        # sub-tasks. Keep the default high.
        self.assertGreaterEqual(bedrock_client.DEFAULT_MAX_TOKENS, 32768)

    def test_anthropic_stop_reason_max_tokens_raises(self):
        payload = {
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": '{"files": {"a": "partial'}],
        }
        client, _ = _make_client("global.anthropic.claude-sonnet-5-5", payload)
        with self.assertRaises(bedrock_client.TruncatedResponseError) as ctx:
            client.invoke("hi", max_tokens=8192)
        msg = str(ctx.exception)
        self.assertIn("truncated", msg.lower())
        self.assertIn("8192", msg)

    def test_anthropic_end_turn_does_not_raise(self):
        payload = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "complete"}],
        }
        client, _ = _make_client("global.anthropic.claude-sonnet-5-5", payload)
        self.assertEqual(client.invoke("hi"), "complete")

    def test_anthropic_missing_stop_reason_does_not_raise(self):
        # Older/fake payloads without stop_reason must still parse (the existing
        # tests rely on this).
        payload = {"content": [{"type": "text", "text": "ok"}]}
        client, _ = _make_client("global.anthropic.claude-sonnet-5-5", payload)
        self.assertEqual(client.invoke("hi"), "ok")

    def test_openai_finish_reason_length_raises(self):
        payload = {
            "choices": [
                {"finish_reason": "length", "message": {"content": '{"files": {"a": "part'}}
            ]
        }
        client, _ = _make_client("global.openai.gpt-5.6-sol", payload)
        with self.assertRaises(bedrock_client.TruncatedResponseError) as ctx:
            client.invoke("hi", max_tokens=8192)
        self.assertIn("truncated", str(ctx.exception).lower())

    def test_openai_finish_reason_stop_does_not_raise(self):
        payload = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "done"}}
            ]
        }
        client, _ = _make_client("global.openai.gpt-5.6-sol", payload)
        self.assertEqual(client.invoke("hi"), "done")


if __name__ == "__main__":
    unittest.main()
