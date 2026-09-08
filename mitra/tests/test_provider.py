from mitra.agent.errors import ProviderError
from mitra.agent.provider import (
    SUPPORTED_PROVIDERS,
    describe_llm,
    fallback_config,
    map_provider_exception,
    provider_name,
    resolve_region,
)


def test_default_provider_is_ollama():
    assert provider_name({}) == "ollama"
    assert "bedrock" in SUPPORTED_PROVIDERS


def test_describe_bedrock_never_claims_ollama_contact(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    info = describe_llm({"provider": "bedrock", "id": "us.amazon.nova-pro-v1:0"})
    assert info["provider"] == "bedrock"
    assert info["model_id"] == "us.amazon.nova-pro-v1:0"
    assert info["region"] == "us-west-2"
    assert info["ollama_contacted"] is False


def test_region_from_env_not_hardcoded(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    assert resolve_region({}) == "eu-west-1"
    assert resolve_region({"region": "ap-south-1"}) == "ap-south-1"


def test_fallback_off_by_default():
    assert fallback_config({"provider": "bedrock", "id": "x"}) is None


def test_fallback_requires_ids():
    try:
        fallback_config({"fallback": {"enabled": True}})
    except ProviderError as e:
        assert e.code == "fallback_misconfigured"
    else:
        raise AssertionError("expected ProviderError")


def test_map_access_denied_is_actionable():
    class Fake(Exception):
        response = {"Error": {"Code": "AccessDeniedException", "Message": "no"}}

    err = map_provider_exception(
        Fake("not authorized"), provider="bedrock",
        model_id="us.amazon.nova-pro-v1:0", region="us-west-2",
    )
    assert err.code == "access_denied"
    assert "InvokeModel" in err.actionable
    assert "nova-pro" in str(err)


def test_map_timeout():
    class ReadTimeoutError(Exception):
        pass

    err = map_provider_exception(
        ReadTimeoutError("timed out"), provider="bedrock",
        model_id="x", region="us-west-2",
    )
    assert err.code == "timeout"


def test_make_model_unknown_provider():
    from mitra.agent.provider import make_model

    try:
        make_model({"provider": "openai", "id": "gpt"})
    except ProviderError as e:
        assert e.code == "unknown_provider"
    else:
        raise AssertionError("expected ProviderError")


def test_bedrock_requires_model_id():
    from mitra.agent.provider import make_model

    try:
        make_model({"provider": "bedrock"})
    except ProviderError as e:
        assert e.code == "missing_model_id"
    else:
        raise AssertionError("expected ProviderError")


def test_mitra_agent_make_model_delegates_to_provider(monkeypatch):
    from mitra.agent.agent import MitraAgent

    calls = []

    def fake(cfg):
        calls.append(cfg["provider"])
        return object()

    monkeypatch.setattr("mitra.agent.provider.make_model", fake)
    assert MitraAgent._make_model({"provider": "bedrock", "id": "x"}) is not None
    assert calls == ["bedrock"]
