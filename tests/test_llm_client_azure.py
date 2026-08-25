import importlib


def test_azure_environment_uses_prefixed_model_as_deployment(monkeypatch):
    module = importlib.import_module("src.models.llm_client")
    captured = {}

    class FakeAzureOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "AzureOpenAI", FakeAzureOpenAI)
    monkeypatch.setenv("LLM__BASE_URL", "https://example.openai.azure.com/")
    monkeypatch.setenv("LLM__API_KEY", "test-key")
    monkeypatch.setenv("LLM__API_VERSION", "2024-05-01-preview")
    monkeypatch.setenv("LLM__MODEL_NAME", "azure/gpt-5.4-mini")

    client = module.LLMClient(provider_name="azure_openai")

    assert client.model == "azure/gpt-5.4-mini"
    assert client.request_model == "gpt-5.4-mini"
    assert client.api_version == "2024-05-01-preview"
    assert captured == {
        "azure_endpoint": "https://example.openai.azure.com/",
        "api_key": "test-key",
        "api_version": "2024-05-01-preview",
    }


def test_prefixed_gpt5_uses_restricted_sampling_rules():
    module = importlib.import_module("src.models.llm_client")

    assert module._restricted_sampling_params("azure/gpt-5.4-mini")
    assert module._restricted_sampling_params("gpt-5.4-mini")
    assert not module._restricted_sampling_params("gpt-4o-mini")

