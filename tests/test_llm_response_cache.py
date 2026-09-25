from types import SimpleNamespace

from src.models.llm_client import LLMClient, RequestBudget


def fake_client(monkeypatch, cache_path, namespace, read='1'):
    monkeypatch.setenv('MEMREC_LLM_CACHE_DB', str(cache_path))
    monkeypatch.setenv('MEMREC_LLM_CACHE_NAMESPACE', namespace)
    monkeypatch.setenv('MEMREC_LLM_CACHE_READ', read)
    llm = LLMClient(api_endpoint='http://127.0.0.1:1/v1', api_key='local-placeholder',
                    model='pinned-local-model', provider_name='openai', sdk_max_retries=0)
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    return llm, calls


def test_exact_cache_reuses_only_same_request_and_namespace(monkeypatch, tmp_path):
    path = tmp_path / 'responses.sqlite'
    messages = [{'role': 'user', 'content': 'same prompt'}]
    first, calls = fake_client(monkeypatch, path, 'revision-a')
    first.request_budget = RequestBudget(2)
    assert first.generate(messages, temperature=0) == '{"ok":true}'
    assert first.generate(messages, temperature=0) == '{"ok":true}'
    assert len(calls) == first.total_physical_requests == first.request_budget.used == 1
    assert first.get_token_stats()['total_cache_hits'] == 1
    assert first.generate(messages, temperature=0.5) == '{"ok":true}'
    assert len(calls) == 2

    second, second_calls = fake_client(monkeypatch, path, 'revision-b')
    assert second.generate(messages, temperature=0) == '{"ok":true}'
    assert len(second_calls) == 1


def test_smoke_writes_cache_but_does_not_read_it(monkeypatch, tmp_path):
    path = tmp_path / 'responses.sqlite'
    messages = [{'role': 'user', 'content': 'warm-up prompt'}]
    smoke, smoke_calls = fake_client(monkeypatch, path, 'revision-a', read='0')
    smoke.generate(messages, temperature=0)
    smoke.generate(messages, temperature=0)
    assert len(smoke_calls) == 2
    full, full_calls = fake_client(monkeypatch, path, 'revision-a', read='1')
    assert full.generate(messages, temperature=0) == '{"ok":true}'
    assert len(full_calls) == 0
    assert full.total_cache_hits == 1
