import pytest

from src.models.selfhost_transformers import extract_json_object, validate_json_schema


def test_extract_json_object_accepts_fenced_compact_output():
    assert extract_json_object('```json\n{"ranking":["A","B"]}\n```') == {"ranking": ["A", "B"]}


def test_local_schema_rejects_extra_or_wrong_typed_fields():
    schema = {
        "type": "object",
        "properties": {"ranking": {"type": "array", "items": {"type": "string"}}},
        "required": ["ranking"],
        "additionalProperties": False,
    }
    validate_json_schema({"ranking": ["A"]}, schema)
    with pytest.raises(ValueError):
        validate_json_schema({"ranking": [1]}, schema)
    with pytest.raises(ValueError):
        validate_json_schema({"ranking": ["A"], "rationale": "extra"}, schema)
