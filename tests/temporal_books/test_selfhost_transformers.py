import pytest

from src.models.selfhost_transformers import (
    extract_json_object,
    extract_json_object_with_repairs,
    validate_hard_memory_fraction,
    validate_json_schema,
)


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


def test_json_repair_is_limited_to_apostrophe_escape_and_extra_closing_brace():
    value, repairs = extract_json_object_with_repairs(r'''{"memory":"reader\'s preference","facets":[]} }''')
    assert value == {"memory": "reader's preference", "facets": []}
    assert repairs == ["unescape_apostrophe", "drop_extra_closing_brace:1"]
    with pytest.raises(ValueError):
        extract_json_object('{"memory":"missing comma" "facets":[]}')


def test_memory_fraction_allows_one_full_visible_gpu_only():
    validate_hard_memory_fraction(0.25)
    validate_hard_memory_fraction(1.0)
    with pytest.raises(ValueError):
        validate_hard_memory_fraction(0.0)
    with pytest.raises(ValueError):
        validate_hard_memory_fraction(1.01)
