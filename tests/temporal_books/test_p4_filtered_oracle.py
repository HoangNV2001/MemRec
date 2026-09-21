from src.temporal_books.p4_filtered_oracle import FILTERED_ARM, build_jobs, overlay_filtered_memories, smoke_events


def fixture():
    prepared = {
        "events": [{"user_id": "u", "timestamp": 10, "gold_item_id": "gold", "candidate_item_ids": ["gold", "other"]}],
        "item_info": {"gold": {"base_memory": "gold base"}, "other": {"base_memory": "other base"}},
    }
    frozen = {
        "packet:source": {"value": {"memory": "positive community evidence"}},
        "stage_r:u:10": {"value": {"facets": []}},
    }
    return prepared, frozen


def test_filtered_overlay_only_changes_supported_gold():
    prepared, frozen = fixture()
    event = prepared["events"][0]
    changed = overlay_filtered_memories(event, prepared, {"gold": ["source"]}, frozen, packet_cap=2)
    unchanged = overlay_filtered_memories(event, prepared, {}, frozen, packet_cap=2)
    assert "positive community evidence" in changed["gold"]
    assert changed["other"] == "other base"
    assert unchanged["gold"] == "gold base"


def test_filtered_jobs_have_separate_cache_key_and_deterministic_smoke():
    prepared, frozen = fixture()
    jobs = build_jobs(prepared, {"gold": ["source"]}, frozen, packet_cap=2)
    assert jobs[0][0] == f"rerank:{FILTERED_ARM}:u:10"
    assert smoke_events(prepared, 1) == prepared["events"]
