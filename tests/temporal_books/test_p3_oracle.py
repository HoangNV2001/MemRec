from src.temporal_books.p3_oracle import overlay_memories, smoke_subset


def test_three_hop_oracle_changes_gold_only():
    event = {"candidate_item_ids": ["gold", "other"], "gold_item_id": "gold"}
    prepared = {
        "item_info": {"gold": {"base_memory": "gold base"}, "other": {"base_memory": "other base"}},
        "origins_by_endpoint_support_gte_threshold": {},
        "origins_by_three_hop_endpoint_support_gte_threshold": {"gold": ["source"]},
    }
    values = overlay_memories(event, prepared, {"source": {"memory": "packet"}}, arm="oracle_three_hop", packet_cap=2)
    assert "packet" in values["gold"]
    assert values["other"] == "other base"


def test_p3_smoke_includes_packets_used_by_both_oracles():
    prepared = {
        "events": [{"user_id": "u", "timestamp": 1, "gold_item_id": "gold", "candidate_item_ids": ["gold"]}],
        "source_packets": [{"source_user_id": value} for value in ("base", "two", "three")],
        "origins_by_endpoint_support_gte_threshold": {"gold": ["two"]},
        "origins_by_three_hop_endpoint_support_gte_threshold": {"gold": ["three"]},
    }
    packets, events = smoke_subset(
        prepared,
        {"smoke_events": 1, "smoke_base_packet_sources": 1, "overlay_packet_cap": 2},
    )
    assert len(events) == 1
    assert {packet["source_user_id"] for packet in packets} >= {"two", "three"}
