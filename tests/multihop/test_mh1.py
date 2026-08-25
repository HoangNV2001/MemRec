from copy import deepcopy

from src.multihop import mh0, mh1


def stage_context():
    neighbors_text = "\n".join(
        [
            "**Collaborative Neighbors:**",
            "1. [Item-10] Base item (score=1.000)",
            "2. [User-2] (overlap_score=0.500) - Recent: Remote book",
        ]
    )
    prompt = mh0.build_prompt(1, "likes mysteries", neighbors_text, n_facets=7)
    return {
        "user_id": 1,
        "user_memory": "likes mysteries",
        "neighbors_text": neighbors_text,
        "neighbor_snippets": {
            "Item-10": "Base item (score=1.000)",
            "User-2": "(overlap_score=0.500) - Recent: Remote book",
        },
        "selected_node_ids": ["Item-10", "User-2"],
        "K_u": 2,
        "T_u": 200,
        "prompt": prompt,
        "prompt_sha256": mh0.canonical_hash(prompt),
    }


def tiny_topology():
    return mh1.Topology(
        user_items={1: (10, 11), 2: (10, 20, 21), 3: (20,), 4: (21,)},
        item_users={10: (1, 2), 11: (1,), 20: (2, 3), 21: (2, 4)},
    )


def render_pool():
    topology = tiny_topology()
    raw = mh1.build_raw_pool(
        stage_context(),
        topology,
        max_remote_items=8,
        max_remote_users=8,
        max_remote_users_per_item=8,
    )
    rendered = mh1.render_remote_pool(
        raw,
        topology,
        {
            20: {"title": "Remote mystery", "description": "A mystery description"},
            21: {"title": "Remote romance", "description": "A romance description"},
        },
    )
    return raw, rendered


def test_raw_pool_has_valid_structural_paths_and_excludes_one_hop_nodes():
    raw, _ = render_pool()
    ids = {node["node_id"] for node in raw}

    assert {"Item-20", "Item-21", "User-3", "User-4"}.issubset(ids)
    assert "Item-10" not in ids
    assert "User-2" not in ids
    for node in raw:
        path = node["witness_path"]
        assert path[0] == "User-1"
        assert path[1] == "Item-10"
        assert path[2] == "User-2"
        assert path[-1] == node["node_id"]


def test_validation_bundles_are_deterministic_and_stage_r_is_candidate_blind():
    _, pool = render_pool()
    context = stage_context()
    control = {
        "stage_r_context": context,
        "ranking_context": {
            "candidate_order_sha256": "candidate-hash",
            "candidates": [100 + index for index in range(10)],
            "gold_item_id": 100,
            "candidate_titles": {"100": "A target title that does not occur in the snippets"},
        },
    }

    first = mh1.build_validation_bundles(
        control, pool, seed=42, quotas=(2, 4, 6), oracle_bundles_per_quota=4, n_facets=7
    )
    second = mh1.build_validation_bundles(
        deepcopy(control), pool, seed=42, quotas=(2, 4, 6), oracle_bundles_per_quota=4, n_facets=7
    )

    assert mh0.canonical_hash(first) == mh0.canonical_hash(second)
    assert len(first["oracle_two_hop"]) == 12
    assert set(first["naive_two_hop"]) == {"2", "4", "6"}
    assert first["eligibility"]["eligible_all_arms"]
    for bundle in [*first["naive_two_hop"].values(), *first["oracle_two_hop"]]:
        stage_r = bundle["stage_r"]
        assert bundle["remote_actual"] <= bundle["quota_requested"]
        assert len(stage_r["selected_node_ids"]) == context["K_u"]
        assert stage_r["T_actual"] <= context["T_u"]
        assert not set(stage_r).intersection(mh1._FORBIDDEN_STAGE_R_FIELDS)


def test_ranking_leakage_is_a_post_construction_audit_not_a_pool_filter():
    _, pool = render_pool()
    control = {
        "stage_r_context": stage_context(),
        "ranking_context": {
            "candidate_order_sha256": "candidate-hash",
            "candidates": [20, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "gold_item_id": 100,
            "candidate_titles": {"100": "A target title that does not occur in the snippets"},
        },
    }

    bundles = mh1.build_validation_bundles(
        control, pool, seed=42, quotas=(2,), oracle_bundles_per_quota=1, n_facets=7
    )

    assert any(node["node_id"] == "Item-20" for node in pool)
    assert not bundles["eligibility"]["eligible_all_arms"]
    assert any("candidate id 20 in prompt" in reason for reason in bundles["eligibility"]["reasons"])


def test_budget_overflow_falls_back_to_one_hop_without_changing_k():
    context = stage_context()
    base = mh1.baseline_nodes(context)
    context["T_u"] = mh1.estimate_tokens(context["neighbors_text"])
    remote = [
        {
            "node_id": "Item-99",
            "node_type": "item",
            "id": 99,
            "snippet": "[Item-99] " + "x" * 500,
            "path_strength": 1.0,
            "witness_path": ["User-1", "Item-10", "User-2", "Item-99"],
        }
    ]

    bundle = mh1.materialize_bundle(
        user_id=1,
        arm="naive_two_hop",
        quota=1,
        variant=None,
        stage_r_context=context,
        base_nodes=base,
        requested_remote=remote,
        n_facets=7,
    )

    assert bundle["remote_actual"] == 0
    assert bundle["remote_shortfall_budget"] == 1
    assert len(bundle["stage_r"]["selected_node_ids"]) == context["K_u"]
    assert bundle["stage_r"]["T_actual"] <= context["T_u"]
