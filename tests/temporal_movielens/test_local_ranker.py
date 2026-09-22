from src.temporal_books.p7_selfhost_local import complete_permutation
from src.temporal_movielens.local_ranker import rerank_jobs, rerank_prompt, stage_jobs, stage_r_prompt


EVENT = {
    "user_id": "u",
    "timestamp": 100,
    "candidate_item_ids": [str(index) for index in range(10)],
    "history": [
        {
            "title": "Example Film (2000)",
            "genres": "Drama|Crime",
            "rating": 4.5,
        }
    ],
}
ITEM_INFO = {
    str(index): {"base_memory": f"Title: Candidate {index}. Genres: Drama."}
    for index in range(10)
}


def test_movie_prompts_use_native_rating_context_not_book_reviews():
    stage_messages, stage_schema = stage_r_prompt(EVENT)
    rerank_messages, rerank_schema = rerank_prompt(EVENT, [], ITEM_INFO)
    text = (stage_messages[0]["content"] + rerank_messages[0]["content"]).lower()
    assert "rating: 4.5/5" in text
    assert "genres: drama, crime" in text
    assert "book recommender" not in text and "reader history" not in text and "prior review" not in text
    assert set(stage_schema) == {"facets"}
    assert rerank_schema["ranking"]["items"]["enum"] == list("ABCDEFGHIJ")


def test_movie_jobs_keep_two_phase_keys_and_complete_permutation():
    stages = {"stage_r:u:100": {"facets": []}}
    first = stage_jobs([EVENT])
    second = rerank_jobs([EVENT], ITEM_INFO, stages)
    assert first[0][0] == "stage_r:u:100"
    assert second[0][0] == "rerank:local:u:100"
    parsed = second[0][4]({"ranking": list("AABCDEFGHI")})
    assert parsed == [str(index) for index in range(10)]
    assert parsed == complete_permutation({"ranking": list("AABCDEFGHI")}, EVENT["candidate_item_ids"])
