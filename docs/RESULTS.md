# RESULTS.md — Bảng kết quả đồ án GRPO cho LM_Mem

> Điền dần theo từng milestone. Xem `docs/RL_PLAN.md` §8 cho định nghĩa đầy đủ.

---

## M0 Baselines

Cấu hình: `instructrec-books`, seed 42, `LLM_Rec` = gpt-4o-mini.

| Config | H@1 | H@3 | N@3 | H@5 | N@5 | Chi phí API thực tế |
|---|---|---|---|---|---|---|
| Vanilla LLM | | | | | | |
| MemRec w/o Collab. Read | | | | | | |
| MemRec (full, prompted) | | | | | | |

Warmup cost extrapolation (100 user → 1k user): _(điền sau khi đo)_

---

## M2 Reward Validation

- Spearman ρ (frozen 1.5B ranker vs gpt-4o-mini), n=150 val users: _TBD_
- Validation B (r(thật) > r(user khác) > r(lorem) ≈ r(rỗng)): _TBD_
- Throughput (reward/s @ batch 64): _TBD_

---

## Bảng chính — instructrec-books, test set, `LLM_Rec` = gpt-4o-mini

| Config | H@1 | H@3 | N@3 | H@5 | N@5 | Tokens/query | Δ vs prompted |
|---|---|---|---|---|---|---|---|
| Vanilla LLM | | | | | | | |
| MemRec w/o Collab. Read | | | | | | | |
| MemRec prompted (7B `LM_Mem`) | | | | | | | — |
| MemRec + SFT-4B `LM_Mem` (M3) | | | | | | | |
| **MemRec + GRPO-4B `LM_Mem` (M4)** | | | | | | | |

## Bảng chống hacking

| Test | Gain giữ được | Kết luận |
|---|---|---|
| Reward ranker (1.5B) → gpt-4o-mini | | |
| → vector reranker | | |
| Books → MovieTV (zero-shot) | | |
| Candidate-blind → candidate-visible | | |
