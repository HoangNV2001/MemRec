# RESULTS.md — Bảng kết quả đồ án GRPO cho LM_Mem

> Điền dần theo từng milestone. Xem `docs/RL_PLAN.md` §8 cho định nghĩa đầy đủ.

---

## M0 Baselines

Cấu hình: `instructrec-books`, 1000 user (`data/eval_user_samples/eval_user_sample_1k_instructrec-books.json`), seed 42, `LLM_Rec` = gpt-4o-mini, `n_eval_candidates=10`.

| Config | H@1 | H@3 | N@3 | H@5 | N@5 |
|---|---|---|---|---|---|
| Vanilla LLM (đã fix vanilla_mode prompt) | 0.425 | 0.614 | 0.534 | 0.731 | 0.582 |
| MemRec w/o Collab. Read (đã fix prompt + eval-time write) | 0.436 | 0.579 | 0.518 | 0.705 | 0.570 |
| MemRec (full, prompted) | 0.510 | 0.709 | 0.625 | 0.808 | 0.666 |

**Trạng thái DoD (§7 RL_PLAN.md):** đạt một phần. H@1 đúng thứ tự (MemRec > w/o Collab. Read > Vanilla). H@3/N@3/H@5/N@5 **chưa** đúng thứ tự (w/o Collab. Read < Vanilla) — nghi do candidate negatives được sample với RNG seed theo thread ID (không theo user_id), nên 3 config không được đánh giá trên cùng bộ đề và kết quả không reproduce được giữa các lần chạy. Quyết định có chủ đích: chấp nhận, không sửa RNG ngay, xem chi tiết trong `docs/RL_PLAN.md` mục M0 và `docs/PROGRESS.md`.

Warmup cost extrapolation (100 user, quan sát thực tế trong `results/m0_warmup_100users.log`): 930,926 token tổng (main LLM + reranker) cho 100 user → ngoại suy ~9.3M token cho 1000 user. Chi phí $ thực tế: xem OpenAI usage dashboard của tài khoản đã dùng (không ước tính lại giá ở đây vì giá gpt-4o-mini có thể đã thay đổi).

### Các bug đã tìm thấy và sửa trong quá trình chạy M0

1. `src/train/trainer_memrec.py`: Stage-ReRank LLM client hardcode `provider_name='azure_openai'` bất kể config — sửa để dùng chung provider với Stage-R/W (cần thiết vì dự án dùng key OpenAI thường, không phải Azure).
2. `src/models/reranker_llm.py` + `src/models/memrec_agent.py`: tham số `vanilla_mode` tồn tại nhưng là dead code (không bao giờ được truyền `True`). Đã wire: `vanilla_mode = not warmup_enabled and not enable_stage_r`, chỉ áp dụng cho baseline Vanilla LLM thật sự.
3. `src/models/reranker_llm.py`: prompt "MemRec mode" luôn khẳng định "we have identified the following preference patterns" ngay cả khi Stage-R bị tắt và facets rỗng — gây prompt tự mâu thuẫn, đo được ảnh hưởng xấu tới điểm reranker qua `llm_conversations.jsonl`. Sửa: chỉ render đoạn này khi facets không rỗng.
4. `configs/memrec_instructrec-books_1k_no_collab_read.yaml`: thiếu `enable_stage_w: false` cho eval loop → ground-truth feedback bị ghi vào item memory dùng chung giữa các user ngay trong lúc eval (cùng loại bug đã bắt được ở bản vanilla đầu tiên, xem `results/m0_vanilla_1k_leaky_writes_DO_NOT_USE`). Đã thêm; warmup vẫn chạy bình thường (code path riêng, không bị gate bởi cờ này).
5. **[Chưa sửa, đã ghi nhận]** `_evaluate_single_user` sample negatives bằng `RandomState(seed=hash(thread.ident))` thay vì seed theo `user_id` — không reproducible, 3 config không dùng chung candidate set. Xem quyết định ở trên.
6. Gotcha (không phải bug, nhưng dễ nhầm): khi config có `eval_user_list`, nó **luôn** override `--n_eval_users` (`trainer_memrec.py:411`, "Priority: eval_user_list > n_eval_users > all users"). Muốn test nhanh trên vài user với các config 1k này phải tạo `eval_user_list` riêng, không dùng `--n_eval_users`.

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
