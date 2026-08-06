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

## M1 Frozen Environment

Snapshot: `data/rl/graph_snapshot_books.json` (17.4 MB) — 2350 user, 40 080 item memory, k=16, seed 42, candidate-blind.
Build: `bash scripts/rl/00_build_snapshot.sh` → `bash scripts/rl/01_build_dataset.sh`.

| Split | Users | Prompt (~token, median) | `M_u` có nội dung | Neighbor/user (median) | \|H_u\| (median) |
|---|---|---|---|---|---|
| train | 1 185 | 1 019 | 1 184 | 16 | 14 |
| val | 149 | 1 031 | 149 | 16 | 14 |
| test | 993 | 1 021 | 992 | 16 | 14 |

**Chi phí warmup (thực đo, không ước tính):** 2350 user, 24 worker, **26.6 phút** wall-clock trên CPU.
Stage-R/W 6 907 837 in + 1 804 548 out (4697 request) · Stage-ReRank 2 634 326 in + 1 063 022 out (2350 request).
Tổng **9.54M input + 2.87M output** → ≈ **$3.15** với đơn giá gpt-4o-mini $0.15/$0.60 per 1M. **0 GPU-hour.**

Warmup thành công 2347/2350 user; 3 user hỏng vì reranker trả về danh sách thiếu target item (`target dropped by reranker`) → không có `M_u`, vẫn giữ trong split (prompt hiển thị "No personal memory recorded yet"), phản ánh đúng ca user lạnh.

### Kiểm tra DoD

`pytest tests/rl/` → **78 pass** (26 s, CPU, không API).

| Kiểm tra | Kết quả |
|---|---|
| Split user-disjoint (assert bằng code) | pass |
| `Item-<gold_id>` xuất hiện trong `prompt` | **0** / 2327 (khớp biên từ, `Item-2125` không khớp `Item-21254`) |
| Tên gold item xuất hiện trong `prompt` | **0** / 2327 sau khi lọc |
| `Item-<candidate_id>` bất kỳ trong `prompt` (candidate-blind §5.3) | **0** / 2327 |
| Instruction InstructRec rò vào `prompt` | **0** / 2327 |
| Parser chịu được output méo | 20/20 ca viết tay |
| Candidate tái tạo được từ dataset (chống bug RNG theo thread của M0) | pass, 100 user |

### Rò rỉ đã phát hiện và xử lý

**Trùng tên sách trong catalogue Books.** Gold item không thể là neighbor của chính user đó — graph chỉ dựng từ `train_data`, còn gold là test item. Nhưng catalogue có **nhiều `item_id` cho cùng một quyển sách** (khác edition/format). Nếu user có bản sao kia trong lịch sử thì tên sách đáp án bị in trong neighbor table.

| Kênh rò | Số user |
|---|---|
| Qua neighbor table | 23 |
| Qua `M_u` | 0 |
| **Tổng** | **23 / 2350 (0.98%)** |

Ví dụ: `Mockingjay: The Hunger Games`, `Stranger in a Strange Land`, `Lies My Teacher Told Me`.
Đã loại khỏi cả 3 split (`src/rl/leakage.py`); danh sách đầy đủ ở `data/rl/stager_books_dropped_users.json`. Vì thế split thực tế là 1185/149/993 thay vì 1200/150/1000 — không bù thêm user vì phải trả thêm tiền warmup cho ~1%.

### Hai confound đã chặn trước khi train

1. **Ngân sách neighbor của packer.** `SnippetPacker.pack()` trừ 300 token cho khối candidate. Bỏ candidate mà không bù thì state RL được 1000 token neighbor còn baseline prompted chỉ 700 → chênh lệch kết quả sẽ lẫn với chênh lệch context. Đã ghim `CANDIDATE_BLOCK_RESERVE = 300` trong `src/rl/env.py`, có unit test.
2. **Negative của warmup trùng negative của eval.** `sample_candidates` dùng cùng RNG stream cho cả hai nên rút đúng 9 distractor giống nhau, mà Stage-R lúc warmup *có* nhìn khối candidate → distractor của bài thi đã góp phần nặn ra `M_u`. Đã tách bằng salt; sửa miễn phí vì candidate eval sinh offline.

### Ghi chú về cấu trúc pipeline (ảnh hưởng thiết kế reward ở M2)

- `LLMRulePruner.prune()` **bỏ qua tham số `candidates`** → `N'_k(u)` vốn đã candidate-blind và chỉ phụ thuộc graph đóng băng. Cache nó không làm đổi hành vi pipeline.
- `SnippetPacker.build_neighbor_snippet()` dựng bảng neighbor từ **metadata tĩnh của item**, không phải từ `M_v` đang tiến hoá. Nghĩa là input duy nhất phụ thuộc memory của Stage-R là `M_u` của chính user đó. Memory của neighbor chỉ đi vào pipeline qua `item_mems` của Stage-ReRank — nên snapshot vẫn phải giữ item memory cho reward ranker.
- State **không** chứa instruction InstructRec: instruction diễn giải lại quyển sách đáp án, và pipeline gốc cũng chỉ đưa nó cho Stage-ReRank. Nó nằm ở trường `instruction` riêng trong jsonl, dành cho frozen ranker.

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
