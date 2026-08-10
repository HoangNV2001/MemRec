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

### Phần A — CPU + API (đã xong, 0 GPU-hour)

Reference gpt-4o-mini đã cache: `data/rl/m2_val_reference_books.json` (0.4 MB, 149 val user, 5.7 phút, ~$0.5).
Sinh bằng `bash scripts/rl/02_validate_reward.sh reference`. Đây là nửa đắt tiền của Validation A/B; §11.6 yêu cầu có sẵn **trước** khi thuê H100, để lúc lên GPU chỉ còn chấm lại đúng các cặp đã cache bằng ranker 1.5B.

**Năm arm, chấm bằng chính `LLMReranker` của repo (tức `LLM_Rec` thật), NDCG@5, n=149:**

| Arm | `M_collab` | NDCG@5 | H@1 |
|---|---|---:|---:|
| `sample1` | gpt-4o-mini, temp 1.0 | **0.7204** | 0.5906 |
| `sample2` | gpt-4o-mini, temp 1.0 (mẫu 2) | 0.7155 | 0.5638 |
| `shuffled` | `M_collab` của user khác | 0.6090 | 0.4295 |
| `lorem` | lorem ipsum | 0.6079 | 0.4430 |
| `empty` | không có memory (= `r_null`) | 0.6092 | 0.4362 |

**Paired delta so với arm `empty`** (cùng user, nên nhiễu nhỏ hơn nhiều):

| Arm | Δ NDCG@5 | 95% CI | tốt hơn / bằng / tệ hơn |
|---|---:|---|---|
| `sample1` (memory thật) | **+0.1112** | [+0.0640, +0.1585] | 45 / 91 / 13 |
| `shuffled` | −0.0002 | [−0.0374, +0.0371] | 21 / 104 / 24 |
| `lorem` | −0.0013 | [−0.0253, +0.0228] | 12 / 123 / 14 |

**Đọc kết quả này:**
1. **Collaborative memory có tác dụng thật** trên `LLM_Rec` thật: +0.111 NDCG@5, khoảng tin cậy không chứa 0. Reward có tín hiệu để tối ưu — đây là điều kiện cần của cả đồ án và giờ đã có bằng chứng trước khi tiêu GPU-hour nào.
2. **Memory sai bị bỏ qua chứ không gây nhiễu.** `shuffled` và `lorem` đều ≈ `empty` (CI chứa 0). gpt-4o-mini đơn giản là phớt lờ memory không liên quan. Đây là tính chất tốt, nhưng nó **bác bỏ thứ tự mà DoD giả định** — xem mục hiệu chỉnh dưới.
3. **Instruction KHÔNG làm phẳng tín hiệu memory.** Toàn bộ số trên đo *khi đã có* instruction trong prompt (đúng như pipeline gốc). Memory vẫn thêm +0.111 → trả lời câu hỏi mở của §5.1: giữ `include_instruction=True` là an toàn và trung thành với `LLM_Rec`. Vẫn sẽ đo cả hai chế độ ở Phần B.

### ⚠️ Phát hiện chặn đường: reward theo rank bị trùng giá trị quá nhiều

Hai `M_collab` lấy mẫu độc lập cho **cùng một user** cho **cùng vị trí gold** ở **111/149 user (74%)**.

| Dạng reward | Tỉ lệ trùng giữa 2 mẫu | Số giá trị phân biệt |
|---|---:|---:|
| **NDCG@5** (mặc định của §5.1) | **80.5%** | 6 |
| NDCG@10 | 74.5% | 11 |
| MRR = 1/(rank+1) | 74.5% | 11 |

74% là **trần cứng** cho mọi reward chỉ phụ thuộc rank: khi hai memory đặt gold vào cùng vị trí thì mọi hàm của rank đều bằng nhau.

Hệ quả nếu để nguyên: trong một group GRPO (cùng prompt, G=8 rollout), reward trùng nhau ⇒ `std(r) = 0` ⇒ advantage = 0 ⇒ **không có gradient**. Đây đúng là chế độ hỏng §9.2, và dynamic sampling §6.4 sẽ lọc vượt xa ngưỡng báo động 60% của kill criteria §M4. Nói cách khác: chạy GRPO với reward hiện tại nhiều khả năng cho đường reward phẳng, và ta sẽ mất vài phiên H100 để phát hiện điều mà phép đo $0.5 này đã nói trước.

**Cách xử lý đã implement:** thêm số hạng liên tục `soft_weight * p_gold`, với `p_gold` là xác suất softmax mà ranker đặt lên candidate gold. Liên tục nên gần như không bao giờ trùng, trong khi NDCG vẫn là số hạng chi phối và metric báo cáo vẫn nguyên nghĩa. **Mặc định `soft_weight = 0.0`, tức đúng công thức §5** — bật hay không sẽ do Phần B quyết định bằng tỉ lệ trùng đo trên ranker 1.5B thật.

Lập luận này không mới: §5.1 đã loại Hit@1 để chọn NDCG@5 **vì đúng lý do đó** ("Hit@1 nhị phân → rất nhiều group có std(r)=0"). Số liệu trên chỉ cho thấy NDCG@5 vẫn chưa đủ mịn.

### Hiệu chỉnh tiêu chí Validation B

DoD §7 M2 viết `r(thật) > r(user khác) > r(lorem) ≈ r(rỗng)`. Bất đẳng thức **ở giữa không đúng trên chính `LLM_Rec` thật** (bảng trên: 0.6090 vs 0.6079, cả hai CI đều chứa 0 so với `empty`). Yêu cầu proxy tái hiện `shuffled > lorem` là đòi proxy phải **dễ bị đánh lừa hơn** mô hình mà nó thay thế.

Tiêu chí đã đổi thành, kèm biên an toàn:

```
r(thật) ≥ max(r(user khác), r(lorem), r(rỗng)) + 0.02
```

Việc ba arm hỏng túm tụm lại với nhau vẫn được báo cáo (là tính chất tốt), nhưng không dùng để gate.

### Phần A — trạng thái DoD

| Hạng mục | Trạng thái |
|---|---|
| `metrics.py` + unit test tính tay | ✅ khớp `src/train/metrics.py` |
| `grounding.py` (source_ids + cosine, encoder tiêm được) | ✅ |
| `composite.py` khớp chữ ký reward của TRL | ✅ |
| `ranker.py` có stub mode | ✅ |
| `tests/rl/test_reward_logic.py` pass trên CPU với stub | ✅ **140 test pass**, 26s, không API/GPU |
| Harness validation chạy end-to-end trên CPU | ✅ 745 cặp, throughput 24 516 reward/s (stub) |

> Chạy `02_validate_reward.sh stub` cho ρ = 0.058 và Validation B FAIL. **Đó là đúng** — stub chấm bằng hash, không có ngữ nghĩa. Chỉ Validation C và "harness chạy được" là có nghĩa ở chế độ stub; A/B do run `hf` trên GPU quyết định. Script in cảnh báo này ra màn hình.

### Phần B — GPU (chạy 2026-08-07, NVIDIA L4 24 GB, ~1.5 GPU-hour)

Không gọi API: chấm lại đúng 745 cặp `(user, arm)` đã cache ở Phần A.

**Kết luận ngắn: ranker `Qwen2.5-1.5B-Instruct` của kế hoạch KHÔNG dùng được. Đã đổi sang `Qwen2.5-3B-Instruct` (đúng phương án dự phòng §M2). Với 3B + `soft_weight = 0.3`, Validation A và B đạt — nhưng đạt sát nút. Validation C chưa đo được trên L4.**

#### Ranker 1.5B — hỏng, không phải sát nút

| | 1.5B (instr. on) | 1.5B (instr. off) | ngưỡng |
|---|---:|---:|---|
| Spearman ρ | **0.3071** | 0.1411 | ≥ 0.6 |
| NDCG@5 `sample1` | 0.4091 | 0.3310 | |
| NDCG@5 `lorem` | **0.4174** | 0.3144 | phải < `sample1` |
| NDCG@5 `empty` | 0.4031 | 0.3111 | |
| Tỉ lệ trùng NDCG@5 | 94.6% | 90.6% | |

`lorem` **thắng** memory thật. Khoảng cách `sample1 − empty` = +0.006, trong khi trên `LLM_Rec` thật là +0.111 — nhỏ hơn ~18 lần, tức 1.5B gần như không đọc memory. Random NDCG@5 với 10 candidate ≈ 0.295, nên 0.41 chỉ hơn ngẫu nhiên một chút.

`include_instruction=True` tốt hơn hẳn (0.307 vs 0.141) → **giữ `True`**, đúng như Phần A dự đoán. Câu hỏi bỏ ngỏ ở §5.1 đã đóng.

#### Ngưỡng ρ ≥ 0.6 là hợp lý, không phải do trần tie

Đo tự-tương quan của chính gpt-4o-mini (`sample1` vs `sample2`, 149 user): **ρ = 0.899**. Trần cao, nên 0.6 không phải yêu cầu bất khả thi — 1.5B fail thật.

#### Ranker 3B (đã chọn) — số chính thức, fp32, batch 32, instruction on

> **Bảng dưới là số CUỐI CÙNG, đo trên 5 mẫu `M_collab`/user (8 arm, 1192 cặp).** Bản đo đầu chỉ có 2 mẫu/user và cho Validation A "đạt sát nút 0.6051" — con số đó **đã bị bác bỏ**, xem mục dưới.

| Validation | Số đo | Ngưỡng | Kết quả |
|---|---:|---|---|
| **A** ρ, reward = NDCG@5 (§5 nguyên bản) | 0.5573 | ≥ 0.6 | ✗ **FAIL** |
| **A** ρ, mọi `w > 0` của `soft_weight` | ≤ 0.5833 | ≥ 0.6 | ✗ **FAIL** |
| **B** `sample1` ≥ max(arm hỏng) + 0.02 | margin +0.0007 | ≥ 0 | ✓ *(cực sát)* |
| **B** trung bình 5 arm thật vs arm hỏng | margin +0.0120 | ≥ 0 | ✓ |
| **C** throughput | 1.38 reward/s @ batch 16 | ≥ 20 @ batch 64 | ⏸ chưa đo được trên L4 |

NDCG@5 theo arm (3B/fp32): `sample1` 0.5526 · `sample2` 0.5616 · `sample3` 0.5623 · `sample4` 0.5622 · `sample5` 0.5809 · `shuffled` 0.4594 · `lorem` 0.5320 · `empty` 0.5123.

**Validation A FAIL ở mọi cấu hình.** Con số 0.6051 báo cáo lúc đầu chỉ đạt được khi (a) bật `soft_weight` và (b) chỉ có 2 arm mẫu thật trong 5 arm. Thêm 3 arm mẫu thật → ρ tụt còn 0.5573–0.5833. ρ gộp nhạy với tỉ lệ arm thật/arm hỏng, và bản 5 mẫu là bản đáng tin hơn.

#### `soft_weight` — đã BẬT rồi lại TẮT. Kết luận cuối: **0.0**

Quy tắc §M2 ("trùng > 50% thì bật, khởi điểm 0.3") kích hoạt vì ranker 3B trùng 70.8%. Em đã bật. **Rồi đo lại đàng hoàng với 5 mẫu/user và phải tắt đi** — quy tắc đó nhìn sai số.

Phân rã trên 296 cặp trong-user mà gpt-4o-mini phân biệt được:

| Ai quyết định | Số cặp | Đồng ý | 95% CI | |
|---|---:|---:|---|---|
| NDCG@5 tự quyết được | 99 | **60.6%** | [50.8, 69.7] | trên ngẫu nhiên |
| NDCG@5 hoà, `p_gold` phá hoà | 197 | **40.1%** | [33.5, 47.1] | **DƯỚI ngẫu nhiên** |
| Gộp lại | 296 | 47.0% | [41.3, 52.6] | = ngẫu nhiên |

`p_gold` được giao đúng những cặp mà NDCG@5 không phán được, và trên đúng những cặp đó nó **phản tín hiệu** — cả khoảng tin cậy nằm dưới 50%. Nó không cứu được group suy biến, nó **đổ nhiễu ngược hướng** vào đó, và kéo tín hiệu 60.6% (yếu nhưng thật) xuống mức ngẫu nhiên.

Hiệu ứng **không tinh chỉnh được bằng `w`**: mọi `w > 0` đều phá sạch 197 cặp hoà, nên kết quả y hệt nhau từ `w = 0.005` tới `w = 0.3`. Đã quét và xác nhận.

> **Hệ quả: bài toán trùng reward (70.8%) vẫn CHƯA có lời giải.** Đây là blocker thật của M4, không phải chi tiết kỹ thuật. Đừng bật lại `soft_weight` để "chữa" nó.

#### ⚠️ Phát hiện quan trọng nhất: reward phân biệt thô được, tinh thì gần như không

Đây là số mà một group GRPO thực sự nhìn thấy, và nó **không** nằm trong DoD gốc. ρ gộp bị chi phối bởi khác biệt *giữa các user* ("user này dễ với mọi model") — thứ GRPO không bao giờ thấy, vì mọi rollout trong một group thuộc **cùng một** user và chỉ khác nhau ở `M_collab`.

Bản đo đầu chỉ có **29 cặp**, CI ~[19%, 59%] — không kết luận được gì. Đã sinh thêm 3 mẫu/user (`src/rl/extend_val_reference.py`, ~$0.33, 3.5 phút) → mỗi user tối đa C(5,2) = 10 cặp.

**Nửa API — tín hiệu tinh CÓ tồn tại, thưa nhưng mạnh:**

| | |
|---|---:|
| Tổng số cặp trong-user | 1490 |
| gpt-4o-mini chấm hoà | 1194 (80.1%) |
| **Cặp phân biệt được** | **296** |
| Biên độ TB của cặp phân biệt được | **0.3413** |
| Hiệu ứng thô (real − empty) để đối chiếu | +0.1112 |
| User có bất kỳ spread nào | 54/149 (36.2%), swing TB 0.3949 |

→ Tín hiệu tinh **tồn tại và lớn** — gấp ~3 lần hiệu ứng thô — chỉ là **tập trung ở ~36% user**. Đây đúng là thứ curriculum §6.4 sinh ra để chọn. **Kết cục xấu nhất ("không có gì để học") đã bị loại trừ.**

**Nửa GPU — proxy chỉ bám được một phần, và chỉ ở nhánh NDCG:**

| Ai quyết định | Số cặp | Đồng ý với gpt-4o-mini | 95% CI |
|---|---:|---:|---|
| NDCG@5 tự quyết được | 99 | **60.6%** | [50.8, 69.7] — trên ngẫu nhiên |
| NDCG@5 hoà → `p_gold` phá hoà | 197 | **40.1%** | [33.5, 47.1] — **dưới ngẫu nhiên** |
| **Reward tổng khi bật `soft_weight`** | **296** | **47.0%** | [41.3, 52.6] — **ngẫu nhiên** |

Hiệu chuẩn: stub ranker (chấm bằng hash, vô nghĩa) cho 50.2% [43.6, 56.9] — metric đúng. Đối chiếu thô: `sample1` vs `empty` đồng ý 72.4%.

**Đọc thẳng:** reward chỉ có tín hiệu trong-user ở **1/3 số cặp** mà NDCG@5 phán được, và ngay cả ở đó cũng chỉ 60.6% với cận dưới CI sát 50%. Hai phần ba còn lại hoà, và cách phá hoà duy nhất đã thử làm mọi thứ tệ hơn.

#### Reward không tất định trong bf16 → chuyển sang fp32

§5.1 chọn thiết kế một-forward-pass **vì nó tất định** ("GRPO không có critic nên nhiễu reward đi thẳng vào variance của advantage"). Trong bf16 lời hứa đó **sai**: padding đổi thứ tự cộng dồn số thực, nên cùng một `(prompt, completion)` cho điểm khác nhau tuỳ rollout nào tình cờ nằm chung batch.

Đo trên 48 val user, batch 24 vs batch 1: **bf16 lệch 2/48 user, fp32 lệch 0/48**. Nguyên nhân: model cực kỳ nhọn — ~99% xác suất dồn vào một chữ cái (đo được: letter mass 0.9998) — nên chênh lệch logit *dưới* hạng 1 rất nhỏ, và NDCG@5 đọc đúng cái đuôi đó.

→ Mặc định `FrozenRanker.dtype = "float32"`. Giá phải trả: VRAM gấp đôi và throughput bằng ~½ bf16.

#### Prompt ranker đã đúng — đã loại trừ, không cần sửa

Nghi ngờ hợp lý là vị trí đọc logit (ngay đầu lượt assistant) khiến model muốn viết "Based"/"The" thay vì một chữ cái, làm softmax hạn chế đọc phần đuôi phân phối. **Không phải:** letter mass trung bình = **0.9998**. Thêm prefix "Answer: " vào lượt assistant còn **phá** nó (mass → 0.0000). Giữ nguyên.

#### Backfill `r_null` / `baseline_h1` / `baseline_p_gold`

`src/rl/backfill_baselines.py`, một job batch, ranker 3B fp32, instruction on.

| split | n | `r_null` (NDCG@5, không memory) | `baseline_h1` | `baseline_p_gold` trong dải [0.2, 0.8] |
|---|---:|---:|---:|---:|
| train | 1185 | 0.5068 | 0.3409 | 141 (11.9%) |
| val | 149 | 0.5123 | 0.3557 | 20 (13.4%) |
| test | 993 | 0.5267 | 0.3545 | 111 (11.2%) |

`r_null` của val = 0.5123 **trùng khít** arm `empty` của Validation B — hai đường code độc lập cho cùng một số, đây là cách bug dưới đây bị bắt.

> **Dải curriculum §6.4 chỉ giữ ~12% user.** Với `baseline_p_gold`, dải `[0.2, 0.8]` giữ 141/1185 user train. Không phải lỗi — ranker rất nhọn nên `p_gold` phân cực — nhưng M4 cần biết trước: 1185 → ~141 prompt là ít hơn nhiều so với giả định của §6.4. Nới dải hoặc bỏ curriculum là quyết định của M4, không phải của M2.

Ghi kèm `data/rl/baselines_provenance.json` (model + dtype + số bản ghi + thời điểm cho từng split) vì ba trường này là thuộc tính của **một** ranker cụ thể, mà nhìn vào jsonl thì không phân biệt được số của ranker nào.

**`baseline_h1` nhị phân không dùng được cho curriculum §6.4.** Với ranker đóng băng tất định, Hit@1 của một user chỉ có thể là 0.0 hoặc 1.0, nên dải `[0.2, 0.8]` của `filter_by_difficulty` sẽ khớp **0 user** và làm rỗng tập train — lỗi chỉ lộ ra sau khi đã trả tiền thuê máy. Đã thêm trường thứ ba `baseline_p_gold` (xác suất softmax ranker đặt lên gold, liên tục trong [0,1]) diễn đạt đúng ý định của §6.4. Cả ba trường đều được ghi; chọn trường nào lái curriculum vẫn là việc của `filter_by_difficulty` ở M4.

> **Hai bug đã bắt được trong lúc backfill (đáng ghi lại — cả hai đều im lặng):**
>
> 1. **Sai model.** Bản đầu của `backfill_baselines.py` tự giữ một default `--ranker_model` = 1.5B, nên lần chạy đầu ghi đè cả 3 file jsonl bằng số của **ranker chưa validate**. Lộ ra vì `r_null` (0.4109) không khớp arm `empty` của Validation (0.5123) — hai đường code lẽ ra phải cho cùng một số. Đã bỏ hẳn default trùng lặp (giờ thừa kế từ `FrozenRanker`) và in cấu hình ranker mỗi lần chạy.
> 2. **OOM giữa chừng.** Lần chạy lại (batch 32) ghi xong `train` + `val` rồi **OOM ở `test`** — L4 24 GB không đủ cho 3B fp32 ở batch 32 với prompt dài nhất. Kết quả: 2 split mới, 1 split còn số cũ của 1.5B, và **nhìn vào dữ liệu thì không thấy được**. Đã chạy lại `test` riêng ở batch 8.
>
> Cả hai lần đều là "số trông hợp lệ nhưng sai nguồn". Vì vậy thêm `data/rl/baselines_provenance.json`. Sổ hash trong `verify_transfer.py` cũng đã cập nhật (nội dung jsonl đổi thật, số dòng không đổi).

#### Trạng thái DoD Phần B

| Hạng mục | Trạng thái |
|---|---|
| Bật chế độ thật của `ranker.py` | ✅ (đổi 1.5B → 3B, fp32) |
| **Validation A — ρ ≥ 0.6** | ❌ **FAIL — 0.5573** (§5 nguyên bản); ≤ 0.5833 với mọi `soft_weight` |
| Validation B — `r(thật) ≥ max(arm hỏng) + 0.02` | ✅ margin +0.0007 (`sample1`) / +0.0120 (TB 5 arm thật) — cực sát |
| Validation C — ≥ 20 reward/s @ batch 64 | ⏸ **chưa đo được**: 3B fp32 không vừa batch 64 trên L4 24 GB. Đạt 1.38/s @ batch 16 |
| Đo tỉ lệ trùng → quyết định `soft_weight` | ✅ 70.8% → bật 0.3 → **đo lại → TẮT về 0.0** |
| Chạy `--no_instruction` đối chứng | ✅ kém hơn hẳn → giữ `include_instruction=True` |
| Backfill 3 file jsonl | ✅ 1185 / 149 / 993 |
| **[thêm] Within-user agreement** | ✅ đã đo — và đây là lý do A fail có ý nghĩa |

### ❌ Verdict: M2 KHÔNG ĐẠT. Không được bắt đầu M4.

§M2 viết rõ: *"Không được đi tiếp M4 với reward chưa validate — 400 step trên reward sai là mất cả phiên thuê máy và cả tuần."* Điều kiện đó đang không thoả:

1. **Validation A fail** ở mọi cấu hình (tốt nhất 0.5833 < 0.6).
2. **Reward không xếp hạng được hai memory tốt** cho cùng một user: 60.6% trên 1/3 số cặp, ngẫu nhiên trên phần còn lại.
3. **Bài toán trùng reward 70.8% chưa có lời giải.** `soft_weight` — phương án duy nhất đã thử — làm tệ hơn. Với `soft_weight = 0`, ~2/3 group GRPO sẽ có `std(r) = 0` → không gradient (§9.2), và dynamic sampling §6.4 sẽ lọc vượt xa ngưỡng báo động 60% của kill criteria M4.

**Điều đã cứu được:** toàn bộ kết luận này mua bằng **~$0.35 API + ~1.5 GPU-hour trên L4 rẻ**, thay vì phát hiện đường reward phẳng sau vài phiên H100. Đây đúng là việc M2 sinh ra để làm.

**Điều KHÔNG kết luận:** rằng ý tưởng đồ án sai. Tín hiệu tinh có thật và lớn (0.34 NDCG@5 trên 296 cặp, gấp 3 lần hiệu ứng thô) — chỉ là **proxy 3B hiện tại không đọc được nó**. Đây là vấn đề của proxy, không phải của bài toán.

### Pointwise scoring (§M2 phương án 2) — đã thử, KHÔNG cứu được

Chấm từng candidate một câu hỏi Yes/No độc lập, xếp hạng theo **hiệu logit** `Yes−No`. 1192 cặp, 3B fp32, L4, ~73 phút.

> **Bug đã bắt trước khi nó làm hỏng số:** bản đầu xếp hạng theo `softmax(P("Yes"))`. Model trả lời "No" cho gần như mọi candidate → `P(Yes) ≈ 1e-30` → **underflow về đúng 0.0** trong float32, làm 8/10 candidate sập về một giá trị và **tái tạo lại chính bài toán trùng mà pointwise sinh ra để diệt**. Hiệu logit là biến đổi đơn điệu tương đương nhưng ổn định số học; sau khi đổi, cả 10 candidate đều phân biệt.

| | listwise | pointwise | gpt-4o-mini |
|---|---:|---:|---:|
| **Validation A** — ρ gộp | **0.5573** | **0.4010** | — |
| **Validation B** — margin | +0.0120 | **+0.0661** | — |
| Độ nhạy thô `real − empty` | +0.0516 | **+0.0861** | +0.0994 |
| Trong-user, cặp NDCG phán được | 60.6% [50.8, 69.7] | 56.3% [46.7, 65.5] | — |
| Trong-user, cặp NDCG hoà (tiebreaker) | 40.1% [33.5, 47.1] | 44.6% [37.7, 51.6] | — |
| **Trong-user, gộp** | **47.0%** | **48.6%** | — |
| Tỉ lệ trùng NDCG@5 | 70.8% | 66.0% | 80.5% |
| Throughput (L4, fp32) | 1.38/s | **0.30/s** | — |

**Đọc kỹ, có hai chiều ngược nhau:**

- **Pointwise TỐT HƠN ở vùng thô.** Độ nhạy `real − empty` là +0.0861, sát `LLM_Rec` thật (+0.0994) hơn hẳn listwise (+0.0516); margin Validation B gấp 5.5 lần. Nó phân biệt "memory thật vs memory rác" tốt hơn thật sự.
- **Pointwise TỆ HƠN ở ρ gộp** (0.40 vs 0.56) — nó chấm "user có thích item này không" theo giá trị tuyệt đối, một bài toán khác với xếp hạng, nên đồng thuận với gpt-4o-mini về *user nào dễ* kém đi.
- **Trong-user thì cả hai đều là ngẫu nhiên** (47.0% và 48.6%, CI của cả hai đều chứa 50%).

#### ⛔ Kết luận: nút thắt KHÔNG nằm ở scoring mode

Hai thiết kế scorer khác nhau về bản chất — một softmax 10 chiều có ràng buộc tổng bằng 1, và mười điểm Yes/No hoàn toàn độc lập — cho **cùng một kết quả trong-user: ngẫu nhiên**. Đổi cách hỏi không giải quyết được. Nút thắt nằm ở tầng trên:

1. **Model 3B đơn giản là yếu hơn hẳn ở chính bài toán này.** NDCG@5 tuyệt đối: gpt-4o-mini 0.709 vs 3B 0.564–0.577 (ngẫu nhiên = 0.295). Một model kém hơn nhiều ở việc xếp hạng thì không thể tái hiện được phán đoán tinh của model giỏi hơn. Đây là giả thuyết mạnh nhất còn lại.
2. **Hoặc trần nằm ở *dạng* reward** — `f(thứ hạng gold)` chỉ có 6 giá trị. Pilot đã xác nhận: pointwise vẫn trùng 66%, vì hai memory đặt gold vào cùng vị trí thì NDCG y hệt nhau bất kể scorer liên tục đến đâu. **Em đã sai khi nói pointwise "thoát trần 6 giá trị"** — trần đó nội tại trong dạng reward, không phải ở scorer.

#### 🔧 Căng thẳng kiến trúc mới lộ ra (quan trọng cho M4)

§4.3 muốn ranker **colocate** cùng policy 4B + vLLM trên một H100 → trần ranker ~3B. Nhưng đo được rằng **3B quá yếu để làm proxy trung thực**. Hai ràng buộc này mâu thuẫn nhau. Ba cách thoát:

| Cách | Đánh đổi |
|---|---|
| Ranker chạy GPU riêng (2 GPU) | $/h cao hơn nhưng có thể ít giờ hơn; cho phép 7B+ |
| Thu nhỏ policy (1.5B thay vì 4B) để nhường VRAM cho ranker 7B | Policy nhỏ hơn → đóng góp "LM_Mem nhỏ" vẫn giữ được, thậm chí mạnh hơn |
| Reward gọi API gpt-4o-mini | 0 VRAM, tương quan hoàn hảo theo định nghĩa; ~$17–30 + latency |

### Đổi `LLM_Rec` sang model mạnh hơn (gpt-5.6-luna) — đã đo, và nó ĐÓNG một họ giải pháp

Chấm lại **đúng 1192 cặp `(user, arm)` đã cache** bằng `gpt-5.6-luna` thay cho gpt-4o-mini. Không sinh memory mới, không GPU. Kèm 298 lời gọi đối chứng nhiễu. **$1.40, 5.5 phút.**

> **Ràng buộc kỹ thuật phải biết trước:** họ gpt-5 **từ chối `temperature=0`** (kiểm chứng trực tiếp với API: `400 Only the default (1) value is supported`), và từ chối `max_tokens` (phải dùng `max_completion_tokens`). Reranker của repo chấm ở `temperature=0.0` để tất định — nên **judge thuộc họ gpt-5 là ngẫu nhiên**. Đã nới điều kiện chọn tham số trong `llm_client.py` (trước đó chỉ khớp chuỗi `nano`); đường gpt-4o-mini không đổi một byte.

| arm | gpt-5.6-luna | gpt-4o-mini |
|---|---:|---:|
| sample1–5 (memory thật, TB) | **0.8214** | 0.7126 |
| shuffled | 0.6411 | 0.6090 |
| lorem | 0.6469 | 0.6079 |
| empty | 0.6455 | 0.6092 |
| **headroom `real − empty`** | **+0.1759** | +0.1112 |
| **Validation B margin** | **+0.1545** | — |

#### ✅ Hai tin tốt

1. **Tiền đề đồ án không chỉ sống sót mà mạnh hơn.** Lo ngại "reranker mạnh hơn thì tự suy ra sở thích, memory thành thừa" đã bị **bác bỏ**: headroom tăng từ +0.1112 lên **+0.1759**. Luna cũng là recommender giỏi hơn hẳn (0.82 vs 0.71 NDCG@5 trên arm thật).
2. **ρ(Luna, gpt-4o-mini) = 0.6635 — vượt ngưỡng Validation A 0.6.** Đây là **thứ đầu tiên trong cả M2 đạt Validation A**. Và nó là **cận dưới**, vì Luna tự nhiễu còn reference thì tất định. Tức là nếu giữ `LLM_Rec` = gpt-4o-mini thì Luna *về mặt tương quan gộp* đủ tư cách làm reward.

#### ⛔ Phép đối chứng nhiễu, và vì sao nó đóng cả một họ giải pháp

Chấm **cùng một memory** (`sample1`) 3 lần cho mỗi user, rồi so với việc chấm **hai memory khác nhau**:

| So cái gì | Cặp | Phân biệt được | Biên độ TB |
|---|---:|---:|---:|
| Hai memory **khác nhau** (Luna) | 1490 | **20.9%** | 0.3312 |
| **Cùng một memory**, chấm 2 lần (Luna) | 447 | **18.3%** | 0.3096 |
| | | **chênh 2.6 điểm** | |

**Gần như toàn bộ khả năng "phân biệt hai memory" của Luna là nhiễu lấy mẫu của chính nó.** Nó phân biệt một memory với *chính nó* gần bằng phân biệt nó với một memory khác.

Đây cũng là một cảnh báo phương pháp luận: hình dạng thống kê "~20% cặp tách được, biên độ ~0.33" **là thứ nhiễu thuần tuý cũng tạo ra**, vì NDCG@5 rất thô — mọi xáo trộn đẩy gold đi một bậc đều tạo bước nhảy ~0.3. Con số 296 cặp / 0.3413 của gpt-4o-mini vẫn đứng vững (nó chấm ở temperature 0, tất định, nên khác biệt bắt buộc đến từ memory), nhưng **biên độ giống nhau không hàm ý nguyên nhân giống nhau**.

#### 🔒 Kết luận cứng: trần 80% là của BÀI TOÁN, không phải của người chấm

| Người chấm | Tỉ lệ hoà trong-user |
|---|---:|
| gpt-4o-mini (temp 0) | **80.1%** |
| gpt-5.6-luna (temp 1) | **79.1%** |

Một model mạnh hơn hẳn cho **cùng một tỉ lệ hoà**. Nguyên nhân là cấu trúc: 10 candidate + NDCG@5 chỉ có 6 giá trị + hai memory tốt thường đặt gold vào cùng một vị trí.

→ **Toàn bộ họ giải pháp "nâng cấp người chấm" đã đóng.** 1.5B → 3B → pointwise → gpt-5.6-luna: không nhánh nào vượt được trần này, vì trần không nằm ở người chấm. Mua kết luận này bằng **$1.40 và 5.5 phút**.

#### Hệ quả cho từng phương án

| Phương án | Trạng thái sau phép đo |
|---|---|
| Luna làm **reward trong vòng lặp M4** | ❌ **Loại.** Không phải vì tiền ($12.1/run, chấp nhận được) mà vì ~88% khả năng phân biệt của nó là nhiễu, và nhiễu reward đi thẳng vào advantage của GRPO (§5.1 chọn one-forward-pass *vì* tính tất định). Muốn khử phải chấm lặp k lần → chi phí và độ trễ ×k, để đổi lấy 2.6 điểm tín hiệu thật |
| Luna làm **`LLM_Rec` cho bảng kết quả** | ⭕ **Hấp dẫn ở trục thô.** Validation B margin +0.1545 (so với +0.0120 của proxy 3B), headroom +0.1759. Nhiễu triệt tiêu khi lấy TB trên 993 test user. Giá chạy lại M0 chỉ ~$2.45. Nhưng **không** giải quyết được trần hoà |
| Nâng cấp người chấm để cứu tín hiệu tinh | ❌ **Đóng vĩnh viễn** — xem bảng tỉ lệ hoà |

### Hướng đi tiếp — sau khi đã loại pointwise

Đã thử và loại: **ranker 3B** (ρ 0.5573), **`soft_weight`** (làm tệ hơn), **pointwise** (ρ 0.4010, trong-user vẫn ngẫu nhiên). Phương án dự phòng §M2 ghi sẵn đã dùng hết.

| Hướng | Chi phí | Lý lẽ |
|---|---|---|
| **Ranker 7B/8B, chẩn đoán** ⭐ | ~1–2 GPU-h trên L40 48 GB | Kiểm tra trực tiếp giả thuyết mạnh nhất còn lại: "3B quá yếu". Nếu 7B đưa trong-user lên rõ trên 50% → biết chắc là vấn đề dung lượng, và bài toán chuyển thành bố trí VRAM (3 cách ở trên). Nếu 7B **cũng** ngẫu nhiên → proxy nội bộ là ngõ cụt, phải đổi sang reward API hoặc thu hẹp phát biểu. **Rẻ và loại trừ được nhiều nhất** |
| Reward = gpt-4o-mini trực tiếp | ~$17–30 + latency mỗi step | Bỏ hẳn proxy. Tương quan hoàn hảo theo định nghĩa. Làm M4 chậm đi đáng kể nhưng không tốn VRAM |
| Đổi *dạng* reward | ~1 GPU-h | Bỏ NDCG@5 (6 giá trị) sang đại lượng liên tục theo thứ hạng — ví dụ log-rank, hoặc điểm gold chuẩn hoá theo phân phối trong chính user. Tấn công trần trùng ở đúng chỗ nó thật sự nằm |
| Thu hẹp phát biểu đồ án | 0 | Nhắm trục **cost** của Figure 4 ("ngang accuracy, memory ngắn/rẻ hơn"). Reward hiện tại **đủ dùng cho việc này** — Validation B đạt, và pointwise đạt tốt (margin +0.0661). Chỉ bỏ tham vọng vượt accuracy |

> Nếu chọn hướng "thu hẹp phát biểu": **dùng pointwise, không dùng listwise.** Nó nhạy với chất lượng memory gần bằng `LLM_Rec` thật (+0.0861 vs +0.0994) — tức là dạy được đúng bài học thô mà phát biểu đó cần. Đổi lại phải chịu 0.3 reward/s, nên phải giải bài throughput trước.

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
