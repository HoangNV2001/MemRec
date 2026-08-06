# RL_LM_REC_EXTENSION.md — Extension huấn luyện `LM_Rec` sau `LM_Mem`

> **Đặt file này tại:** `docs/RL_LM_REC_EXTENSION.md` trong repo `rutgerswiselab/MemRec`.
>
> **Loại kế hoạch:** `EXTENSION` — chỉ thực hiện sau khi hoàn tất Plan gốc `docs/RL_PLAN.md`.
>
> **Điều kiện bắt buộc để bắt đầu:** Plan gốc đã đạt M7a/MVR, đã chốt checkpoint `LM_Mem` tốt nhất, bảng kết quả chính, candidate set cố định và ranker-swap.
>
> **Nguyên tắc bất di bất dịch:** trong extension này, `LM_Mem` luôn **đóng băng**. Không joint-update `LM_Mem` và `LM_Rec`.
>
> **Trạng thái:** `REVIEWED 2026-08-06` — đã đối chiếu với hiện trạng repo sau M0/M1 và hiệu chỉnh. Xem §0.1.

---

## 0.1 Nhật ký hiệu chỉnh — 2026-08-06 (sau khi M1 hoàn tất)

Bản đầu được viết trước khi M1 chạy xong, nên có một số chỗ lệch so với code và dữ liệu thực tế. Các sửa đổi dưới đây **không** đổi mục tiêu hay phạm vi của extension, chỉ làm nó khớp thực tế và bịt hai lỗ thiết kế.

**Sai sót phải sửa (nếu không sẽ hỏng thí nghiệm):**

| # | Vấn đề | Đã sửa ở |
|---|---|---|
| 1 | Ma trận 4 dòng **bị confound**: `D > B` đổi đồng thời model ranker (gpt-4o-mini → 4B) và thêm GRPO, nên không tách được hiệu ứng nào gây ra gain | §E3 — thêm 2 dòng control A′/B′ cùng base model; phát biểu chính đổi thành `D > B′` |
| 2 | Thiếu **xáo thứ tự candidate**: set cố định + thứ tự cố định ⇒ model nhớ được vị trí gold theo từng user, triệu chứng giống hệt overfit thường | §4.2.1, §9.4bis, DoD E2 |
| 3 | Sơ đồ luồng dữ liệu **bỏ sót candidate item memories**, trong khi reranker gốc dùng chúng cho cả 10 candidate (đo được: 10/10 ở mọi record) ⇒ dòng A sẽ không còn là baseline MemRec | §4.1 |
| 4 | Yêu cầu "không có gold id/title trong prompt" **không thể đạt** với `LM_Rec` — gold là 1 trong 10 candidate | §E0, khung đính chính |
| 5 | Đề xuất **tái dùng Stage-ReRank conversations của M0** làm teacher — không dùng được: sai bộ candidate, nhiễm GT, sai format | §4.4 |

**Chỗ chưa chính xác so với thực tế (đã cập nhật số liệu):**

| # | Vấn đề | Đã sửa ở |
|---|---|---|
| 6 | Cổng yêu cầu "còn ≥20 H100-hour" nhưng §10 cần 24–36h | §3, §10 |
| 7 | Gate "candidate sampling đã deterministic" — đúng cho **đường RL** (M1), sai cho đường eval gốc (vẫn seed theo thread, cố ý) | §3 |
| 8 | Split thực tế là **1185/149/993**, không phải 1200/150/1000 | §3.1 |
| 9 | Đề xuất `fixed_candidates_books.jsonl` tạo **nguồn sự thật thứ hai** cạnh jsonl M1 | §5.2 — bỏ file, thay bằng hash |
| 10 | DoD E0 đòi re-run materialization khớp hash 100% — không đạt được vì batching của vLLM, sẽ fail vì lý do vô can | §E0 |
| 11 | Không nêu quan hệ ngân sách với Phase 2; extension và M5/M6 đầy đủ trên thực tế loại trừ nhau | §3, §10 |
| 12 | Không nói rõ phải tái dùng module M1 thay vì viết lại | §5.1 |
| 13 | `over_length`, `malformed`, `duplicate_or_missing` chưa có định nghĩa ⇒ train và eval dễ implement lệch nhau | §4.3 |

**Giữ nguyên, đánh giá là đúng:** ý tưởng tổng thể (freeze `LM_Mem` → SFT + GRPO `LM_Rec`); `NDCG@10` làm reward chính trên 10 candidate; cấm gpt-4o-mini trong vòng lặp GRPO; ma trận robustness §E4 theo nguồn memory; ba khả năng kết luận A/B/C ở §11; nguyên tắc không joint training.

---

## 0. TL;DR

Plan gốc tối ưu:

```text
RL LM_Mem → frozen LM_Rec
```

Extension này thực hiện bước tiếp theo:

```text
freeze trained LM_Mem → SFT + GRPO LM_Rec
```

Mục tiêu là kiểm tra liệu một `LM_Rec` học được có tận dụng tốt hơn collaborative memory đã được tối ưu hay không, và liệu gain của hai module có cộng dồn:

1. prompted `LM_Mem` + frozen `LM_Rec`
2. RL `LM_Mem` + frozen `LM_Rec`
3. prompted `LM_Mem` + RL `LM_Rec`
4. RL `LM_Mem` + RL `LM_Rec` theo thứ tự tuần tự

Đây là **extension**, không thay đổi main contribution của luận văn. Main contribution vẫn là RL cho `LM_Mem`; RL cho `LM_Rec` là phần kiểm tra bổ sung về modularity và sequential co-adaptation.

---

## 1. Câu hỏi nghiên cứu

### RQ-E1 — Gain có cộng dồn không?

Với `LM_Mem` đã được tối ưu và đóng băng, RL cho `LM_Rec` có cải thiện ranking thêm so với frozen `LM_Rec` không?

### RQ-E2 — Tối ưu module nào hiệu quả hơn?

Với cùng protocol và compute gần tương đương:

- RL chỉ `LM_Mem`
- RL chỉ `LM_Rec`
- RL tuần tự cả hai

phương án nào cho trade-off tốt nhất giữa accuracy, token, latency và chi phí?

### RQ-E3 — Có co-adaptation quá mức không?

`LM_Rec` train trên output của RL `LM_Mem` có còn hoạt động tốt khi nhận:

- prompted memory;
- SFT memory;
- memory rỗng;
- memory bị nhiễu nhẹ?

Nếu chỉ hoạt động tốt với đúng checkpoint `LM_Mem` đã dùng khi train, đó là dấu hiệu co-adaptation quá mức.

---

## 2. Phạm vi

### Trong phạm vi

- Freeze checkpoint `LM_Mem` tốt nhất từ M4/M7a của Plan gốc.
- Materialize `M_collab` một lần cho train/val/test; không sinh lại trong vòng lặp RL.
- SFT warm-start cho một local `LM_Rec`.
- GRPO/Rank-GRPO cho `LM_Rec` trên candidate ranking.
- So sánh ma trận 4 configuration chính.
- Robustness test khi đổi nguồn memory.
- Dùng cùng candidate set cố định và cùng test users với Plan gốc.

### Ngoài phạm vi

- Không cập nhật `LM_Mem`.
- Không joint training hai model.
- Không mở Stage-W hoặc dynamic graph.
- Không full-catalog retrieval; giữ protocol 10 candidates của Plan gốc.
- Không online A/B.
- Không train lại curation policy.
- Không thay đổi đường eval gốc của MemRec.

---

## 3. Cổng bắt đầu extension

Agent chỉ được bắt đầu E0 khi tất cả điều kiện sau đã đạt:

- [ ] M7a của `docs/RL_PLAN.md` đã hoàn tất.
- [ ] Có checkpoint `LM_Mem` tốt nhất và config đầy đủ.
- [x] ~~Candidate sampling đã được sửa để deterministic theo `user_id + global_seed`.~~ **Đã xong ở M1**, nhưng chỉ cho **đường RL**: `src/rl/splits.py:sample_candidates` seed theo `(seed, user_id, salt)`. Đường eval **gốc** (`_evaluate_single_user`) vẫn seed theo `thread.ident` — cố ý không sửa (quyết định M0, xem `docs/RESULTS.md` bug #5). Hệ quả bắt buộc: xem gạch đầu dòng kế tiếp.
- [ ] **Dòng A của ma trận §E3 phải được đo lại trên candidate set cố định của M1** — KHÔNG được bê số từ bảng "M0 Baselines" sang. Số M0 đo trên candidate sinh bằng RNG theo thread, khác hoàn toàn bộ candidate trong `data/rl/stager_books_test.jsonl`. Trộn hai nguồn là so sánh hai bài thi khác nhau.
- [x] ~~Tất cả configuration dùng cùng một candidate file đã materialize.~~ **Nguồn sự thật duy nhất là `data/rl/stager_books_{train,val,test}.jsonl` của M1** (trường `candidates`). Không tạo file candidate thứ hai — xem §5.
- [ ] Có bảng prompted / SFT / GRPO `LM_Mem` trên test.
- [ ] Ranker-swap của `LM_Mem` đã chạy.
- [ ] `LM_Mem` checkpoint, graph snapshot (`data/rl/graph_snapshot_books.json`, 17.4 MB) và 3 file jsonl đã lưu trên storage bền.
- [ ] **Ưu tiên ngân sách:** M5 Ưu tiên 1 của Plan gốc (~2h, ranker-swap + cross-domain + đọc tay + breakdown theo `|H_u|`) phải chạy **trước** extension. Nhóm đó bảo vệ trực tiếp đóng góp chính; extension là đóng góp khác.
- [ ] Còn tối thiểu **36 H100-hour** hoặc ngân sách tương đương — bằng cận trên của §10, không phải 20h. Extension gần như **gấp đôi tổng chi phí đồ án** (Phase 1 ~50h ≈ $190 → +24–36h ≈ +$75–125). Nếu chỉ còn 20h thì không đủ để chạy hết E0–E4, và một extension dở dang không dùng được vào luận văn.

Nếu bất kỳ mục nào chưa đạt: **dừng extension và quay lại Plan gốc**.

### 3.1 Số liệu thực tế M1 mà extension kế thừa

| Hạng mục | Giá trị thực tế | Ghi chú |
|---|---|---|
| Split | train **1185** / val **149** / test **993** | không phải 1200/150/1000; đã loại 23 user lộ đáp án (`src/rl/leakage.py`) |
| Candidate/user | 10, cố định, tất định | salt `EVAL_CANDIDATE_SALT` |
| Vị trí gold trong candidate list | phân bố đều trên 0–9 | nhưng **cố định theo user** — xem rủi ro §9.5 |
| `M_u` có nội dung | 1184/1185 train, 992/993 test | 3 user warmup hỏng |
| Candidate có item memory | **10/10** ở mọi record | quan trọng, xem §4.1 |

---

## 4. Thiết kế hệ thống

### 4.1 Luồng dữ liệu

```text
Frozen graph snapshot  (data/rl/graph_snapshot_books.json)
        │
        ▼
Frozen best LM_Mem
        │
        ▼
Materialized M_collab per example
        │
        ├── user instruction            (trường `instruction` của jsonl M1)
        ├── candidate list A–J          (trường `candidates`, thứ tự XÁO theo rollout — §4.2)
        ├── candidate item memories     (trường `candidate_memories` — BẮT BUỘC, xem dưới)
        └── gold item chỉ dùng trong reward
        │
        ▼
Trainable LM_Rec
        │
        ▼
Ranking permutation  ──(adapter)──►  {"scores":[{item_id, score}]}  của pipeline gốc
        │
        ▼
Direct ranking reward
```

**Bắt buộc đưa candidate item memories vào prompt.** Sơ đồ ban đầu bỏ sót mục này. `MemRecAgent.rerank()` gọi `_get_item_mems(candidates)` và `LLMReranker.build_rerank_prompt()` in ra `• Item {id} ({title}): {memory}` cho **cả 10** candidate. Đo trên `stager_books_test.jsonl`: **10/10 candidate của mọi record đều có item memory**. Nếu extension bỏ chúng đi thì dòng A của ma trận §E3 **không còn là baseline MemRec**, và toàn bộ so sánh mất ý nghĩa.

`LM_Rec` nhìn thấy candidate list vì nhiệm vụ của nó là ranking.

> **Đính chính quan trọng:** câu "gold item tuyệt đối không được xuất hiện trong prompt" **không áp dụng được cho `LM_Rec`** — gold item *là một trong 10 candidate*, nên tên nó bắt buộc phải có trong prompt. Phát biểu đúng của yêu cầu này ở §E0.

### 4.2 Action format

Output tối thiểu, không yêu cầu rationale trong vòng lặp train:

```json
{"ranking": ["C", "A", "J", "B", "D", "E", "F", "G", "H", "I"]}
```

Yêu cầu:

- đủ 10 label;
- không lặp;
- chỉ dùng label hợp lệ A–J;
- parse được bằng deterministic parser (tái dùng lối viết của `src/rl/policy.py`: **không bao giờ raise**, output méo là tín hiệu reward chứ không phải crash);
- rationale chỉ bật ở final qualitative evaluation, không dùng trong reward mặc định.

### 4.2.1 Bắt buộc xáo thứ tự candidate theo từng rollout

Đây là bổ sung **bắt buộc**, không có trong bản kế hoạch đầu.

Candidate set cố định theo user (đúng, cần thiết cho so sánh công bằng), nhưng nếu **thứ tự trình bày** cũng cố định thì mỗi user luôn có gold ở đúng một vị trí trong mọi epoch. Với 1185 train user và ~300 step × 32 rollout ≈ 8 epoch, một LoRA 4B thừa sức nhớ "user 8 → gold ở vị trí G". Reward tăng đẹp, test tụt, và triệu chứng **giống hệt** overfit bình thường nên rất khó chẩn đoán.

Đo trên test set M1: vị trí gold phân bố đều trên 0–9 (103/93/83/91/106/103/98/101/110/105) — tức là **không** có prior vị trí toàn cục để học, nhưng prior **theo từng user** thì tồn tại và cố định.

Quy tắc:

- **Set** candidate lấy nguyên từ jsonl M1, không đổi.
- **Thứ tự** trình bày (ánh xạ item → nhãn A–J) xáo lại theo permutation seed bằng `(user_id, global_seed, step)` ở lúc train, và bằng `(user_id, global_seed)` ở lúc eval để eval vẫn tái lập được.
- Reward tính trên **item_id**, không trên nhãn: giải nhãn về item_id trước khi chấm.
- Log thêm `gold_label_entropy` mỗi eval_step. Nếu tụt về 0 nghĩa là xáo đang hỏng.

Lưu ý: `LM_Mem` ở Plan gốc **không** gặp vấn đề này vì nó candidate-blind (§5.3 Plan gốc). Đây là rủi ro riêng của extension.

### 4.2.2 Adapter về format của pipeline gốc

`LLMReranker` hiện có nhận `{"scores":[{"item_id","score","rationale"}]}` và `MemRecAgent.rerank()` sắp xếp theo `score` giảm dần. Extension sinh permutation A–J, nên **phải** có adapter một chiều:

```text
permutation ["C","A",...]  →  giải nhãn → item_id  →  score = 1 - rank/10
```

Adapter nằm ở `src/rl/rec/policy.py`. Đúng như §5, **không** sửa `src/models/reranker_llm.py` — nhờ vậy mọi số baseline cũ vẫn tái lập được bằng lệnh gốc (§10.4 Plan gốc).

### 4.3 Reward

Dùng reward trực tiếp từ vị trí gold item trong permutation, không cần frozen reward ranker:

```text
r = NDCG@10
    + 0.2 * Hit@1
    - 1.0 * malformed
    - 0.5 * duplicate_or_missing
    - 0.05 * over_length
```

Định nghĩa chính xác từng hạng (bản đầu để trống, dễ implement lệch nhau giữa train và eval):

| Hạng | Định nghĩa |
|---|---|
| `NDCG@10` | `1 / log2(rank_gold + 2)`, `rank_gold` 0-indexed **theo item_id sau khi giải nhãn**. Với 1 item liên quan và 10 candidate, giá trị luôn > 0 — đây chính là lý do chọn @10 thay vì @5. |
| `Hit@1` | `1.0` nếu `rank_gold == 0`, ngược lại `0.0`. |
| `malformed` | `1.0` nếu parser không lấy được permutation nào. Loại trừ lẫn nhau với `duplicate_or_missing`. |
| `duplicate_or_missing` | `1.0` nếu parse được nhưng permutation lặp nhãn, thiếu nhãn, hoặc chứa nhãn ngoài A–J. Khi đó xếp các item thiếu vào cuối theo thứ tự gốc rồi vẫn chấm NDCG — để length penalty và format penalty không cộng dồn lên cùng một lỗi. |
| `over_length` | `max(0, (n_completion_tokens - 96) / 32)`. Khớp với `max_completion_length = 96` ở §E2. Bản đầu không nêu đơn vị; Plan gốc §5 dùng "mỗi 100 token vượt budget 400", ở đây budget nhỏ hơn nhiều nên đơn vị phải nhỏ theo. |

Log **riêng** `r_ranking` và `r_format`, không gộp (xem §9.4).

Lý do dùng `NDCG@10` khi train:

- candidate set có 10 item;
- reward vẫn phân biệt được gold ở vị trí 6–10;
- ít group có reward bằng nhau hơn so với chỉ dùng Hit@1 hoặc NDCG@5.

Metric báo cáo cuối vẫn giữ đúng Plan gốc:

- H@{1,3,5}
- N@{3,5}
- token/query
- latency/query

### 4.4 Model mặc định

| Vai trò | Mặc định | Fallback |
|---|---|---|
| Frozen `LM_Mem` | best GRPO checkpoint từ Plan gốc | best SFT checkpoint nếu GRPO thất bại |
| Trainable `LM_Rec` | Qwen3-4B-Instruct + LoRA | Qwen2.5-3B-Instruct |
| Teacher SFT | gpt-4o-mini, sinh mới trên candidate cố định của M1 | ~~reuse Stage-ReRank conversations từ M0~~ — **không dùng được, xem dưới** |
| Final external evaluator | gpt-4o-mini / original MemRec protocol | giữ nguyên |

Không dùng GPT-4o-mini trong vòng lặp GRPO.

> **Đính chính: không thể tái dùng Stage-ReRank conversations của M0 làm teacher.** Ba lý do độc lập, mỗi lý do đủ để loại:
>
> 1. **Sai bộ candidate.** M0 sinh negative bằng `RandomState(hash(thread.ident))`, nên ranking của teacher là ranking trên một danh sách 10 item **khác hẳn** danh sách trong `stager_books_*.jsonl`. Không ánh xạ được.
> 2. **Nhiễm ground-truth.** Run `m0_memrec_full_1k` để `enable_stage_w` mặc định `True`, nên trong lúc eval có ghi memory từ GT click. Item memory in ra trong prompt Stage-ReRank của user sau có thể đã bị cập nhật bởi GT click của user trước.
> 3. **Sai format.** Output M0 là `{"scores":[...]}` kèm rationale, không phải permutation A–J; và chỉ phủ 1000 user của M0 chứ không phủ 1185 train user của M1.
>
> Chi phí sinh mới: 1185 train user × 1 lời gọi ≈ 1.2M input token ≈ **$0.3**. Rẻ hơn nhiều so với rủi ro dùng dữ liệu bẩn.

---

## 5. File mới cần tạo

```text
src/rl/rec/
├── __init__.py
├── dataset.py              # load materialized memory + fixed candidates
├── policy.py               # prompt, parser, validation ranking permutation
├── reward.py               # NDCG@10, Hit@1, format penalties
├── sft.py                  # SFT warm-start cho LM_Rec
├── train_grpo.py           # GRPO/Rank-GRPO entrypoint
├── eval.py                 # ma trận 4 configuration + robustness
└── materialize_memory.py   # chạy frozen LM_Mem một lần và cache output

configs/rl/rec/
├── e1_materialize.yaml
├── e2_sft.yaml
├── e3_grpo.yaml
└── e4_eval.yaml

scripts/rl/rec/
├── 00_check_gate.sh
├── 01_materialize_memory.sh
├── 02_build_sft_data.sh
├── 03_train_sft.sh
├── 04_train_grpo.sh
└── 05_eval_matrix.sh

data/rl/rec/
├── books_train.jsonl       # = jsonl M1 + trường m_collab đã materialize
├── books_val.jsonl
├── books_test.jsonl
└── memory_variants/        # m_collab của prompted / SFT / empty / shuffled (chỉ để eval)

docs/
├── RL_LM_REC_EXTENSION.md
└── RESULTS_LM_REC_EXTENSION.md
```

Mọi code extension nằm dưới `src/rl/rec/`. Không sửa hành vi của `src/models/reranker_llm.py` hoặc đường eval gốc.

### 5.1 Bắt buộc tái dùng module của M1, không viết lại

M1 đã tồn tại và đã test; extension **phải** import chứ không được implement lại — hai bản implement sẽ trôi khỏi nhau và phá tính so sánh được:

| Module M1 | Extension dùng để làm gì |
|---|---|
| `src/rl/splits.py` | Split user + `sample_candidates`. **Không sinh candidate mới.** |
| `src/rl/dataset.py` | `load_records` / `write_records` / `backfill` |
| `src/rl/leakage.py` | `gold_leak_reason` — màn lọc rò, cần chỉnh phạm vi cho `LM_Rec` (§E0) |
| `src/rl/env.py` | `GraphSnapshot` để đọc item memory |
| `src/rl/policy.py` | Lối viết parser chịu lỗi; **không** tái dùng prompt (khác nhiệm vụ) |

### 5.2 Bỏ `fixed_candidates_books.jsonl`

Bản đầu đề xuất một file candidate riêng. **Không tạo.** Candidate đã nằm trong `data/rl/stager_books_*.jsonl` (trường `candidates`) và đã có test tái lập trong `tests/rl/test_no_leakage.py`. Một file thứ hai chỉ tạo ra hai nguồn sự thật có thể lệch nhau — đúng loại lỗi mà bug #5 của M0 đã gây ra.

Thay vào đó: E0 tính **SHA-256 của cột `candidates`** trong cả 3 split, ghi vào `docs/RESULTS_LM_REC_EXTENSION.md`, và mọi run sau kiểm tra lại hash này trước khi chạy.

---

## 6. Roadmap

# E0 — Gate, freeze và materialize

**Mục tiêu:** tạo môi trường tĩnh cho RL `LM_Rec`.

- [ ] Chạy `00_check_gate.sh`; fail ngay nếu thiếu artifact Plan gốc.
- [ ] Load best `LM_Mem` checkpoint ở chế độ eval, disable gradient.
- [ ] Materialize một `M_collab` cho mỗi train/val/test example, **decode greedy** (`temperature=0`, `do_sample=False`).
- [ ] Lưu checkpoint SHA, config SHA, git SHA và SHA-256 của cột `candidates` (§5.2).
- [ ] **Màn lọc rò — phạm vi đã sửa, xem khung dưới.**
- [ ] Tạo thêm các memory variant chỉ phục vụ eval: prompted, SFT, empty, shuffled-user, truncated-50%.

> ### Đính chính màn lọc rò cho `LM_Rec`
>
> Mục gốc ghi *"Kiểm tra không có gold item id/title trong instruction hoặc memory prompt"*. Với `LM_Rec` yêu cầu này **không thể đạt và cũng không nên đạt**:
>
> - **Tên gold BẮT BUỘC có trong prompt** — nó là 1 trong 10 candidate. Đây là bản chất bài toán ranking, không phải rò rỉ.
> - **Instruction của InstructRec được sinh ra từ chính item đích.** Đo trên 879 test user có tên đủ dài: 2.2% có shingle 3-từ của tên sách xuất hiện nguyên văn trong instruction, và phần còn lại diễn giải lại nội dung sách (ví dụ user 1: *"comprehensive look at James Dean's life... statements from friends and colleagues, descriptions of his last hours"*). Đây là **thiết kế của benchmark**, không phải bug của ta. Instruction được cấp **như nhau cho cả 4 configuration**, nên nó không làm lệch so sánh giữa các dòng; nhưng nó **kéo trần headroom xuống** và **phải ghi vào Limitations**.
>
> Ba kiểm tra thay thế, đúng phạm vi:
>
> 1. `M_collab` không được nêu tên/id gold — tái dùng `src.rl.leakage.gold_leak_reason` nhưng **chỉ chạy trên trường `m_collab`**, không chạy trên toàn prompt.
> 2. Gold không được đánh dấu khác các candidate còn lại: cùng template, cùng độ dài trường, không có cờ/thứ tự đặc biệt.
> 3. Gold không được nhận ra qua vị trí: xác nhận permutation nhãn của §4.2.1 đang hoạt động (`gold_label_entropy` gần `log2(10)`).

**DoD:**

- Ba file train/val/test tồn tại, số dòng khớp M1: **1185 / 149 / 993**.
- SHA-256 cột `candidates` khớp giá trị đã ghi ở E0.
- `LM_Mem` không có gradient (assert `requires_grad == False` trên mọi param).
- **Materialize đúng một lần rồi hash và đóng băng.** Bản đầu yêu cầu *"re-run materialization trên 100 mẫu cho hash giống nhau 100%"* — yêu cầu này quá chặt và sẽ fail vì lý do không liên quan tới tính đúng đắn: vLLM/HF sinh theo batch nên kernel reduction đổi thứ tự theo cách batch được ghép, nên ngay cả greedy decode cũng không bit-exact giữa hai lần chạy khác batch. Thay bằng:
  - materialize một lần, ghi ra file, **hash file và không bao giờ sinh lại** — đây mới là ràng buộc thật sự cần (môi trường tĩnh);
  - kiểm tra tính tất định ở mức *có ý nghĩa*: re-run 100 mẫu **với đúng batch composition cũ** phải khớp 100%; re-run với batch khác thì yêu cầu ≥ 95% khớp chuỗi và 100% khớp tập facet sau khi chuẩn hoá.
  - nếu tỉ lệ khớp < 95%: nghi decode chưa greedy, dừng và kiểm tra config trước khi đi tiếp.

---

# E1 — Baseline và SFT warm-start

**Mục tiêu:** có local `LM_Rec` biết format ranking trước khi RL.

- [ ] Reproduce frozen `LM_Rec` baseline trên materialized dataset.
- [ ] Sinh teacher ranking bằng gpt-4o-mini hoặc reuse conversation sạch từ M0.
- [ ] Chỉ giữ teacher output parse hợp lệ và có đủ permutation A–J.
- [ ] SFT LoRA 2 epoch, learning rate khởi điểm `1e-5`.
- [ ] Eval trên val với deterministic decoding.

**DoD:**

- Format-valid rate ≥ 99%.
- SFT `LM_Rec` ≥ base local `LM_Rec` trên val NDCG@5.
- Có checkpoint `checkpoints/rl/rec/sft_books/`.
- **Đo luôn A′ và B′ trên test** (SFT-4B với prompted memory và với RL memory). Hai dòng này là control khử confound của §E3; đo ngay ở đây vì checkpoint đang sẵn, đợi tới E3 thì phải load lại.
- Ghi baseline và SFT vào `docs/RESULTS_LM_REC_EXTENSION.md`.

Nếu SFT không vượt base sau hai cấu hình hợp lý: dừng GRPO và chẩn đoán dataset/prompt trước.

---

# E2 — GRPO cho `LM_Rec`

**Mục tiêu:** tối ưu ranking trực tiếp trong khi `LM_Mem` giữ nguyên.

Cấu hình khởi điểm:

| Tham số | Giá trị |
|---|---:|
| `num_generations` | 8 |
| `temperature` | 1.0 |
| `max_completion_length` | 96 |
| `learning_rate` | 1e-6 |
| `max_steps` | 300 |
| `beta` | 0.005 |
| `loss_type` | `dr_grpo` hoặc rank-level objective nếu đã implement ổn định |
| LoRA `r/alpha/dropout` | 32 / 64 / 0.05 |

Logging bắt buộc:

- `reward_mean`, `reward_std`;
- H@1, NDCG@5 val;
- format-valid rate;
- duplicate/missing rate;
- completion length;
- `% group std=0`;
- VRAM peak và wall-clock.

Quy trình:

- [ ] Unit test reward bằng các permutation tính tay.
- [ ] Unit test parser với ≥20 ca permutation méo viết tay (thiếu nhãn, lặp nhãn, nhãn lạ, JSON cụt, có rationale thừa) — theo đúng chuẩn đã dùng ở M1.
- [ ] **Test xáo nhãn**: cùng một example, hai permutation khác nhau phải cho cùng reward khi rank của gold theo `item_id` như nhau.
- [ ] Dry-run 20 step với model 0.5B trên GPU rẻ.
- [ ] Test checkpoint/resume.
- [ ] Smoke run 20 step với model thật.
- [ ] Full run 300 step.
- [ ] Eval best checkpoint trên test bằng candidate cố định của M1 (kiểm SHA trước khi chạy).
- [ ] **Test đảo thứ tự** (§9.4bis): eval lại test set với permutation nhãn khác; chênh H@1 phải trong nhiễu.

**DoD:**

- GRPO `LM_Rec` > SFT `LM_Rec` trên val và test NDCG@5.
- Format-valid rate ≥ 99%.
- Không có duplicate/missing label.
- H@1 không giảm khi NDCG tăng.
- Training curve không collapse.
- **Test đảo thứ tự đạt**: chênh H@1 giữa hai permutation nhãn ≤ 3 điểm (§9.4bis). Không đạt nghĩa là model bám vị trí, kết quả không dùng được dù reward có đẹp.

**Kill criteria:**

- Sau 150 step reward phẳng và `% group std=0 > 60%`: tăng temperature hoặc oversample user trung bình-khó.
- Reward tăng nhưng test metric không tăng: kiểm tra overfit candidate artifacts.
- Output dài hoặc sinh rationale ngoài format: giảm max length, tăng format penalty.
- Hai full run không vượt SFT: giữ negative result và chuyển sang E3/E4; không tiếp tục đốt compute.

---

# E3 — Ma trận đánh giá bắt buộc

Chạy cùng test users (993 user của M1), cùng fixed candidates, cùng seed.

> **Đính chính lớn: ma trận 4 dòng gốc bị confound.** Dòng A/B dùng `LM_Rec` = **gpt-4o-mini**; dòng C/D dùng `LM_Rec` = **Qwen3-4B + LoRA**. Nên `D > B` đổi **hai** thứ cùng lúc: (i) đổi model ranker gpt-4o-mini → 4B local, (ii) thêm GRPO. Nếu `D > B` ta **không kết luận được** là RL-cho-ranker có tác dụng hay chỉ là "4B đã fine-tune xấp xỉ gpt-4o-mini". Đây đúng là loại lỗi mà Plan gốc §5.3 cảnh báo ở chỗ khác.
>
> Sửa bằng cách thêm **hai dòng control cùng base model**. Chúng gần như miễn phí: checkpoint SFT đã có sẵn từ E1, chỉ cần thêm một lượt eval trên test.

| ID | `LM_Mem` | `LM_Rec` | Ý nghĩa |
|---|---|---|---|
| A | prompted | gpt-4o-mini (frozen) | baseline gốc — **phải đo lại trên candidate M1**, không bê từ M0 |
| B | RL | gpt-4o-mini (frozen) | đóng góp chính của Plan gốc |
| A′ | prompted | **SFT-4B (frozen)** | control: cùng base model với C, chưa có GRPO |
| B′ | RL | **SFT-4B (frozen)** | control: cùng base model với D, chưa có GRPO |
| C | prompted | GRPO-4B | ranker-only RL |
| D | RL | GRPO-4B | sequential extension |

Nhờ vậy tách được hai hiệu ứng:

```text
hiệu ứng đổi model ranker  =  B′ − B
hiệu ứng GRPO cho ranker   =  D − B′      ← đây mới là câu hỏi của extension
hiệu ứng RL memory dưới ranker mới =  D − C
```

Báo cáo:

- H@1, H@3, N@3, H@5, N@5;
- token/query; latency/query;
- GPU-hour train; API cost;
- delta A→B, A→C, **B′→D** (chính), B→D (có ghi rõ là confounded);
- bootstrap 95% CI trên per-user metric, **n = 993**;
- riêng `r_ranking` và `r_format` (§4.3).

Kết luận extension chỉ được coi là thành công khi:

```text
D > B′   (cùng base model, chỉ khác GRPO)
```

trên NDCG@5 hoặc H@1 mà không làm giảm rõ metric còn lại, **và** khoảng tin cậy 95% bootstrap không chứa 0.

`D > B` là điều kiện phụ, báo cáo kèm cảnh báo confound; **không** được dùng làm phát biểu chính.

Không yêu cầu:

```text
D > B > C > A
```

vì gain của memory và ranker có thể không cộng tuyến tính.

> **Lưu ý về sức mạnh thống kê:** n = 993, H@1 quanh 0.5 → sai số chuẩn ≈ 1.6%, nên khoảng tin cậy 95% rộng ≈ ±3.1 điểm. Chênh lệch dưới ~3 điểm H@1 **không** phân biệt được. Nếu `D − B′` được dự đoán nhỏ hơn thế, hãy dùng paired bootstrap trên cùng user (tương quan cao nên chặt hơn nhiều) và nói rõ đang dùng paired test.

---

# E4 — Robustness và co-adaptation

Dùng cùng `LM_Rec` đã train ở E2, không train lại:

| Memory input | Mục đích |
|---|---|
| RL `M_collab` | in-distribution |
| SFT `M_collab` | đổi policy memory |
| Prompted `M_collab` | cross-style |
| Empty memory | dependency test |
| Shuffled-user memory | chống shortcut |
| Truncate 50% facets | robustness |

Metric chính:

- phần trăm gain giữ lại so với RL memory;
- H@1/NDCG@5;
- sensitivity gap giữa đúng memory và shuffled-user memory.

**DoD:**

- Prompted/SFT memory giữ ≥ 60% gain của RL memory.
- Shuffled-user memory phải kém đúng memory rõ ràng.
- Empty memory phải kém đúng memory; nếu không, `LM_Rec` đang bỏ qua memory.
- Ghi ít nhất 20 ví dụ định tính: đúng hơn, sai hơn, bỏ qua memory, phụ thuộc style.

Nếu chỉ RL memory hoạt động còn prompted/SFT tụt mạnh, báo cáo rõ là **co-adaptation**, không tuyên bố `LM_Rec` tổng quát.

---

## 7. Ablation tùy chọn

Chỉ làm sau khi E0–E4 hoàn tất:

1. Train `LM_Rec` trên mixture:
   - 50% RL memory;
   - 25% prompted memory;
   - 25% empty/corrupted memory.

2. `NDCG@10` reward vs `NDCG@5 + Hit@1`.

3. SFT-only vs GRPO.

4. GRPO sequence-level vs rank-level objective.

5. Books → MovieTV zero-shot với frozen RL `LM_Mem`.

Ưu tiên mixture-memory trước vì trực tiếp xử lý co-adaptation.

---

## 8. Bảng kết quả cần điền

### Bảng chính

Test set: 993 user của M1, candidate cố định, SHA-256 ghi ở E0.

| ID | Config | H@1 | H@3 | N@3 | H@5 | N@5 | Tokens | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Prompted Mem + gpt-4o-mini Rec | | | | | | | |
| B | RL Mem + gpt-4o-mini Rec | | | | | | | |
| A′ | Prompted Mem + SFT-4B Rec | | | | | | | |
| B′ | RL Mem + SFT-4B Rec | | | | | | | |
| C | Prompted Mem + GRPO-4B Rec | | | | | | | |
| **D** | **RL Mem + GRPO-4B Rec (sequential)** | | | | | | | |

| Delta | Giá trị | 95% CI (paired bootstrap) | Diễn giải |
|---|---:|---|---|
| A→B | | | đóng góp chính của Plan gốc |
| B→B′ | | | đổi model ranker (confound cần tách) |
| **B′→D** | | | **hiệu ứng GRPO cho ranker — phát biểu chính của extension** |
| C→D | | | RL memory dưới ranker đã train |
| B→D | | | báo cáo kèm cảnh báo confounded |

### Bảng robustness

| Memory source khi eval | H@1 | N@5 | Gain giữ lại | Kết luận |
|---|---:|---:|---:|---|
| RL memory | | | 100% | |
| SFT memory | | | | |
| Prompted memory | | | | |
| Empty memory | | | | |
| Shuffled-user memory | | | | |
| Truncated memory | | | | |

---

## 9. Rủi ro chính

### 9.1 `LM_Rec` bỏ qua memory

Dấu hiệu: RL/empty/shuffled memory cho kết quả gần nhau.

Xử lý:

- kiểm tra prompt placement;
- thêm training mixture có contrast đúng-memory vs shuffled-memory;
- không tăng compute trước khi chứng minh model đọc memory.

### 9.2 Co-adaptation theo style

Dấu hiệu: tốt với RL memory nhưng giảm mạnh với prompted/SFT memory.

Xử lý:

- train trên mixture memory;
- randomize thứ tự facet;
- normalize format trước khi đưa vào `LM_Rec`.

### 9.3 Học artifact của candidate sampling

Dấu hiệu: train/val tăng nhưng test hoặc candidate-resample giảm mạnh.

Xử lý:

- fixed deterministic candidates;
- một candidate-resample test chỉ dùng ở cuối;
- không dùng thread ID hoặc process ID làm seed.

### 9.4bis Nhớ vị trí gold theo user — rủi ro nguy hiểm nhất của extension

Bản kế hoạch đầu không có mục này. Nó nguy hiểm vì triệu chứng **trùng khít với overfit thường**, nên rất dễ chẩn đoán nhầm rồi đi sửa learning rate trong khi nguyên nhân nằm ở dữ liệu.

Dấu hiệu: train reward tăng đều, val/test đứng yên hoặc tụt; và đặc biệt — nếu xáo lại thứ tự candidate lúc eval thì điểm **sụp**.

Nguyên nhân: candidate set cố định theo user là đúng, nhưng nếu thứ tự trình bày cũng cố định thì gold của mỗi user luôn ở đúng một nhãn. Với 1185 train user và ~8 epoch, LoRA 4B thừa sức nhớ bảng tra đó mà không cần đọc memory hay instruction.

Phòng: §4.2.1 — xáo thứ tự theo `(user_id, global_seed, step)`, chấm reward theo `item_id`.

Phát hiện: **test đảo thứ tự**. Eval best checkpoint hai lần trên cùng test set, chỉ khác permutation nhãn. Chênh lệch H@1 giữa hai lần phải nằm trong nhiễu. Nếu lệch > 3 điểm thì model đang bám vào vị trí, không phải nội dung — kết quả không dùng được.

### 9.5 Rò rỉ kế thừa từ benchmark (không sửa được, phải công bố)

Instruction của InstructRec được sinh từ chính item đích, nên nó luôn mô tả đáp án ở mức ngữ nghĩa. Điều này áp dụng **đồng đều cho cả 6 dòng** của ma trận §E3 nên không làm lệch so sánh, nhưng:

- nó kéo trần headroom xuống — phần lớn tín hiệu có thể đã nằm trong instruction chứ không phải trong memory;
- vì thế `A′ → B′` (hiệu ứng của RL memory) có thể nhỏ một cách giả tạo;
- **phải viết vào Limitations** của cả luận văn chính lẫn extension.

Kiểm chứng rẻ, nên làm ở E0: chạy một dòng "instruction-only, no memory, no item memory" trên test. Nếu nó đã đạt gần bằng dòng A thì toàn bộ ma trận đang đo trên một dải rất hẹp — biết sớm còn hơn biết lúc viết luận văn.

### 9.4 Format reward chi phối ranking reward

Dấu hiệu: format-valid 100% nhưng NDCG không tăng.

Xử lý:

- format penalty chỉ là hard guard;
- log riêng ranking reward và format reward;
- không gộp thành một metric duy nhất trong báo cáo.

---

## 10. Ngân sách và điều kiện dừng

Ước tính extension LEAN:

| Hạng mục | H100-hour | API ($) |
|---|---:|---:|
| Materialize + baseline (gồm dòng A, A′ đo lại) | 1–2 | ~2 |
| Teacher SFT (gpt-4o-mini, 1185 user) | 0 | ~0.3 |
| SFT | 3–4 | 0 |
| GRPO runs | 12–20 | 0 |
| Eval + robustness (6 dòng ma trận + 6 memory variant) | 3–5 | ~4 |
| Dự phòng | 5 | ~2 |
| **Tổng** | **24–36** | **~$8** |

**Đối chiếu với ngân sách toàn đồ án (Plan gốc §11):**

| | H100-hour | Ước chi phí |
|---|---:|---:|
| Phase 1 (M0–M4, M7a) | ~50 | ~$190 |
| Phase 2 M5 Ưu tiên 1 (bắt buộc trước extension) | ~2 | ~$2 |
| **Extension E0–E4** | **24–36** | **~$75–125** |
| Cộng dồn | **76–88** | **~$270–320** |
| Phase 2 đầy đủ (M5 +35, M6 +70) nếu vẫn muốn làm | +103 | +$310 |

Nghĩa là extension và Phase 2 đầy đủ **loại trừ nhau** trên thực tế. Phải chọn, và chọn sau M7a chứ không phải bây giờ. Cổng ở §3 yêu cầu còn ≥36h chính là để tránh bắt đầu rồi bỏ dở.

Nguyên tắc:

- Mọi plumbing chạy bằng model 0.5B trên GPU rẻ.
- Không dùng H100 để build JSONL, parse, vẽ hình hoặc viết báo cáo.
- Dừng extension nếu hai full GRPO runs không vượt SFT.
- Không mở joint training dù kết quả E3 hấp dẫn; joint training thuộc Future Work.
- **Nếu phải chọn giữa extension và M5 đầy đủ: ưu tiên M5.** M5 củng cố đóng góp chính (RL cho `LM_Mem`); extension mở một đóng góp phụ. Một đóng góp chính vững hơn tốt hơn hai đóng góp đều lung lay — đúng tinh thần §2.5 của Plan gốc.

---

## 11. Definition of Done toàn extension

Extension hoàn tất khi có:

- [ ] Checkpoint SFT và GRPO `LM_Rec`.
- [ ] Ma trận 4 configuration trên cùng fixed test set.
- [ ] Bootstrap 95% CI.
- [ ] Robustness theo nguồn memory.
- [ ] Phân tích co-adaptation.
- [ ] Compute/API cost thực tế.
- [ ] Một kết luận rõ ràng thuộc một trong ba trường hợp:

### Kết quả A — Positive

Sequential RL cải thiện thêm so với RL `LM_Mem` đơn độc.

### Kết quả B — Saturation

RL `LM_Mem` đã lấy phần lớn gain; RL `LM_Rec` không cải thiện đáng kể.

### Kết quả C — Co-adaptation/negative

RL `LM_Rec` chỉ tốt với đúng style memory hoặc không vượt SFT.

Cả ba trường hợp đều hợp lệ nếu protocol sạch và phân tích đầy đủ.

---

## 12. Giao ước với coding agent

1. Đọc và hoàn tất `docs/RL_PLAN.md` trước file này. **Khi hai file mâu thuẫn, Plan gốc thắng** — extension là phần phụ, không được sửa phạm vi của đóng góp chính.
2. Không tự động bắt đầu extension chỉ vì M4 đã có checkpoint; phải qua M7a **và** M5 Ưu tiên 1 (§3).
2b. Tái dùng module M1 theo §5.1; không implement lại split, candidate sampling hay màn lọc rò.
3. Không cập nhật hoặc overwrite checkpoint `LM_Mem`.
4. Không joint-update hai model.
5. Không sửa đường baseline/eval gốc.
6. Mỗi milestone extension dùng branch riêng: `rl-rec/e0-materialize`, `rl-rec/e2-grpo`, ...
7. Sau mỗi milestone, append kết quả vào `docs/RESULTS_LM_REC_EXTENSION.md`.
8. Mọi run lưu config, seed, git SHA, candidate hash và memory checkpoint SHA.
9. Nếu thực tế code mâu thuẫn với file này: dừng, sửa plan trong cùng PR và ghi lý do.
10. Main contribution của luận văn vẫn là RL `LM_Mem`; không viết lại narrative chính chỉ vì extension đạt gain lớn.
