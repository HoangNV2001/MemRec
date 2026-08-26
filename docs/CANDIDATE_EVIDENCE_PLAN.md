# CANDIDATE_EVIDENCE_PLAN.md — Candidate-Conditioned 1-Hop Evidence

> **Trạng thái:** hoàn tất ở CE1 — hard stop cho lexical candidate-conditioned
> evidence hiện tại. Đây là hướng tách biệt với selective multi-hop đã dừng ở MH2.

## 1. Câu hỏi mới

MH2 cho thấy thay node 1-hop mạnh bằng node 2-hop xa hơn không tạo headroom.
Pilot này giữ nguyên 1-hop và frozen Stage-R, rồi hỏi một câu hẹp hơn:

> Signal 1-hop có được dùng tốt hơn nếu reranker nhận evidence phù hợp với
> request hoặc với từng candidate, thay vì chỉ nhận một summary facets chung?

Đây không phải graph expansion, propagation, SFT hay RL. Evidence chỉ tồn tại
ở ranking-time; không ghi ngược vào graph/memory.

## 2. Hợp đồng đánh giá

| Thành phần | Quy tắc khóa trước CE1 |
|---|---|
| Cohort pilot | 100 user đầu tiên theo `user_id` trong 141-user MH2 locked cohort; CE0 materialize danh sách/hashes. |
| Source collaborative signal | Chỉ `stage_r_context.neighbor_snippets` của MH0 1-hop. Không C2/n-hop. |
| Stage-R | Reuse exact cached `stage_r:<user>:one_hop` từ MH2; prompt SHA256 phải khớp MH0 control. |
| Candidates | Cùng 10 candidate và thứ tự cố định/user. Candidate text được phép ở ranking-time evidence arm; gold ID/outcome tuyệt đối không vào CE0 selector. |
| Selector | Deterministic length-normalized lexical overlap; frozen 1-hop order break tie. Không embedding, LLM, tuning hay oracle. |
| Evidence budget | Tối đa 384 estimated serialized tokens/arm; 10 slot tổng cho request arm và 1 slot/candidate cho candidate arm. Mọi row ghi token estimate/node IDs. |
| Reranker | Cùng Azure deployment, schema, max completion, retry policy; hai rerank độc lập/arm/user. |
| Metric | Paired NDCG@5; 10,000 paired-user bootstrap. H@1/H@3/H@5/NDCG@3 là phụ. |

## 3. Arm pilot CE1

| Arm | Reranker input thêm ngoài frozen Stage-R facets | Mục đích |
|---|---|---|
| `baseline_one_hop` | Không thêm evidence | MemRec 1-hop reference độc lập. |
| `request_one_hop` | 10 snippet 1-hop chung, chọn theo request | Tách tác dụng của raw 1-hop evidence/context. |
| `candidate_one_hop` | Một snippet 1-hop cho mỗi candidate, chọn theo request + title + memory của chính candidate đó | Đo candidate-conditioned alignment. |

`request_one_hop` và `candidate_one_hop` có cùng số evidence slot và cùng cap
384 token; hai arm này là comparison trực tiếp cho câu hỏi alignment. Baseline
không có evidence mới, vì vậy comparison baseline chỉ là diagnostic về tác dụng
của thêm raw evidence, không phải claim equal-final-prompt-context.

## 4. Milestones và gate

### CE0 — Freeze/materialize evidence offline

- Build `data/multihop/ce0_candidate_evidence_val.jsonl` từ MH0 control và MH1
  eligibility; tạo manifest/hash.
- Assert candidate order, exact source node IDs, fixed slot count và cap token.
- Unit test chứng minh selector output không đổi khi thay gold ID trong input
  fixture; CE0 selector không nhận gold/outcome field.

**Kết quả CE0 (2026-08-25):** 100 row deterministic, artifact SHA256
`b7b296df71769da578f8191f4274c44c53dc4b073e0dd64a0b9c7bfc8504a266`; request
estimate 210–359 token, candidate estimate 226–383 token, 0 LLM call. Rerun
`--force` giữ nguyên artifact hash.

### CE1 — Independent reranking pilot

- Reuse only exact one-hop Stage-R completion from MH2; không gọi Stage-R,
  không chọn oracle và không sửa parameter sau kết quả.
- Run 100 × 3 arm × 2 repeat = 600 rerank completion; cache raw JSONL resumable.
- Report mean của hai repeat per user và paired CIs; selection/report không bị
  trộn vì pilot không có selection pass.

**Admission gate cho CE2 full validation:** candidate arm phải đồng thời (a)
vượt request arm ít nhất +0.03 NDCG@5 với CI lower > 0, và (b) vượt baseline
ít nhất +0.02. Nếu không đạt, không tune lexical parameters hoặc đổi budget
sau khi đã xem pilot; ghi negative result và dừng hướng CE hiện tại.

**Kết quả CE1 (2026-08-25):** 100/100 user được phân tích. Candidate evidence
đạt NDCG@5 0.7767, thấp hơn baseline −0.0080 (95% CI [−0.0434, +0.0281]) và
thấp hơn request evidence −0.0110 (95% CI [−0.0464, +0.0245]). Request evidence
so với baseline là +0.0030 (95% CI [−0.0329, +0.0370]). Không đạt bất kỳ phần
nào của admission gate, nên **CE2 không được admission** và không tune lexical
score, token cap hay selector sau khi đã xem pilot.

### CE2 — Full validation (conditional)

Chỉ khi CE1 pass. Materialize cohort 141 user bằng **đúng selector/config CE0**
rồi chạy independent report mới. Không dùng pilot để chỉnh score/token cap. Nếu
CE2 qua một gate được khóa trước run, mới cân nhắc một encoder semantic thay
lexical như một experiment mới có protocol riêng.

## 5. Rủi ro cần report

| Rủi ro | Guard |
|---|---|
| Evidence leak gold qua selector | CE0 chỉ nhận request + title/memory từng candidate; fixture thay gold không đổi output; gold chỉ dùng metrics sau rank. |
| Gain do thêm context, không phải conditioning | So sánh trực tiếp candidate vs request với cùng 10 slot/cap. |
| Evidence lặp giữa candidates | Ghi số unique source node; không suy luận relevance từ số slot vì slot luôn bằng nhau. |
| Stage-R variability | Reuse exact MH2 one-hop completion cho cả ba arm; report rerank có 2 repeat độc lập. |
| Pilot/tuning overfit | User cohort, selector, cap và gate được materialize/ghi trước CE1; CE2 không đổi chúng. |
| Lexical selector yếu | CE1 là feasibility gate cho mechanism/config này, không được diễn giải là chứng minh tổng quát về mọi semantic encoder. |

## 6. Artifacts

~~~
data/multihop/
  ce0_candidate_evidence_val.jsonl
  ce0_manifest.json
results/multihop/
  ce1_val_books_gpt54mini_pilot100/
    run_manifest.json
    raw_completions.jsonl
    metrics.json
~~~

`PROGRESS.md` và `RESULTS.md` đã được cập nhật sau CE0/CE1. Kết quả CE không
thay đổi kết luận MH2: multi-hop expansion vẫn là hard stop.
