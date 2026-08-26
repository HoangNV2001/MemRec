# BUFFERED_PROPAGATION_PLAN.md — Buffered Item-Side Propagation

> **Trạng thái:** dừng ở feasibility gate P1 trên InstructRec-Books. Không có
> P2 oracle, không có LLM request và không có memory write từ protocol này.
> Kết quả/nhật ký: [RESULTS.md](RESULTS.md), [PROGRESS.md](PROGRESS.md).
>
> **Phân biệt protocol:** [MULTIHOP_PLAN.md](MULTIHOP_PLAN.md) đã bác bỏ
> *read-side raw 2-hop context under equal budget*. [CANDIDATE_EVIDENCE_PLAN.md](CANDIDATE_EVIDENCE_PLAN.md)
> đã bác bỏ candidate-conditioned evidence ở ranking time. Tài liệu này kiểm
> tra một giả thuyết Stage-W/write-side mới, do đó không "cứu" hay retune các
> protocol đã đóng.

## 1. Pivot và phạm vi claim

Đề xuất trong [brainstorming.md](brainstorming.md) chỉ ra đúng khác biệt khái
niệm: limitation của MemRec nói về **asynchronous collaborative propagation**
ở Stage-W, trong khi MH2 thay node trong Stage-R context. Negative result MH2
không tự nó bác bỏ write-side propagation.

Tuy nhiên, interaction file của dataset này chỉ chứa thứ tự **bên trong từng
user**, không chứa một global event clock giữa users. Vì vậy không được replay
"A update B rồi B update C" và cũng không được claim dynamic/asynchronous
multi-hop propagation trên InstructRec-Books hiện có.

Biến thể duy nhất được phép kiểm tra ở dataset này là:

> **Static, split-safe, source-only item-memory augmentation:** histories của
> user thuộc train source split tạo packet candidate-blind; packet chỉ có thể
> đến item endpoint qua graph source-only; evaluation user chỉ đọc immutable
> item overlay đã dựng xong.

Đây **không** phải implementation hay reproduction của RecNet: không có router
agent/RL, không dùng receiver agent để quyết định theo event stream, không
claim evolution động; item-only được chọn để tránh user-profile drift. Nếu một
dataset có global timestamp được dùng sau này, protocol động phải được mở lại từ
P0, không được suy diễn từ kết quả static này.

## 2. Câu hỏi và protocol điều kiện

**RQ-P:** Có đủ đường đi source-only, candidate-blind để buffered item-side
propagation có headroom tổng thể, trước khi chi phí LLM được bỏ ra không?

| Probe | Mục đích | Điều kiện để đi tiếp |
|---|---|---|
| P0 — temporal audit | Xác định có global cross-user clock để replay động hay không. | Chỉ nhánh dynamic cần pass strict global clock. |
| P1 — structural route coverage | Dựng ledger `origin user -> anchor item -> peer user -> endpoint item` hoàn toàn từ source users; candidate/gold chỉ đọc sau cùng để đo coverage. | Với static P2: support >=2 phải cover ít nhất 30% validation gold **và** ít nhất 30% validation user có >=1 candidate endpoint. |
| P2 — candidate-blind semantic oracle | Chỉ khi P1 pass: giữ base item memory immutable, thêm tối đa 1–2 overlay packet/item; gold chỉ dùng sau construction để chọn bounded oracle. | Oracle >= +0.05 NDCG@5, CI lower > +0.01. |
| P3 — implementation | Chỉ khi P2 pass: original 1-hop write, naive item write, buffered/support-gated item overlay; selector/threshold khóa trước validation report. | Actual buffered arm >= +0.02 và CI lower > 0 trước locked test. |
| P4 — locked test | Chỉ config hash đã khóa từ P3; paired test report và breakdown low-degree item. | Báo cáo cuối, không tuning. |

Ngưỡng P1 được ghi ở đây **sau feasibility scan đầu tiên, trước mọi P2 LLM
call**; vì vậy P1 là reconnaissance chứ không được trình bày là preregistered
confirmatory result. Mục đích ngưỡng là loại trường hợp coverage quá thấp để
một gain toàn tập rõ ràng là không thực tế, chứ không phải tối ưu threshold.

## 3. Leakage contract nếu P2/P3 từng được admission

- Source user là đúng cohort `mh0_control_train` và phải thuộc train split;
  val/test user, candidate list, instruction, gold và ranking outcome không
  được đọc khi sinh packet, route hay merge.
- Source history chỉ gồm interaction trước hai held-out item của source user.
  Base item memory không bị overwrite; overlay versioned, provenance-carrying
  và có thể bỏ đi hoàn toàn.
- Ledger được materialize trước evaluation; support là số `origin_user` độc lập,
  không phải số witness path nhân bản từ cùng một source.
- Packet không được biết candidate item. Candidate/gold chỉ được dùng sau khi
  ledger đóng để báo coverage hoặc (ở P2) chọn oracle; selection pass và report
  pass phải tách cache/call như MH2.
- Không dùng graph cap, support threshold, semantic score hoặc overlay budget
  được tune theo P2/P3 report. Các arm có cùng candidate list/ranker/retry
  policy; delta paired theo cùng user set với bootstrap CI.

## 4. P0 — temporal admissibility (complete, fail nhánh dynamic)

Command:

```bash
python -m src.propagation.p0_temporal_audit --config configs/multihop/mh0_books.yaml --force
```

| Check | Kết quả |
|---|---:|
| Events / users | 207,759 / 7,377 |
| Distinct timestamp values | 2,226 |
| Timestamp values shared by >1 user | 1,479 |
| Events at cross-user-ambiguous timestamps | 207,012 (99.64%) |
| Largest user count at one timestamp | 7,377 |
| Global file-order inversions | 7,376 |
| Per-user monotonic histories | 7,377 / 7,377 |
| Decision | **dynamic cross-user propagation inadmissible** |

Artifact: `data/propagation/p0_temporal_audit.json` (derived, gitignored),
input SHA256 `891d4007…efc24`. Timestamp hoạt động như ordinal theo user: nó đủ
để construct history mỗi user, không đủ để xác định event của user A xảy ra
trước event của user B. Không có phép tie-break từ file order nào được dùng để
tạo một causal replay giả tạo.

## 5. P1 — static source-only route coverage (complete, fail)

Command:

```bash
python -m src.propagation.p1_source_route_audit --config configs/multihop/mh0_books.yaml --force
```

**Construction.** 1,185 source users là đúng materialized MH0 train cohort;
149 evaluation users là MH0 validation cohort và disjoint. Một eventual packet
của mỗi source user có thể dùng tối đa 8 anchor item gần nhất (distinct), 16
peer source user/anchor, 16 remote item/peer. Không có LLM call, semantic score,
candidate hoặc gold trong ledger. Đây là envelope đã rộng hơn one-anchor pilot,
nhưng vẫn giữ số packet dự kiến là một/source user.

| Support độc lập | Candidate slots covered | Gold covered | Users có >=1 candidate endpoint |
|---|---:|---:|---:|
| >=1 | 71 / 1,490 (4.77%) | 9 / 149 (6.04%) | 53 / 149 (35.57%) |
| >=2 | 60 / 1,490 (4.03%) | 9 / 149 (6.04%) | 46 / 149 (30.87%) |
| >=3 | 49 / 1,490 (3.29%) | 8 / 149 (5.37%) | 39 / 149 (26.17%) |

Ledger có 10,455 endpoint item, SHA256
`9e82902b117d3a6bc5e6e3ff29388a2bfdf29a0340696d387c7419f3d2d77678`.
Nhánh static fail vì coverage gold support >=2 là 6.04%, thấp xa gate 30%; dù
mọi gold được cover đều được cải thiện hoàn hảo thì biên trên lý thuyết cho
mean NDCG@5 chỉ là 9/149 = +0.0604, gần sát riêng target oracle +0.05 và không
chừa margin cho hiệu quả thực tế. Do đó P2 không có feasibility đáng để trả
chi phí LLM và không được admission.

## 6. Kết luận và hướng khả thi tiếp theo

Pivot khái niệm từ *multi-hop retrieval* sang *multi-hop memory evolution* là
hợp lý, nhưng **không khả thi để kiểm chứng claim động trên dataset hiện tại**;
biến thể static source-only cũng không đạt coverage để tiếp tục. Kết luận này
không nói buffered propagation vô ích nói chung; nó chỉ nói repository/data
split hiện có không cung cấp evidence path và time semantics cần thiết.

Một pilot temporal tách biệt trên Amazon Books 2014 hiện được ghi ở
[AMAZON_BOOKS_2014_TEMPORAL_PLAN.md](AMAZON_BOOKS_2014_TEMPORAL_PLAN.md). Nó
không đảo kết luận InstructRec này: dataset, split và strict-past batch contract
đều khác.

Nếu tiếp tục hướng này, bước đúng là đổi **data/protocol**, không tăng hop,
cap hay tuning score sau P1:

1. Chọn dataset có event-level global timestamps đáng tin cậy và tạo temporal
   train/validation/test cut; rerun P0 trước mọi LLM call.
2. Materialize event-causal packet/buffer ledger với provenance, TTL và
   independent-support counter; freeze it trước ranking evaluation.
3. Chạy item-only candidate-blind oracle P2 trước; chỉ sau headroom dương mới
   thử receiver gate / community consensus. Aggregate facets là một hypothesis
   mới và phải có oracle riêng, không được coi là cách retune P1 đã fail.
