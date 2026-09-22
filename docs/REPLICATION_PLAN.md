# Replication report — Temporal Transition-PPR

> Trạng thái: internal Amazon Books replication **complete / pass**. Protocol
> và config được lock trước khi materialize result. Active config:
> `configs/temporal_amazon_books_2014/transition_ppr_replication_200.yaml`,
> SHA256 `4a0aa4fee584983eaa4dedacd0633c1293e187da66739ab4ba621d7f43b62098`.
> P7-v2 test không được dùng để tune thêm; replication không retune
> alpha, restart, walk budget, depth hay prompt sau khi mở labels.

## 1. Mục tiêu

Replication phải trả lời hai câu hỏi tách biệt:

1. Gain +0,0752 có lặp lại trên một cohort user-disjoint mới của cùng Amazon
   Books hay không?
2. Sau khi internal replication pass, transition semantics có transfer sang
   một temporal book dataset khác hay không?

Primary replication trước mắt là câu 1. External dataset là phase tiếp theo,
không trộn result hai phase.

Kết luận câu 1: **có**. Effect replicate với gain lớn hơn discovery
cohort và CI lower dương rõ ràng.

## 2. Frozen method

Không tune các giá trị sau:

| Component | Frozen value |
|---|---|
| Positive rating | >= 4 |
| Candidate task | 1 novel positive + 9 deterministic uniform unseen negative |
| Graph cutoff | global validation cutoff; strict-past batches |
| Seeds | 6 recent unique items |
| Restart/stop probability | 0,15 |
| Walks / max steps | 50.000 / 64 |
| Fusion | within-candidate min–max, alpha 0,80 |
| Local ranker | Qwen3-4B-Instruct-2507 revision `cdbee75f…0554`, bf16, greedy |
| Metrics | NDCG@5 primary; Hit@5 secondary |
| Bootstrap | 10.000 paired resamples, seed fixed before labels |

Không thêm semantic filter, hub rule, path cap, learned gate hoặc oracle selector
trong primary replication.

## 3. R0 — cohort feasibility, CPU only

- Exclude mọi user từng xuất hiện trong P3–P7 calibration/test/train artifacts.
- Chọn **200** first-novel positive events sau validation cutoff bằng một hash
  salt mới được ghi trước khi đọc outcome.
- Mỗi user đúng một event, ít nhất năm positive strict-past interactions.
- Candidate pool chỉ gồm positive item first-seen trước train cutoff.
- Audit bắt buộc: duplicate candidate = 0; gold-in-history = 0; same-day read =
  0; overlap với cohort cũ = 0.
- Nếu không đủ 200 event, báo exact eligible count và giảm cohort một lần trước
  scoring; không đổi rating/history/candidate rule để cứu sample size.

Output: prepared cohort + exclusion ledger + manifest/hash. LLM requests = 0.

## 4. R1 — graph smoke, CPU only

- Dựng graph đúng frozen cutoff và contract.
- Smoke 20 target event, 5.000 walk/event; replay một event hai lần phải giống
  bit-for-bit.
- Report one-step/PPR nonzero slots, seed cap, graph stats và runtime.
- Không dùng gold để score hoặc chọn path. Fail bất kỳ leakage/determinism check
  nào thì hard stop trước GPU.

## 5. R2 — local ranker, smoke-first GPU

- 200 event × (Stage-R + rerank) = 400 primary generations.
- Retry reserve tối đa 40; journal physical attempt trước mỗi generation.
- Smoke 20–30 event bằng đúng model/prompt/schema/candidate order full run và
  bắt buộc reuse smoke cache.
- Permutation-completion cap giữ ở 5 cho toàn run; bất kỳ parser repair, OOM,
  config/hash mismatch hoặc peak vượt card duy nhất đều fail.
- Dùng một H100 vì model chỉ cần ~8 GiB và contract là tensor-parallel 1. Quyền
  có hai GPU không tự động cho phép đổi worker/model placement.
- Xong mỗi invocation phải unload và xác nhận VRAM về baseline.

## 6. R3 — locked evaluation

Alpha 0,80 được apply trực tiếp; không có calibration grid mới. Labels chỉ mở
sau khi graph scores, local journal và mọi hash đã khóa.

Primary replication pass khi đồng thời:

- mean delta NDCG@5 >= +0,03;
- paired bootstrap 95% CI lower > 0.

Report bắt buộc: Hit@5, improved/worsened/unchanged, one-step/PPR gold và
negative reachability, degree/seed-length buckets, request/token/runtime/VRAM.
Oracle best-of chỉ là non-deployable headroom appendix.

## 7. R4 — robustness ablations, chỉ sau primary result

Ablation không được thay primary decision và phải chạy trên validation hoặc
cohort riêng, không tune replication test:

- one-step only vs multi-hop PPR;
- terminal-state estimator vs exact/sparse power-iteration PPR;
- recent seeds 1/3/6;
- candidate pools 10 vs 100 nếu có retrieval protocol thực tế.

Các ablation phải có run ID/config/output riêng và pre-register grid trước khi
mở labels của cohort dùng để report.

## 8. R5 — external replication

Chỉ bắt đầu sau R3. Dataset phải có stable user ID, stable item ID, rating,
global timestamp và item text. Rerun data audit; không reuse cutoff/cohort từ
Amazon Books. Method hyperparameters vẫn frozen ở giá trị P7-v2; nếu schema bắt
buộc thay đổi, result được gọi là adaptation study chứ không phải replication.

## 9. Stop rules

- Không đủ temporal/user-disjoint cohort: stop và báo feasibility.
- Graph smoke/leakage fail: stop trước GPU.
- LLM smoke fail: sealed run ID; sửa contract bằng run ID mới rồi smoke lại.
- Primary fail: không tune alpha/restart/depth trên test; kết luận effect chưa
  replicate và chuyển sang error analysis read-only.

## 10. Execution record

| Phase | Status | Evidence |
|---|---|---|
| R0 cohort | Pass | 200 events, 200 users; exclude 1.500 prior-cohort users |
| R1 graph smoke | Pass | 20 events; deterministic replay; labels không dùng khi score |
| R2 LLM smoke | Pass | 20 events, 40/40 generations, 0 retry/repair |
| R2 LLM full | Complete | 400/400 primary generations, 0 retry/repair |
| R3 evaluation | **Pass** | +0,1371 NDCG@5; CI lower +0,0904 |

R0 audit có 10 distinct candidates/event, `gold-in-history = 0`, history dài
5–6 và candidate pool 189.859 item. R1 dùng cùng graph snapshot với
1.008.972 users, 142.572 source item, 511.907 consecutive batch pairs và
876.669 source-to-group links.

GPU run dùng đúng một H100, tensor parallel 1. Smoke peak 7,829 GiB; full
peak 7,863 GiB. Smoke cache được reuse: full invocation chỉ sinh 360 call
còn lại. Sau mỗi invocation, model thoát và VRAM trở về 4 MiB/1 MiB.

## 11. Final result cho report

| Arm | NDCG@5 | Hit@5 |
|---|---:|---:|
| Frozen local Qwen ranker | 0,624675 | 0,820 |
| Transition-PPR residual, alpha 0,80 | **0,761807** | **0,905** |

- Delta NDCG@5: **+0,137133**.
- Paired bootstrap 95% CI: **[+0,090385; +0,183835]**.
- Improved / worsened / unchanged: 64 / 36 / 100 event.
- Replication gate (`delta >= +0,03`, CI lower > 0): **pass**.
- Gold reachability: one-step 36/200; multi-hop PPR 123/200.
- Negative reachability: one-step 8/1.800 (0,44%); PPR 172/1.800
  (9,56%).
- Post-hoc best-of-two oracle: 0,803821 (`+0,179147` vs local), chỉ là
  headroom, không deployable.

Tổng LLM workload là 400 physical request: 341.893 input token + 21.418
output token = 363.311 token. Không có graph LLM request; graph scoring và
bootstrap là CPU-only. Evaluation 200 event sau graph build mất 69,30 giây.

## 12. Artifact integrity

| Artifact | SHA256 |
|---|---|
| Prepared cohort | `6b1726a620313051bdf0ea488d4a87f7c8dc42bf259cb87dea9b0e816e54a41e` |
| Locked method | `4e3d6e98cfed44143b43ce33c0fd6363352f494a353a069d40bd8b8e47d3936a` |
| Local LLM manifest | `77aec295a6745c9259720d98d1baddadd2b3dc5ade91869fa8bc15a2c814f36c` |
| Test scores, 200 rows | `58c550c5dc67dcc6a26d1e0f8cf3695e91b4669120b9de693e801894c7ebe219` |
| Metrics | `956cb632e00b7a1b45ae7307cbe06cd08c95953b3fd9e030cdd9eb6a99cefaf6` |
| Evaluation manifest | `688fb5c1fee96e047c7e3905334f389edbe861c246acc3e829ffb87be7efc3a7` |

## 13. Next step

Primary Amazon result đã đóng; không tune thêm trên 200 labels này.
Tiếp theo là external replication/adaptation study trên dataset khác. MovieLens
32M mới chỉ được đặt trong `data/ml-32m/`; chưa audit, convert hay chạy.
