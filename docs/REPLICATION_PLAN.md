# Replication plan — Temporal Transition-PPR

> Trạng thái: protocol draft cho run kế tiếp; chưa materialize cohort, chưa mở
> label và chưa dùng GPU. P7-v2 test tuyệt đối không được dùng để tune thêm.

## 1. Mục tiêu

Replication phải trả lời hai câu hỏi tách biệt:

1. Gain +0,0752 có lặp lại trên một cohort user-disjoint mới của cùng Amazon
   Books hay không?
2. Sau khi internal replication pass, transition semantics có transfer sang
   một temporal book dataset khác hay không?

Primary replication trước mắt là câu 1. External dataset là phase tiếp theo,
không trộn result hai phase.

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
