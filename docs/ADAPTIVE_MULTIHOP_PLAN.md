# ADAPTIVE_MULTIHOP_PLAN.md — P6 learned depth-adaptive graph scoring

> **Protocol khóa trước khi chạy:** 2026-09-21.
> **Trạng thái:** **complete — hard stop** trên fresh test.

## Mục tiêu và tính hợp lệ

P5 cho thấy full graph tăng reachability nhưng fixed 5-layer tăng negative
evidence nhanh hơn gold. P6 vẫn giữ multi-hop, nhưng học trọng số riêng cho
evidence nông và evidence chỉ xuất hiện ở depth sâu, sau đó chỉ fuse khi margin
đủ lớn.

100 event P3–P5 đã được quan sát nhiều lần nên được hạ vai trò thành
**validation/calibration**. Primary result P6 phải đo trên 100 test event mới,
ở timestamp sau validation cutoff. Không báo gain trên cohort cũ như test gain.

## Cohort

- Graph: rating >=4; mọi path chỉ đọc interaction có timestamp strict `< target`.
- Candidate pool: item đã có positive interaction trước global train cutoff.
- Train scorer: 1.200 user/event trong 365 ngày cuối trước train cutoff; target
  positive, novel với user, user có ít nhất 5 positive event quá khứ.
- Test: 100 user/event sau validation cutoff, target/candidate rule giống train;
  user không trùng train hoặc validation cohort cũ.
- Mỗi event có gold + 9 deterministic uniform negative chưa thấy; candidate
  order được deterministic shuffle. Label chỉ dùng để fit/evaluate, không đi
  vào path feature, Stage-R hay rerank prompt.

## Scorer và calibration

Mỗi candidate có feature tách `<=3 item-layers` và incremental `4–5 layers`:
presence, log path score, best path, số diverse path, independent anchor và
candidate-side peer. Pairwise ridge logistic học từ `(gold − negative)` trên
train cohort bằng Newton solver cố định, không cần GPU/LLM.

Validation cũ chỉ được chọn một cặp trong grid đã khóa:

- fusion alpha: `[0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]` (alpha 0 cho
  phép calibration giữ nguyên local thay vì ép một graph model tệ);
- model-score margin gate: `[0, 0.10, 0.25, 0.50, 1.00]`.

Arm deployable duy nhất dùng cấu hình validation NDCG@5 cao nhất; tie-break lần
lượt là margin cao hơn rồi alpha thấp hơn. Nếu margin dưới gate, giữ nguyên
local rank. Sau khi ghi/hash locked model, không đổi grid/feature/gate theo test.

## Fresh-test local baseline

Test local ranking dùng đúng Qwen3-4B-Instruct-2507 revision đã pin ở P3-v4,
Stage-R + listwise rerank, greedy structured decoding. Smoke 20 event bắt buộc
và được reuse trong full 100. Tổng primary generation 200, retry reserve 20;
đúng một H100 nhìn thấy process, memory fraction 0,25, process phải thoát và trả
VRAM ngay sau mỗi task.

## Smoke, metric và gate

1. Unit tests local.
2. Graph smoke: 24 train + 20 test event, cùng candidate/path/feature code với
   full; kiểm finite feature, strict-past và gold không tham gia scoring.
3. LLM smoke 20 test event trên cluster, 100% schema/ranking hợp lệ, zero repair,
   sau đó mới full.
4. Primary metric fresh-test paired NDCG@5; secondary H@5; bootstrap 10.000.

P6 pass khi adaptive residual − fresh local >= +0,02 và paired CI lower >0.
Nếu fail: hard stop cho learned depth weighting dưới representation hiện tại.
Oracle/post-hoc trên fresh test chỉ được tính sau primary và phải ghi rõ là
diagnostic, không được đổi decision.

## PROGRESS & RESULTS

Protocol ở trên được khóa khi chưa có P6 result. Sau đó:

- Unit/regression suite: 19 passed.
- Sửa temporal guard trước full fit: interaction cùng timestamp được xử lý như
  một batch và không được tính là history của nhau.
- Materialize 1.200 train + 100 fresh-test event; test prompt đều có 5–6 review
  strict-past. Prepared SHA256:
  `bdda8d846376ae62d4d47e0747f7bdd2805a8c2d78d5f7e57dc98054ee6b8dd4`.
- Graph smoke 24 train + 20 test pass: 440 candidate slot, evidence ở 16 slot
  layer-3 và 76 slot layer-5; label không được dùng để score. Smoke SHA256:
  `b93da194eedcb002315e08ed32e621bff72ce1e941dc1d20cb4c97b55e89dfe0`.
- Fit 10.800 pair trong 97,95 giây CPU-only. Calibration chọn alpha `0,6`,
  margin gate `0,1`, active 44/100; NDCG@5 validation 0,6018 so với local
  0,5752 (+0,0265). Đây chỉ là selection result, không phải primary test.
  Locked-model SHA256:
  `1e7a13e48bdee9f9753b54ccaffdf12c51874d31c87b627664180e729aa3cd7f`.
- LLM smoke 20 event pass; full có 200/200 primary success, zero retry/error/
  repair. Qwen revision giữ nguyên, peak 7,84 GiB trên đúng một H100; process
  thoát và cả bốn card về 1 MiB. Local manifest SHA256:
  `9916675d13f02c76e4bfcfb6900d3dc69838feb72b1b0845a74f45a0e53cc5bb`.

Fresh-test primary result, chỉ mở label sau khi model/alpha/gate đã khóa:

| Arm | NDCG@5 | H@5 | Δ vs local |
|---|---:|---:|---:|
| fresh local | 0,5897 | 0,79 | reference |
| adaptive multi-hop | 0,5672 | 0,77 | **−0,0225** |

Paired CI của delta là [−0,0559; +0,0113]; adaptive cải thiện 8, làm tệ 22 và
giữ nguyên 70 event. Gate active 57/100, cao hơn validation 44/100. Layer-3/5
cover 9/18 gold; negative evidence rate 2,56%/13,78%.

Post-hoc oracle chọn local/adaptive tốt hơn theo từng event đạt 0,6202,
+0,0305 vs local. Đây là diagnostic không deployable: nó cho thấy signal vẫn
có, nhưng margin gate không nhận diện được khi nào nên can thiệp. Không retune
alpha/gate trên fresh test. Metrics SHA256:
`91389ca081e24a6fcb981d74f1add16f630801f81de37e22be3799169a305816`;
manifest SHA256:
`2dccc686dd5eeb09657076e660fef91c49c80f68db0816a8a96eefd357515619`.

**Quyết định:** hard stop cho pairwise depth weighting + scalar margin gate
hiện tại. Nếu vẫn tiếp tục multi-hop, protocol mới phải học trực tiếp
intervention risk (`graph giúp local` hay `làm hại local`) trên cohort lớn hơn
và đánh giá trên một fresh cohort khác; không được dùng 100 P6 test event để
tune rồi báo lại như test.
