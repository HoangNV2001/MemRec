# CANDIDATE_GRAPH_5L_PLAN.md — Candidate-directed 3/5-layer graph probe

> **Protocol khóa:** 2026-09-21, trước smoke/full result.
> **Trạng thái:** **complete — hard stop** cho cả P5 sampled và P5b full graph.

## Câu hỏi

Packet overlay P3/P4 có ít headroom. P5 kiểm tra một representation khác: dùng
multi-hop chỉ để tính numeric graph evidence riêng cho từng candidate, rồi fuse
residually với frozen local ranking. Không sinh memory text, không gọi LLM/GPU.

Trong naming này, `item_layers=3` là path
`source→anchor→peer→bridge→peer→candidate` (5 graph edges). `item_layers=5`
thêm hai collaborative item layers nữa.

## Candidate-directed search

- Cùng 100 event × 10 candidate frozen từ P3; scorer xử lý mọi candidate như
  nhau và không đọc gold.
- Graph gồm sampled user interactions rating>=4, timestamp strict `< target`.
- Search ngược từ candidate về tối đa sáu recent positive source anchors.
- Mỗi step giữ tối đa 8 peer/item, global beam 64; simple path không lặp node.
- Recency half-life 730 ngày; length decay 0,70; hub penalty weight 0,50.
- Aggregate tối đa tám path có `(anchor, candidate-side peer)` khác nhau.
- So max 3 và max 5 item-layers dưới cùng caps/score.

Graph score được min-max theo event rồi fuse cố định:

```text
final(candidate) = frozen_local_rank_score(candidate) + 0.25 * graph_score(candidate)
```

Không tune alpha/path cap/decay sau result.

## Smoke, metrics và gate

Smoke deterministic 20 event trước full 100; smoke không dùng gold để scoring.
Primary metric NDCG@5, secondary H@5; paired bootstrap 10.000, seed `20260921`.

P5 pass chỉ khi đồng thời:

1. residual 5-layer − local >= +0,02 và CI lower >0;
2. residual 5-layer − residual 3-layer >= +0,01 và CI lower >0.

Nếu fail, không train learned candidate graph scorer trên cohort này. Mọi output
là derived/gitignored; manifest phải hash raw reviews, frozen P3 inputs, smoke,
scores và metrics. LLM requests = 0, GPU = none.

## P5 sampled-graph result

P5 dùng graph sample `blake2b % 48 == 0` kế thừa packet pilot:

- layer-3 / layer-5 có evidence cho 2 / 5 gold trên 100 event;
- negative evidence rate là 0 / 0,11%; graph rất precise nhưng quá sparse;
- residual layer-5 = 0,5765 NDCG@5, chỉ +0,0013 vs local;
- residual layer-5 và layer-3 giống hệt ranking, delta 0;
- gate: **hard stop**. Runtime 0,65 giây sau load; LLM/GPU = 0.

Artifact SHA256: smoke
`ef946ba8a1e92a58e11bcdc4c3eaed0fd1b1ad8d08d9f29f981a9dbab6a8698f`,
scores `01e4254e58cbeceff412ded37faa125e7c14359b9444855ccb3b28470e9a5e96`,
metrics `6821da6348bfdd4df6f87c3f73b5704dc0537bda24d72c95a085ef6259324416`.

## P5b full-graph extension

Sampled result cho thấy blocker là graph density, không phải path noise. Vì
candidate scoring không tiêu packet/LLM budget, P5b thay đúng một yếu tố đã khóa
trước run: user hash modulus `48 → 1`, tức dùng toàn bộ positive interaction
graph trước validation cutoff. Scoring, beam/caps, 20-event smoke, alpha,
candidate cohort, metrics và gate giữ nguyên P5. P5b dùng run ID/artifact path
riêng; không overwrite hoặc trộn P5 sampled output.

## P5b full-graph result

Smoke 20 event pass: layer-3/5 evidence ở 4/28 trên 200 candidate slots, so
với 1/1 của sampled graph. Full 100 event:

| Arm | NDCG@5 | H@5 | Δ vs local |
|---|---:|---:|---:|
| frozen local | 0,5752 | 0,76 | reference |
| graph-only layer-3 | 0,3776 | 0,54 | −0,1976 |
| graph-only layer-5 | 0,4513 | 0,63 | −0,1239 |
| residual layer-3 | 0,5848 | 0,78 | +0,0096 |
| residual layer-5 | 0,5829 | 0,77 | **+0,0077** |

Residual layer-5 vs local CI [−0,0100; +0,0266]. So với residual layer-3,
layer-5 = **−0,0019**, CI [−0,0137; +0,0086]. Cả hai gate fail.

Full graph chứng minh density giải quyết reachability nhưng làm giảm precision:

- layer-3: evidence cho 15% gold và 2,33% negative slots;
- layer-5: evidence cho 30% gold nhưng 13,78% negative slots;
- evidence-slot precision xấp xỉ 41,7% ở layer-3 và chỉ 19,5% ở layer-5;
- post-hoc best-of local/residual-3/residual-5 cũng chỉ +0,0188 vs local,
  dưới gate +0,02. Diagnostic này không phải deployable gate.

P5b dùng 751.982 user và 196.942 positive item; runtime scoring 7,51 giây sau
load, LLM requests=0, GPU=none. Integrity SHA256: smoke
`0caf6d9a3f60e7bad4038b9c4fd7a77fd4f731eaecac334f76e4ce7049c73be2`,
scores `b22e9168403f28c6fe7e151553241d8ef54a5ea705a0b0c679d79da66c9cac1b`,
metrics `928c12300868ac662355373fe2d83bfcbdf426f2416cabf0293689661da84168`,
manifest `3785c5998140e8ed1b0675078263230f5396b769f71e4f0b2245bec319774c18`.

**Quyết định:** không tăng depth hoặc train scorer từ chính cohort này. Full
graph tốt hơn sampled graph về recall, nhưng fixed 5-layer evidence tăng negative
nhanh hơn gold và kém residual 3-layer. Muốn tiếp tục cần supervised training
cohort độc lập lớn hơn và depth-adaptive gate; đó là protocol mới, không phải
tune P5/P5b sau result.
