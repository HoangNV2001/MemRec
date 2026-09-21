# FILTERED_THREE_HOP_PLAN.md — Positive, temporal, diversity-aware 3-hop

> **Protocol khóa:** 2026-09-21, trước LLM smoke/result.
> **Trạng thái:** **complete — hard stop**. Structural và LLM smoke/full đều
> hoàn tất; filtered 3-hop không qua hai ranking gate.

## Câu hỏi

P3 raw 3-hop tăng support>=2 từ 17 lên 24 gold nhưng bảy gold mới có mean
delta −0,0609. P4 kiểm tra liệu lọc đường đi theo quality có đổi precision đủ
lớn không; đây là protocol mới, không sửa P3 hậu nghiệm.

## Bộ lọc candidate-blind

Mọi route vẫn strict `< packet_time`:

```text
source → anchor → peer1 → bridge → peer2 → endpoint
```

- Chỉ giữ interaction có `review/score >= 4.0` trên cả path.
- Recency score dùng half-life 730 ngày, tính từ năm edge của path.
- Hub penalty tại packet time:
  `1/sqrt(log2(2+degree(anchor))*log2(2+degree(bridge)))`.
- Mỗi origin chỉ cộng tối đa ba path có bridge khác nhau; origin xếp theo tổng
  quality score, rồi user ID. Không dùng candidate/gold/LLM output để lọc.
- Giữ nguyên caps raw P3: 16 peer1/anchor, 16 bridge/peer1, 8 peer2/bridge,
  8 endpoint/peer2; overlay tối đa hai packet lên labelled gold.

Ledger phải serialize trước khi mở 100 validation target. Structural admission:
support>=2 gold coverage tối thiểu 12%. Nếu fail thì không chạy LLM.

Kết quả structural đã khóa:

- smoke 24 source: 800 endpoint, 20.389 positive path, 0,11 giây sau load,
  `labels_opened=false`;
- full 560 source: 3.879 endpoint; gold coverage support>=1/2/3 =
  22%/19%/18%, nên pass gate 12%;
- 19 filtered support>=2 event là subset của raw 3-hop; filter loại 5 raw event;
  top-2 origin overlap trên phần chung chỉ 33%, nên routing thực sự thay đổi;
- ledger SHA256:
  `058f79c55f4f29f404a50b60a848f9370ed4b8bc0c053d1571886804dcb58078`;
  preflight SHA256:
  `d2af8355c09e27c40c2098e8deea5c0e95f84f0a19ec80332b8ee254b1772a16`.

## Reuse và request budget

P4 pin SHA256 của P3 prepared/calls/manifest. Nó reuse đúng 560 packet, 100
Stage-R, local ranking và raw 3-hop ranking đã sinh bằng cùng exact model. P4
chỉ tạo filtered rerank mới:

| Pha | Physical generation mới |
|---|---:|
| deterministic smoke | 20, được reuse ở full |
| full còn lại | 80 |
| retry reserve | 20 |
| hard cap | 120 |

Model: `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, greedy, JSON-schema
constrained. Full bị block nếu smoke 20 event chưa pass 100%, có repair, mismatch
hash/cache hoặc peak VRAM >20 GiB theo contract P3/P4 hiện tại.

## Metrics và gate

100 fixed event; NDCG@5 primary, H@5 secondary; paired bootstrap 10.000,
seed `20260921`.

Admission chỉ pass khi đồng thời:

1. filtered 3-hop − local >= +0,05 và CI lower >0;
2. filtered 3-hop − raw 3-hop >= +0,02 và CI lower >0.

Fail một điều kiện là hard stop cho bộ lọc này; không đổi rating threshold,
half-life, hub formula, diversity cap hoặc cohort sau khi thấy result.

## Kết quả

| Arm | NDCG@5 | Δ vs local | 95% paired CI | H@5 |
|---|---:|---:|---:|---:|
| local frozen P3 | 0,5752 | reference | — | 0,76 |
| raw 3-hop frozen P3 | 0,5907 | +0,0155 | [−0,0117; +0,0465] | 0,80 |
| filtered 3-hop | 0,5939 | **+0,0186** | **[−0,0021; +0,0425]** | 0,80 |

Filtered vs raw 3-hop = **+0,0031**, CI [−0,0160; +0,0238]; cải thiện 5,
làm tệ 4 và giữ nguyên 91 event. Vì vậy fail cả gate +0,05 vs local và +0,02
vs raw.

Breakdown offline:

- trên 19 event có filtered support>=2: raw vs local +0,0814; filtered vs
  local +0,0980; filtered vs raw +0,0165;
- 5 event chỉ raw cover và 76 event không arm nào cover đều không đổi ranking;
- post-hoc best-of local/raw/filtered đạt +0,0370 vs local, CI
  [+0,0127; +0,0669], vẫn dưới point gate +0,05. Đây không phải selector hợp lệ.

Execution: smoke 20/20, full journal 100/100 primary success, 0 retry, 0
repair. Full invocation mới dùng 80 request, 81.964 token và peak 7,85 GiB trên
một H100. Process thoát ngay sau task; GPU về 1 MiB.

Integrity:

- completion manifest SHA256:
  `2321936b677ea46e729fa4b38336a8a359692b0d49d9868a37c8e0ed0993631d`;
- metrics SHA256:
  `4ec4305354b186f94e0d92c74ba4e01407a33ac847ccb9ec691c2ea1ef79a910`;
- attempts/calls SHA256:
  `dfbe0796430733b4b0cc31526aacd8b051eb748b42d872db31fab3f0b730e6c0` /
  `e530f437ec6ca8fcdec91f633d0e2c8df585a4b55eae154491aefdeaa5fe2fe7`.

**Quyết định:** bộ lọc path quality giúp đúng hướng nhưng magnitude quá nhỏ và
không ổn định. Không tune threshold/half-life post-hoc. Nếu tiếp tục, cần đổi
sang candidate-level graph scoring/retrieval thay vì tiếp tục overlay packet.
