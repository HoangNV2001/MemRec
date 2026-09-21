# FILTERED_THREE_HOP_PLAN.md — Positive, temporal, diversity-aware 3-hop

> **Protocol khóa:** 2026-09-21, trước LLM smoke/result.
> **Trạng thái:** structural smoke/full pass; LLM smoke/full chưa chạy.

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
