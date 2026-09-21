# THREE_HOP_ORACLE_PLAN.md — Bounded 3-hop oracle, self-hosted

> **Ngày khóa protocol:** 2026-09-21
> **Trạng thái:** structural smoke/preflight pass. Self-host smoke v1 dừng ở
> packet đầu vì JSON malformed/truncated; v2 chưa chạy, full vẫn bị block.
> **Model:** `Qwen/Qwen3-4B-Instruct-2507` revision
> `cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, greedy, một H100.

## 1. Câu hỏi

Sau khi target-aware 2-hop oracle trên Azure chỉ đạt `+0.0324` NDCG@5, liệu
mở thêm đúng một tầng graph có tạo **headroom bổ sung** đủ lớn không?

Không so trực tiếp score cũ từ Azure với self-host model. P3 rerun đồng thời ba
arm bằng cùng checkpoint/Stage-R/candidate order:

1. `local`: base item memory;
2. `oracle_two_hop`: support>=2 từ ledger P1, tối đa hai packet lên gold;
3. `oracle_three_hop`: support>=2 từ ledger P3, tối đa hai packet lên gold.

Cả hai oracle đều nhìn gold sau khi ledger candidate-blind đã đóng, nên là upper
bound không deploy được. Chúng không được trình bày như selector/router.

## 2. Định nghĩa 3-hop và causal guard

```text
source user
  → anchor item
  → peer-1 user
  → bridge item
  → peer-2 user
  → endpoint item
```

Packet phát tại `t`; mọi edge graph phải có timestamp `< t`. Same-day edge bị
loại. Ledger được build/serialize trước khi đọc validation target/candidate.

Route caps khóa trước coverage/result:

| Cap | Giá trị |
|---|---:|
| anchors/source | giữ nguyên tối đa 3 từ P1 packet |
| peer-1/anchor | 16 |
| bridge item/peer-1 | 16 |
| peer-2/bridge | 8 |
| endpoint item/peer-2 | 8 |
| witness path lưu/item | 8 |

Origins của 3-hop được xếp theo số path source→endpoint giảm dần, sau đó user ID;
oracle lấy tối đa hai packet đầu. Không chọn origin bằng NDCG hoặc output model.

## 3. Structural smoke và headroom

Smoke 24 source chạy trước full, không mở label:

- 1.625 endpoint;
- 21.648 path;
- 0,054 giây sau khi load graph;
- không graph explosion, `labels_opened=false`.

Full source-only ledger 560 packet có 5.773 endpoint. Chỉ sau serialize mới đọc
100 fixed validation target:

| Independent source support | 2-hop | 3-hop |
|---|---:|---:|
| >=1 | 24% | 29% |
| >=2 | 17% | **24%** |
| >=3 | 12% | **24%** |

3-hop thêm 7 gold target ở support>=2. Coverage này đủ để admit LLM oracle, nhưng
không tự chứng minh ranking tốt hơn.

## 4. Self-host contract và tài nguyên

- Không dùng OpenAI/Azure key hoặc output P2-v2.
- Transformers offline trực tiếp; không external API server.
- Exact model revision như đầu file; BF16; `do_sample=false`; seed `20260921`.
- `CUDA_VISIBLE_DEVICES` phải chứa đúng một GPU; `tensor_parallel_size=1`.
- PyTorch memory fraction hard-cap `0.25`; smoke phải đo peak `<=20 GiB`, ưu
  tiên `<18 GiB`.
- Chỉ chạy trong reserved Slurm job bằng `--overlap`; full bị block nếu thiếu
  smoke manifest/hash/cache.
- Chi tiết vận hành nội bộ ở `internal_docs/H100_RESOURCE_RULES.md` (gitignored).

## 5. LLM smoke-first và budget

Deterministic smoke chọn 20 event. Nó gồm 20 base source packet cộng mọi source
thực sự được hai oracle đọc, tổng 36 packet:

| Pha smoke | Generation |
|---|---:|
| Semantic packets | 36 |
| Shared Stage-R | 20 |
| Rerank, 3 arm | 60 |
| **Tổng** | **116** |

Smoke pass khi: 100% JSON/schema/ranking hợp lệ; process thấy một GPU; peak VRAM
<=20 GiB; không OOM/NaN; config/prepared/model/sample key có hash. Smoke outputs
dùng cùng cache key trong full run.

Full budget:

| Pha | Primary generation |
|---|---:|
| Semantic packets | 560 |
| Shared Stage-R | 100 |
| Rerank, 3 arm | 300 |
| **Primary** | **960** |
| Identical retry reserve | 40 |
| **Hard cap** | **1.000** |

### Smoke v1 incident

Run `p3_oracle_selfhost_qwen3_4b_2507_hnv` load đúng model/GPU nhưng packet đầu
trả cùng `JSONDecodeError` ở hai identical attempts dưới output cap 220. Runner
dừng ngay; không có Stage-R/rerank/full output. Journal v1 được giữ nguyên và
không reuse. V2 chỉ tăng packet/Stage-R output cap lên 384, dùng run ID/artifact
path mới; model, data, graph caps, prompts, arms, seed và gate không đổi.

## 6. Metrics và gate khóa trước run

- Primary metric: NDCG@5 trên cùng 100 event.
- Secondary: H@5.
- CI: paired bootstrap 10.000 resample, seed `20260921`.

P3 chỉ được coi là có headroom thực dụng khi **đồng thời**:

1. `oracle_three_hop - local >= +0.05` NDCG@5 và CI lower > 0;
2. `oracle_three_hop - oracle_two_hop >= +0.02` NDCG@5 và CI lower > 0.

Nếu một điều kiện fail: dừng tăng hop cho cơ chế source-packet/item-overlay này;
không post-hoc tăng cap, đổi origin score, đổi prompt/model hoặc chọn cohort.

## 7. Artifacts

- Config: `configs/temporal_amazon_books_2014/p3_selfhost.yaml`
- Structural v2: `p3v2_structural_smoke-hnv.json`,
  `p3v2_item_route_ledger-hnv.jsonl`, `p3v2_preflight_manifest-hnv.json`
- Prepared v2: `p3v2_selfhost_prepared-hnv.json`
- LLM v2: `p3v2_selfhost_{attempts,calls,smoke_manifest,metrics,manifest}-hnv.*`

Mọi artifact ở `data/temporal_amazon_books_2014/` là derived/gitignored; manifest
ghi SHA256 và kết quả được kéo về local ngay sau run.
