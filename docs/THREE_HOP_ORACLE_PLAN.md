# THREE_HOP_ORACLE_PLAN.md — Bounded 3-hop oracle, self-hosted

> **Ngày khóa protocol:** 2026-09-21
> **Trạng thái:** **complete — hard stop**. V4 grammar-constrained smoke và full
> 100 event đã hoàn tất; 3-hop không qua hai headroom gate đã khóa.
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

V2 với cap 384 xác nhận output không bị truncate: nội dung hoàn chỉnh nhưng model
escape apostrophe thành `\\'` và thêm một closing brace. V3 dùng instruction JSON
rõ hơn và parser repair có audit, chỉ cho phép hai biến đổi syntax không đổi nội
dung: bỏ backslash trước apostrophe, và bỏ closing brace dư sau object. Mọi lỗi
khác vẫn hard-fail. V3 dùng run ID/artifact path mới; không reuse v1/v2 output.

V3 pass 5 packet rồi model trả JSON thiếu trường `memory`; deterministic retry
fail giống nhau. Không mở rộng parser. V4 dùng `lm-format-enforcer` đã có sẵn
trên cụm để constrain decoding theo exact JSON schema. Smoke v4 yêu cầu 0 parser
repair; model/data/prompt semantics/graph/arms/seed/gate vẫn giữ nguyên và v4 có
run ID/artifact path mới.

### V4 execution

- Smoke pass 20 event + 36 packet: **116/116** response hợp lệ, 0 retry, 0
  parser repair; peak VRAM 7,83 GiB.
- Full reuse toàn bộ smoke cache và sinh thêm 844 response: journal cuối
  **960 primary / 0 retry / 960 success**.
- Full invocation dùng 640.442 input + 62.219 output token; peak VRAM 7,85 GiB.
- Process chỉ thấy GPU 0. Sau khi task xong, process thoát và GPU 0 về 1 MiB;
  không giữ model/VRAM idle. GPU thứ hai không được dùng vì runner hiện tại là
  single-writer; chạy thêm process sẽ làm race journal/cache key.

## 6. Metrics và gate khóa trước run

- Primary metric: NDCG@5 trên cùng 100 event.
- Secondary: H@5.
- CI: paired bootstrap 10.000 resample, seed `20260921`.

P3 chỉ được coi là có headroom thực dụng khi **đồng thời**:

1. `oracle_three_hop - local >= +0.05` NDCG@5 và CI lower > 0;
2. `oracle_three_hop - oracle_two_hop >= +0.02` NDCG@5 và CI lower > 0.

Nếu một điều kiện fail: dừng tăng hop cho cơ chế source-packet/item-overlay này;
không post-hoc tăng cap, đổi origin score, đổi prompt/model hoặc chọn cohort.

### Kết quả khóa

| Arm | NDCG@5 | Δ vs local | 95% paired CI | H@5 |
|---|---:|---:|---:|---:|
| `local` | 0,5752 | reference | — | 0,76 |
| `oracle_two_hop` | 0,5893 | +0,0140 | [−0,0007; +0,0338] | 0,79 |
| `oracle_three_hop` | 0,5907 | **+0,0155** | **[−0,0117; +0,0465]** | 0,80 |

So với 2-hop, 3-hop chỉ tăng **+0,0014** NDCG@5, CI
[−0,0188; +0,0208]; cải thiện 4, làm tệ 4 và giữ nguyên 92 event. Cả hai điều
kiện admission đều fail, nên quyết định là **hard stop**.

Breakdown offline giải thích vì sao coverage không chuyển thành ranking gain:

| Gold support | N | Δ 2-hop vs local | Δ 3-hop vs local | Δ 3-hop vs 2-hop |
|---|---:|---:|---:|---:|
| có ở cả 2-hop và 3-hop | 17 | +0,0826 | +0,1161 | +0,0335 |
| chỉ có thêm ở 3-hop | 7 | 0 | **−0,0609** | **−0,0609** |
| không arm nào chạm | 76 | 0 | 0 | 0 |

Như vậy 3-hop tăng support>=2 từ 17 lên 24 gold nhưng bảy endpoint mới trung
bình làm ranking tệ hơn. Một diagnostic post-hoc cực lạc quan, chọn tốt nhất
giữa local/2-hop/3-hop cho từng event bằng label, cũng chỉ đạt +0,0279 NDCG@5
(CI [+0,0073; +0,0558]), vẫn dưới gate +0,05. Diagnostic này không phải kết
quả preregistered và không được dùng như selector.

**Quyết định:** không tăng tiếp 4-hop/n-hop bằng cùng source-packet/item-overlay.
Nếu tiếp tục multi-hop, phải mở protocol mới thay representation/scoring (ví dụ
candidate-level path score hoặc graph retrieval có residual gate), không chỉ
thêm tầng hay chọn giữa ba output hiện có.

## 7. Artifacts

- Config: `configs/temporal_amazon_books_2014/p3_selfhost.yaml`
- Structural v4: `p3v4_structural_smoke-hnv.json`,
  `p3v4_item_route_ledger-hnv.jsonl`, `p3v4_preflight_manifest-hnv.json`
- Prepared v4: `p3v4_selfhost_prepared-hnv.json`
- LLM v4: `p3v4_selfhost_{attempts,calls,smoke_manifest,metrics,manifest}-hnv.*`

Mọi artifact ở `data/temporal_amazon_books_2014/` là derived/gitignored; manifest
ghi SHA256 và kết quả được kéo về local ngay sau run.

V4 integrity:

- completion manifest SHA256: `dee6ecc05cf7f97d2f2989f799911e0df5d96fac11aeb9353048f7d7b78e4edd`;
- metrics SHA256: `3ab1d1a6f754473bb24d8190347bff9f023c4a49a16d3c908bc9c50a40fd2784`;
- attempts/calls SHA256: `979d79071a2009bc7f6cbdb850a089c57d4190f80c41a3be968221c1b5f02ff8` /
  `1bc217a28c87e9b1c554c0f19c079c2b2486a9d97dba2fd33897bf6a1dc6bb2a`.
