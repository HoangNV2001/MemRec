# MemRec — nghiên cứu cải tiến toàn hệ thống

Mục tiêu luận văn là cải tiến **full MemRec** (collaborative memory, Stage-R,
LLM ReRank và Stage-W), rồi kiểm chứng bằng đối chứng full MemRec và SASRec
trên cùng tập user/candidate. **Chưa có kết quả xác nhận đạt mục tiêu này.**
Temporal Transition/PPR cho local ranker là kết quả thăm dò, không phải method
thesis đã được xác nhận.

Hướng hiện tại là **CM-IRank**: giữ Stage-R/Stage-W, thay Stage-ReRank bằng
chính sách loại dần candidate, fine-tune **Qwen/Qwen3.5-4B bằng PPO** với MPSS.

## Trạng thái hiện hành — 2026-10-09

| Phần việc | Bằng chứng / giới hạn |
|---|---|
| Benchmark đối chứng | Full MemRec NDCG@5 **0,747918**, SASRec **0,343320**, cùng 200 dev users/candidates; MemRec warm-up 700 users, không phải full-data/paper replication |
| Data/protocol CM-IRank | Khóa 1.497 train / 300 validation; snapshot target-blind và 3.594 candidate sets v2 đã audit CPU, không có cờ trong 7 probe cố định; chưa chứng minh không còn shortcut |
| Policy infrastructure | RankRequest/replay, N−1, parser, MPSS có CPU tests; Qwen3.5 synthetic smoke 20/20 và một optimizer update bỏ đi đã pass; **chưa train PPO** |
| Real memory | V6 fail đã review; **v7 secondary control + independent review pass**, 20/20 user, 100 calls, 20 warm-up writes / 0 pseudo writes; không chứng minh semantic grounding hay thay primary |
| Method result | **Chưa có checkpoint/score CM-IRank**; held-out 5.377 users vẫn niêm phong |

Kế hoạch và đánh giá khả thi mới nhất: [design §0.14](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#014-overall-review-and-next-gates--2026-10-07).
CPU review v6 đã đối chiếu 35 request/schema và replay đúng 11 lần Stage-W;
không sửa ID, bỏ user, tune prompt hay nới hard gate để thông qua smoke.
Researcher đã duyệt **v7 constrained-decoding secondary control**: schema chỉ cho
phép ID từ input, không thay upstream-aligned primary; cần compile schema CPU
rồi smoke thật 20 user/100 calls và review riêng ([§0.15](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#015-approved-secondary-input-id-decoding-control--2026-10-07)). Sau memory gate mới tới real N−1/direct smoke,
one-card PPO compatibility và tiny PPO; không tự nối sang full training.
**V7 terminal:** 16,55 phút / 240.750 tokens / 0 cache hits, peak observed VRAM
48 GiB; còn 67 candidate-context citations phải giữ cảnh báo, không chứng minh
semantic grounding. 26/26 artifacts hash-match; GPU về 1 MiB trước handback.
`omni-gen-1` đã chạy lại; từ nay GPU task xong phải unload rồi bật lại riêng
keeper theo runbook, kể cả khi còn CPU review. GPU0/job reserved được giữ nguyên.
Independent review đã xác nhận exact schemas/domains, 20 raw writes và 20 pseudo
inputs read-only, không có unknown ID. Bước hiện tại: **N−1/direct functional
smoke của Qwen3.5-4B**, 20 frozen inputs secondary / tối đa 200 generations,
CPU token audit đã pass **40 prompts / 709–1.044 tokens**, 319 tests pass;
**V2 GPU0 terminal:** 86,51 giây /177 generations, N−1 hợp lệ **14/20**, direct
**0/20** vì thiếu answer tags (không truncation/repair). GPU đã unload rồi
keeper0 chạy lại; GPU1/job được bảo vệ. 20/20 artifacts hash-match và CPU replay
khớp. Lỗi format dẫn tới đề xuất **format-only SFT trước PPO**
([§0.18](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#018-real-qwen35-n1direct-smoke-outcome-and-format-decision--2026-10-09)).
Researcher đã cho tiếp tục: **format-only SFT**256 synthetic examples,64 updates
đã khóa; trước đó smoke20 examples/5 updates và save/reload. Giữ no-SFT controls,
không nhãn Books/tune ranking;340 tests pass. CPU data/mask/token/hash gate đã pass
256 examples +20 holdout /80 prompt audit. Đã có xác nhận keeper0; v1 GPU preflight
dừng trước reclaim vì GPU1 đổi sang workload TTS. V2 sửa guard để GPU1 luôn
read-only, giữ nguyên SFT recipe; tiến độ và receipts ở design §0.19–0.20.
**SFT smoke PASS:**5 updates/134,44s; synthetic N−1/direct20/20; save/reload20
logit probes delta0. GPU đã unload và keeper bật lại. Sau review và xác nhận mới,
**Full SFT đã xong:**64 updates/11,31 phút; N−1/direct đều20/20 trên synthetic
và cùng20 Books inputs; save/reload delta0,24 artifacts hash-match. GPU đã unload,
keeper chạy lại. Đây chỉ là **format initializer**, chưa có gain NDCG/Hit.
Bước tiếp: PPO actor/critic/reference compatibility trong env riêng. Researcher
đã đồng ý chọn/re-pin CUDA12-compatible (design §0.23); profile CPU đầu tiên là
VeRL0.9 /Torch2.11+cu129 /vLLM0.20+cu129 /Transformers5.10.1 /TRL0.25.1.
**CPU gate đã pass**:237 dependencies/pip-check/20 miniature probes,11,11 phút,
12 artifacts hash-match. Đây vẫn là **candidate runtime**, chưa phải PPO4B đã
xác nhận. **FA2 build/import đã pass** sau67,82 phút CPU resume,6 artifacts
khớp SHA. Đang chuẩn bị smoke kernel/full4B actor/reference/native critic20
mẫu trên một GPU; **không optimizer update hay ranking metric** (§0.24).
PPO worker/rollout/optimizer memory và canonical primary memory vẫn còn gate.
Không đổi driver/env baseline, không tự dùng GRPO/LoRA/2GPU.
Chưa có score/checkpoint PPO CM-IRank hoặc quyết định thay primary.

Migration và cleanup nguồn cũ đã hoàn tất, CPU smoke sau cleanup pass. Mọi compute
mới theo root/account/allocation hiện hành trong runbook nội bộ; SSH dùng chung
ControlMaster, không hủy job reserved và không giữ model idle. Chi tiết lịch sử
lỗi hạ tầng được giữ ở design §0.5–0.13, không nhầm với tiến độ nghiên cứu.

## Bản đồ tài liệu — chỉ 5 tài liệu dự án

| Tài liệu | Nguồn chuẩn cho |
|---|---|
| README này | Điểm vào, cấu trúc repo và setup |
| [CM-IRank design](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md) | Kế hoạch nghiên cứu/triển khai mới, chuẩn học thuật, G0–G3 và ledger tiến độ |
| [BOOKS_BENCHMARK](docs/BOOKS_BENCHMARK.md) | Dữ liệu/protocol InstructRec Books, full-agent baseline, SASRec, Stage-R trial, hashes và tái lập |
| [EXPLORATORY_HISTORY](docs/EXPLORATORY_HISTORY.md) | Phương pháp/kết quả Amazon–MovieLens, các hướng dừng, Laya, Goodreads và audit trail |
| [H100_RESOURCE_RULES](internal_docs/H100_RESOURCE_RULES.md) | Quy tắc nội bộ về Slurm/GPU; không đưa nội dung hạ tầng này vào tài liệu công khai |

Roadmap cũ đã được hợp nhất vào CM-IRank design; các hướng đã đóng nằm trong
lịch sử thăm dò. Artifacts, per-user predictions, configs và source code vẫn là bằng
chứng gốc; tài liệu chỉ là bản diễn giải. Tệp metadata đi kèm dataset, manifest
dependency và cache sinh tự động không phải tài liệu nghiên cứu của repo.

## Exploratory experiment surface (không phải thesis benchmark)

```text
configs/temporal_amazon_books_2014/
├── dataset_audit.yaml
├── p7v2_transition_ppr.yaml
└── transition_ppr_replication_200.yaml

src/temporal_books/
├── common.py
├── current_support.py
├── p0_audit.py
├── p7_selfhost_local.py
└── p7_transition_ppr.py
```

Amazon Books Kaggle nằm ở `data/amazon_books/` (`raw/` và `artifacts/`),
InstructRec Books ở `data/processed/instructrec-books/`, MovieLens 32M ở
`data/ml-32m/`, và Goodreads raw ở `data/goodreads/`. Không gộp metric giữa
các nguồn này: task, candidate construction và mức độ tin cậy thời gian khác
nhau. Raw/derived data được quản lý theo `.gitignore` hiện hành.

## Setup và kiểm tra

```bash
conda create -n memrec python=3.10
conda activate memrec
pip install -r requirements.txt

python -m pytest -q tests/temporal_books
python -m pytest -q tests/test_cmirank_g0.py
python scripts/cmirank/02_preview_policy_split.py --seed cmirank-books-v1-draft-20260930 --min-prefix-length 5 --verify-locked configs/cmirank/policy_split_manifest.json
python scripts/cmirank/03_audit_prefix_snapshot.py --query-users 20
python scripts/cmirank/04_smoke_pseudo_memory_cpu.py --users 20
python -m src.temporal_books.p0_audit
python -m src.temporal_books.p7_transition_ppr --prepare
python -m src.temporal_books.p7_transition_ppr --graph-smoke
```

Self-host LLM phải chạy smoke-first qua Slurm theo
[runbook nội bộ](internal_docs/H100_RESOURCE_RULES.md); không chạy model trên
login node và phải nhả GPU ngay khi task kết thúc. Lệnh ở trên chỉ là smoke
cho pipeline graph thăm dò, **không** tái lập kết quả full MemRec. Lệnh Books,
config, model revision và các cổng kiểm tra nằm trong
[Books benchmark](docs/BOOKS_BENCHMARK.md).

## Upstream MemRec

Core memory/model/training modules trong `src/{memory,models,train,data}` và các
config `configs/memrec_*.yaml` là điểm khởi đầu để tái lập full baseline. Dữ
liệu Books có original fixed test candidates trong
`data/processed/instructrec-books/booksAll_recagent.pkl`; bật
`--use_pregenerated_candidates` khi đánh giá Books. Hiện chỉ có baseline
full-architecture **700 warm-up/200 dev user**; chưa có full 7.377-user hay
held-out result, và chưa có method đạt improvement gate. Paper gốc:

> Chen et al., “MemRec: Collaborative Memory-Augmented Agentic Recommender
> System,” ACL 2026. https://aclanthology.org/2026.acl-long.2061.pdf
