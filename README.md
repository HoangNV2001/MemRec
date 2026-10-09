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
khớp. Cần researcher chốt có thêm **format-only SFT trước PPO** không
([§0.18](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#018-real-qwen35-n1direct-smoke-outcome-and-format-decision--2026-10-09)).
Chưa có score/checkpoint CM-IRank, PPO pass hoặc quyết định thay primary.

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
