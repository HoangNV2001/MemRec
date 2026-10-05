# MemRec — nghiên cứu cải tiến toàn hệ thống

Mục tiêu luận văn là cải tiến **full MemRec** (collaborative memory, Stage-R,
LLM ReRank và Stage-W), rồi kiểm chứng bằng đối chứng full MemRec và SASRec
trên cùng tập user/candidate. **Chưa có kết quả xác nhận đạt mục tiêu này.**
Temporal Transition/PPR cho local ranker là kết quả thăm dò, không phải method
thesis đã được xác nhận.

Hướng đang triển khai là **CM-IRank**: giữ memory Stage-R/Stage-W, nghiên cứu
chính sách Stage-ReRank loại dần candidate và MPSS reward. Chính sách chính đã
chốt là **Qwen/Qwen3.5-4B fine-tune bằng PPO**; G0 có interface, reward/parser
tests, split policy khóa, snapshot graph và Stage-W wiring được smoke bằng CPU.
**Stage-W pseudo-memory với LLM thật chưa được chứng minh sạch, chưa fine-tune,
chưa có score method**. Smoke Qwen3.5-4B ngày 2026-10-01 đã tải/load checkpoint
nhưng dừng do lỗi serialize log trước inference/backward; GPU đã nhả sạch.
Lỗi log đã được sửa (129 tests pass). Smoke v2 load/backward/optimizer chạy được,
nhưng chỉ 4/20 output đúng schema; GPU đã nhả sạch. Chưa train PPO hoặc đo ranking
Books; ledger và bước tiếp theo nằm trong CM-IRank design §0.5.
V3 với schema rõ đã **pass 20/20 synthetic output và backward/optimizer**;
peak VRAM 34,66 GiB, khoảng 40 giây, GPU đã trả về 1 MiB. 141 tests CPU pass.
Đây chỉ là infrastructure smoke; G0/PPO và ranking Books chưa được chứng minh.

Cập nhật **2026-10-05**: candidate preparation đã hoàn tất 1.797 policy users /
3.594 bộ; **195 tests CPU pass**. Audit tiếp theo phát hiện shortcut semantic do
hard negatives chọn quanh positive: probe chỉ dùng candidate-set đạt NDCG@5
0,6633 trên pseudo-validation, **không phải score CM-IRank**. PPO trên v1 bị chặn;
researcher đã cho triển khai v2 đổi riêng semantic anchor sang allowed history
prefix, giữ 3+3+3 và tiêu chí audit. V2 phải smoke/audit CPU lại trước GPU.
Kết quả, giới hạn và bước tiếp theo ở design **§0.9–0.10**; chưa nạp GPU mới.
V2 đã hoàn tất candidate/feature smoke và full audit CPU: **1.797 users / 3.594 bộ,
không có cờ trong 7 probe cố định**; semantic-centrality probe NDCG@5 còn 0,3110.
Đã kiểm hash toàn bộ 26 tệp kết quả. Đây không phải score method hay PPO-ready.
Bước tiếp theo là **20-user real LM_Mem/LM_Rec smoke** (§0.11), giữ nguyên baseline
và tách target khỏi policy input. Smoke v1 dừng trước inference do vLLM 0.10.2
không nhận GPU UUID trong binding; v2 dừng ở kiểm UUID do khác tiền tố giữa
PyTorch/NVML. V3 đã inference nhưng dừng sau 2 request: Stage-R cite một candidate
đã thấy trong prompt như collaborative neighbor; chưa có Stage-W/user hoàn tất.
GPU đã nhả sạch sau cả ba lượt. V4 theo amendment §0.12 giữ nguyên upstream output,
ghi candidate-visible thành cảnh báo vai trò; ID bịa/lỗi schema/label vẫn hard-fail.
**237 tests CPU pass**; chưa có memory smoke hoàn tất, fine-tune hay score CM-IRank.
Review CPU v3 đã xong; v4 sẵn sàng nhưng **chưa chạy** vì preflight mới thấy cả
4 GPU đang có process/giữ VRAM. MemRec không giữ card hay server để chờ.

**Cập nhật resource:** user yêu cầu đổi account/reserved allocation và chuyển toàn
bộ server workspace sang `/mnt/data/users/hoangnv242/memrec-hnv`. Migration phải
copy/hash/rebase/smoke trước khi xóa nguồn cũ; hiện chưa được coi là hoàn tất.
Quyền mới chỉ dùng GPU được duyệt, không bao giờ hủy job reserved; runbook nội bộ
và design §0.13 là nguồn chuẩn, không dùng account/path cũ cho compute mới.

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
