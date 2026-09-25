# MemRec — full-system improvement research

Mục tiêu thesis là cải tiến **full MemRec** (collaborative memory, Stage-R,
LLM re-rank và Stage-W) và kiểm chứng tốt hơn full MemRec lẫn SASRec trên
cùng protocol. **Chưa có kết quả xác nhận đạt mục tiêu này.** Các experiment
temporal graph/PPR cũ chỉ dùng local ranker, nay là exploratory archive.

## Tài liệu hiện tại

- [Roadmap full MemRec hiện tại](docs/THESIS_ROADMAP.md)
- [Audit baseline và dữ liệu Books](docs/FULL_MEMREC_BASELINE_AUDIT.md)
- [Bản thảo cũ đã rút lại](docs/THESIS_DRAFT.md)
- [Exploratory transition/PPR đã lưu](docs/TRANSITION_PPR_METHOD.md)

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

Toàn bộ Amazon Books nằm dưới `data/amazon_books/`: raw CSV ở `raw/` và
derived outputs ở `artifacts/`. Cả hai đều gitignored ngoại trừ `.gitkeep`.

## Setup và kiểm tra

```bash
conda create -n memrec python=3.10
conda activate memrec
pip install -r requirements.txt

python -m pytest -q tests/temporal_books
python -m src.temporal_books.p0_audit
python -m src.temporal_books.p7_transition_ppr --prepare
python -m src.temporal_books.p7_transition_ppr --graph-smoke
```

Self-host LLM phải chạy smoke-first qua Slurm theo local runbook
`internal_docs/H100_RESOURCE_RULES.md`; không chạy model trực tiếp trên login
node và phải nhả GPU ngay khi task kết thúc.

## Upstream MemRec

Core memory/model/training modules trong `src/{memory,models,train,data}` và các
config `configs/memrec_*.yaml` là điểm khởi đầu để tái lập full baseline. Dữ
liệu Books có original fixed test candidates trong
`data/processed/instructrec-books/booksAll_recagent.pkl`; bật
`--use_pregenerated_candidates` khi đánh giá Books. Chưa chạy full benchmark
cho đến khi hoàn tất các leakage/resource gates trong roadmap. Paper gốc:

> Chen et al., “MemRec: Collaborative Memory-Augmented Agentic Recommender
> System,” ACL 2026. https://aclanthology.org/2026.acl-long.2061.pdf
