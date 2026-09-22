# MemRec — Temporal Transition-PPR extension

Repo này giữ implementation gốc của MemRec và hướng nghiên cứu hiện tại:
directed temporal item-transition graph + multi-hop PPR cho next-item ranking.
P7-v2 đã tăng NDCG@5 từ `0,6533` lên `0,7285` trên fresh Amazon Books cohort
(`+0,0752`, paired 95% CI `[+0,0179; +0,1375]`).

## Tài liệu hiện tại

- [Phương pháp, protocol và kết quả](docs/TRANSITION_PPR_METHOD.md)
- [Replication plan](docs/REPLICATION_PLAN.md)
- [Các hướng đã đóng](docs/ARCHIVED_DIRECTIONS.md)

## Active experiment surface

```text
configs/temporal_amazon_books_2014/
├── dataset_audit.yaml
└── p7v2_transition_ppr.yaml

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
config `configs/memrec_*.yaml` được giữ để đối chiếu baseline. Paper gốc:

> Chen et al., “MemRec: Collaborative Memory-Augmented Agentic Recommender
> System,” ACL 2026. https://aclanthology.org/2026.acl-long.2061.pdf
