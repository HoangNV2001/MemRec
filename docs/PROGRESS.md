# PROGRESS.md — Nhật ký milestone

> Append một mục sau mỗi milestone (§10.3 của `docs/RL_PLAN.md`). Không sửa các mục cũ, chỉ thêm mới.

## M0 — Setup (chuẩn bị, chưa chạy pipeline)

Trạng thái: IN PROGRESS

Đã làm:
- Đọc `docs/RL_PLAN.md`, khảo sát repo.
- Phát hiện + sửa bug: `src/train/trainer_memrec.py` hardcode `reranker_provider_name = 'azure_openai'` bất kể `provider.name` trong config — khiến Stage-ReRank luôn cố dùng Azure client dù cấu hình provider khác. Sửa thành dùng chung `provider_name` với Stage-R/W.
- Thêm `python-dotenv`, load `.env` ở đầu `scripts/run_train.py`.
- Đổi `provider.name` trong `configs/memrec_instructrec-books.yaml` và `configs/memrec_instructrec-books_1k.yaml` từ `azure_openai` sang `openai`, trỏ `endpoint`/`api_key` tới `OPENAI_API_BASE`/`OPENAI_API_KEY` (thay vì biến Azure) — khớp với việc dùng key OpenAI thường, không phải Azure.
- Thêm `.gitignore` (chưa từng có trong repo) để tránh commit nhầm `.env`, `data/`, `results/`, `checkpoints/`.
- Thêm `.env.example` làm template.
- Tạo `data/iagent/` (rỗng, chờ tải dataset thủ công).

Việc tiếp theo:
- Người dùng tải `booksAll_recagent.pkl` + `combined_books_asin_mapping.csv` vào `data/iagent/`.
- Người dùng điền `OPENAI_API_KEY` thật vào `.env`.
- Chạy `bash scripts/convert_all_instructrec.sh` (hoặc lệnh convert riêng cho books) rồi verify `data/processed/instructrec-books`.
- Đo warmup 100 user, ngoại suy chi phí, rồi chạy full pipeline 1k user theo DoD của M0.
