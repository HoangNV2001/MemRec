# PROGRESS.md — Nhật ký milestone

> Append một mục sau mỗi milestone (§10.3 của `docs/RL_PLAN.md`). Không sửa các mục cũ, chỉ thêm mới.

## M0 — Reproduce baseline — 2026-08-06

Trạng thái: DONE (một phần — xem "Lệch so với kế hoạch")

Đã làm:
- Đọc `docs/RL_PLAN.md`, khảo sát repo.
- Setup: conda env `memrec` (Python 3.10, torch 2.9.0+cpu), `.gitignore`, `.env.example`, `python-dotenv` load trong `scripts/run_train.py`.
- Đổi `provider.name` trong config books sang `openai` (key OpenAI thường, không phải Azure). Người dùng tự tải `booksAll_recagent.pkl` + `combined_books_asin_mapping.csv` vào `data/iagent/`, tự tạo `.env`, tự chạy convert + toàn bộ pipeline 1k user lần đầu (trước khi tôi kiểm tra lại kết quả).
- Đo warmup 100 user: 930,926 token tổng cho 100 user (xem `results/m0_warmup_100users.log`).
- Chạy MemRec full pipeline + 2 baseline nội bộ (`w/o Collab. Read`, `Vanilla LLM`) trên 1000 user, seed 42.
- **Phát hiện DoD fail lần đầu:** thứ tự sai trên mọi metric (`w/o Collab. Read` < `Vanilla` ở cả H@1/H@3/N@3/H@5/N@5) — người dùng đã tự bắt được 1 bug write-leak trước đó (`results/m0_vanilla_1k_leaky_writes_DO_NOT_USE`), tôi tìm và sửa thêm 3 bug nữa (chi tiết đầy đủ ở `docs/RESULTS.md` mục "Các bug đã tìm thấy và sửa"): Stage-ReRank hardcode `azure_openai`; `vanilla_mode` là dead code; prompt "MemRec mode" tự mâu thuẫn khi facets rỗng; thiếu `enable_stage_w: false` cho eval loop của `no_collab_read`.
- Rerun `no_collab_read` + `vanilla` full 1000 user sau fix (tình cờ chạy full thay vì smoke 5-user, do gotcha `eval_user_list` override `--n_eval_users` — xem `docs/RESULTS.md`).

Số đo: H@1 sau fix đúng thứ tự (0.510 > 0.436 > 0.425). H@3/N@3/H@5/N@5 **vẫn sai thứ tự**.

Lệch so với kế hoạch:
- DoD của M0 (RL_PLAN §7) yêu cầu **toàn bộ** H@{1,3,5}/N@{3,5} đúng thứ tự VÀ reproduce 2 lần ra cùng số. Cả hai điều kiện này **chưa đạt đầy đủ**.
- Nguyên nhân nghi ngờ: `_evaluate_single_user` sample negative candidates bằng `RandomState(seed=hash(thread.ident))` — không theo `user_id`, không reproducible, 3 config không dùng chung candidate set → so sánh giữa 3 config bị nhiễu.
- **Quyết định đã ra (người dùng chủ động chọn), 2026-08-06:** không sửa RNG này ngay, chấp nhận phần sai lệch H@3/H@5/NDCG là nhiễu sampling, dùng H@1 làm baseline đủ tin cậy để đi tiếp M1. Đã ghi rõ deviation này vào `docs/RL_PLAN.md` (mục M0, đánh dấu `☑~`) theo đúng quy tắc §10.2/§10.8 của chính plan.

Việc tiếp theo:
- Bắt đầu M1 (đóng băng môi trường & dựng RL dataset), tái sử dụng `llm_conversations/` từ các run M0 full-1k đã có.
- Nếu sau này cần bảng H@3/H@5/NDCG đáng tin cậy hơn (M7a/M7b viết luận văn), phải quay lại sửa RNG seed theo `user_id` trước, rồi rerun cả 3 config lại từ đầu.
