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

## M1 — Đóng băng môi trường & dựng RL dataset — 2026-08-06

Trạng thái: DONE (DoD đạt đầy đủ)

Đã làm:
- Branch `rl/m1-env`. Toàn bộ code mới nằm dưới `src/rl/`, `tests/rl/`, `scripts/rl/`, `configs/rl/` — không sửa một dòng nào của đường eval gốc (§10.4).
- `src/rl/splits.py` — split user-disjoint + sinh candidate tất định theo `(seed, user_id, salt)`, **không** theo thread id. Đây cũng là bản vá cho lớp bug #5 của M0, nhưng chỉ áp dụng cho đường RL mới.
- `src/rl/policy.py` — prompt Stage-R candidate-blind + parser JSON không bao giờ raise (cứu được cả output bị cắt giữa chừng, salvage facet ở mọi độ sâu lồng nhau).
- `src/rl/env.py` — `GraphSnapshot`, cache `N'_k(u)`, dựng neighbor table bằng chính packer của repo (không viết lại) để prompt RL trùng byte-với-byte prompt Stage-R gốc trừ khối candidate.
- `src/rl/warmup.py` + `src/rl/build_snapshot.py` — dựng memory sạch-test; `src/rl/build_dataset.py` + `src/rl/dataset.py` — sinh và nạp jsonl; `src/rl/leakage.py` — màn lọc rò đáp án.
- Warmup 2350 user (train 1200 + val 150 + test 1000 ghim theo eval sample 1k của M0), 24 worker, 26.6 phút, 100% CPU.

Số đo:
- Warmup: 2347/2350 user thành công. 9.54M input + 2.87M output token ≈ **$3.15**. **0 GPU-hour.**
- Snapshot 17.4 MB; jsonl: train 1185 / val 149 / test 993; prompt ~1020 token (median).
- `pytest tests/rl/` → **78 pass**, 26 giây, không cần API/GPU (thoả §10.5 smoke test <5 phút).

Lệch so với kế hoạch:
1. **Không tái sử dụng dump M0, mà warmup lại từ đầu (+$3.15).** Bắt buộc: `results/m0_memrec_full_1k/memory.jsonl` là trạng thái **sau** vòng eval, mà config MemRec chính để `enable_stage_w` mặc định `True` → ground-truth click trên **test item** đã được ghi vào memory dùng chung. Dùng nó làm graph đóng băng là rò đáp án vào mọi rollout. Warmup mới nhắm `history[-2]` (valid item) nên sạch test. Ngoài ra M0 chỉ warm 1000 user, M1 cần 2350. Người dùng đã chọn phương án warmup thêm; tôi warm lại cả 2350 trong một lần thay vì replay log M0 cho 1000 + warm 1350, vì replay 5808 dòng log có thứ tự ghi đua nhau giữa 16 thread nên chỉ là xấp xỉ. Chênh ~$1.6 để đổi lấy một lệnh tất định. Có `--memory_file` để tái tạo snapshot offline miễn phí nên tiền warmup chỉ tiêu một lần.
2. **Split 1185/149/993 thay vì 1200/150/1000**, do loại 23 user bị lộ tên sách đáp án (xem dưới). Không bù thêm user vì tốn thêm tiền warmup cho ~1%.
3. **State không chứa instruction InstructRec.** §3 viết `s = (I_u, M_u, Rep(N'_k(u)))`, nhưng instruction của InstructRec diễn giải lại chính quyển sách đáp án, và `MemRecManager.build_stage_r_prompt` gốc cũng **không** nhận instruction — nó chỉ đi vào Stage-ReRank. Hiểu `I_u` là biểu diễn lịch sử tương tác (đã có trong neighbor table). Instruction vẫn được lưu ở trường riêng cho frozen ranker dùng ở M2.

Quyết định đã ra + lý do:
- **Loại 23 user bị rò tên sách (0.98%).** Catalogue Books có nhiều `item_id` cho cùng một quyển sách; nếu user có bản sao kia trong lịch sử thì tên sách đáp án hiện trong neighbor table. Toàn bộ 23 ca đều qua neighbor table, 0 ca qua `M_u`. DoD yêu cầu grep ra 0 kết quả nên phải loại thật, không thể ghi chú rồi bỏ qua.
- **Ghim ngân sách neighbor của packer.** `SnippetPacker` trừ 300 token cho khối candidate; bỏ candidate mà không bù thì policy được 1000 token neighbor còn baseline prompted chỉ 700 → "GRPO thắng prompted" sẽ lẫn với "được nhìn nhiều neighbor hơn". Đã ghim `CANDIDATE_BLOCK_RESERVE`.
- **Tách RNG của warmup và eval bằng salt.** Trước đó cả hai rút đúng 9 distractor giống nhau, mà Stage-R lúc warmup có nhìn khối candidate → distractor của bài thi góp phần nặn ra `M_u`. Sửa miễn phí vì candidate eval sinh offline.
- **Track `data/rl/user_splits_books.json` trong git** (ngoại lệ của `.gitignore`): file nhỏ nhưng **định nghĩa** thí nghiệm; thiếu nó thì không tái lập được phân hoạch.

Ghi chú cấu trúc pipeline (quan trọng cho M2):
- `LLMRulePruner.prune()` bỏ qua tham số `candidates` → `N'_k(u)` vốn đã candidate-blind, cache được vô hại.
- `SnippetPacker.build_neighbor_snippet()` dựng bảng neighbor từ metadata tĩnh của item, **không** từ `M_v`. Nên input duy nhất phụ thuộc memory của Stage-R là `M_u`. Memory của neighbor chỉ vào pipeline qua `item_mems` của Stage-ReRank — snapshot vẫn giữ item memory cho reward ranker ở M2.

Việc tiếp theo:
- M2 Phần A (CPU, không GPU): `reward/metrics.py`, `reward/grounding.py`, `reward/composite.py`, `reward/ranker.py` ở chế độ stub, `tests/rl/test_reward_logic.py`.
- Chuẩn bị sẵn điểm NDCG@5 của gpt-4o-mini trên 149 user val, cache ra file **trước** khi thuê GPU (§11.6) để phiên T2 chỉ còn việc so sánh.
- `r_null` + `baseline_h1` vẫn là `null` trong cả 3 jsonl; backfill bằng `src.rl.dataset.backfill` sau khi reward function chạy được.

## Review `docs/RL_LM_REC_EXTENSION.md` — 2026-08-06

Trạng thái: REVIEWED, đã hiệu chỉnh cả file extension lẫn Plan gốc. **Không thực thi gì** — extension bị chặn sau cổng M7a + M5 Ưu tiên 1.

Đã làm:
- Đối chiếu toàn bộ file extension với hiện trạng code và dữ liệu sau M0/M1. Sửa file extension trước (theo yêu cầu), rồi mới chỉnh Plan gốc cho khớp.
- Kiểm chứng bằng số 3 giả định mà file extension dựa vào (script ở scratchpad, kết quả ghi trong file extension §0.1).

Số đo dùng để kết luận:
- Instruction InstructRec có shingle 3-từ của tên sách đích: **19/879 test user (2.2%)** khớp nguyên văn; phần còn lại diễn giải nội dung. → yêu cầu "không có gold title trong prompt" của E0 vừa bất khả thi (gold là candidate) vừa sai trọng tâm.
- Vị trí gold trong candidate list: phân bố **đều** trên 0–9 (103/93/83/91/106/103/98/101/110/105) nhưng **cố định theo user** qua mọi epoch → kênh hack "nhớ vị trí" là thật.
- `candidate_memories`: **10/10 candidate ở cả 993/993 record** đều có item memory → sơ đồ luồng dữ liệu của extension bỏ sót là lỗi thật sự, sẽ làm dòng A không còn là baseline MemRec.

Hai lỗi thiết kế đã bịt (nếu để nguyên thì kết quả extension không dùng được):
1. **Ma trận 4 dòng bị confound.** `D > B` đổi đồng thời model ranker (gpt-4o-mini → Qwen3-4B) và thêm GRPO. Đã thêm 2 dòng control A′/B′ (SFT-4B, chưa GRPO) để tách `hiệu ứng GRPO = D − B′` khỏi `hiệu ứng đổi model = B′ − B`. Phát biểu thành công đổi từ `D > B` sang `D > B′`. Chi phí thêm ~0 vì checkpoint SFT đã có từ E1.
2. **Thiếu xáo thứ tự candidate.** Đã thêm §4.2.1 (xáo theo `(user_id, seed, step)`, chấm reward theo `item_id`), rủi ro §9.4bis, và "test đảo thứ tự" vào DoD của E2.

Ba chỗ khác phải sửa: teacher SFT không tái dùng được conversation M0 (sai bộ candidate + nhiễm GT + sai format); bỏ `fixed_candidates_books.jsonl` để không có nguồn sự thật thứ hai; DoD E0 đòi hash khớp 100% sau re-run là bất khả thi vì batching của vLLM.

Quyết định đã ra + lý do:
- **Extension xếp sau M5 Ưu tiên 1.** M5 Ưu tiên 1 chỉ ~2 GPU-hour và bảo vệ trực tiếp đóng góp chính; extension mở một đóng góp phụ. Nếu ranker-swap cho thấy gain không giữ được thì mọi nhánh sau đều vô nghĩa.
- **Extension và M5 đầy đủ loại trừ nhau về ngân sách** (24–36h vs 33h, trong khi sau Phase 1 chỉ còn ~40–70h). Đã viết §7.1 của Plan gốc thành cổng chọn-một-nhánh, mặc định là M5. Không quyết bây giờ — quyết sau M7a khi biết số giờ còn lại thật.
- **Cổng yêu cầu ≥36h** thay vì 20h như bản đầu, bằng cận trên của chính §10 file extension: extension dở dang không dùng được vào luận văn.
- **Giữ nguyên §2 "Ngoài phạm vi: train LLM_Rec"** của Plan gốc, chỉ thêm ghi chú. Trong toàn bộ M0–M7b `LLM_Rec` vẫn đóng băng tuyệt đối; extension là nhánh tách biệt có branch/file kết quả riêng.

Việc tiếp theo: không đổi — M2 Phần A (CPU). Extension chỉ được đụng tới sau M7a.

## M2 Phần A — Reward function trên CPU — 2026-08-06

Trạng thái: DONE (Phần A). Phần B chờ GPU.

Đã làm:
- Branch `rl/m2-reward`. `src/rl/reward/{metrics,ranker,grounding,composite}.py` + `src/rl/validate_reward.py` + `src/rl/build_val_reference.py` + `scripts/rl/02_validate_reward.sh`.
- `tests/rl/test_reward_logic.py` (46 test) + `tests/rl/test_validate_reward.py` (Spearman đối chiếu khớp scipy tới 1e-9). Tổng **140 test pass**, 26 giây, không API/GPU.
- Cache sẵn nửa gpt-4o-mini của Validation A/B (§11.6): 149 val user × 5 arm = 745 cặp đã chấm bằng chính `LLMReranker` của repo. 5.7 phút, ~$0.5, 0 GPU-hour.
- Chạy harness end-to-end trên CPU bằng stub ranker: 745 cặp, throughput 24 516 reward/s. Chứng minh đường ống chạy trước khi thuê máy.
- Thêm trường `neighbor_snippets` vào 3 file jsonl (rebuild miễn phí, số dòng không đổi).

Số đo:
- **Memory có tác dụng thật:** `M_collab` thật cho +0.1112 NDCG@5 so với không memory trên `LLM_Rec` thật, 95% CI [+0.0640, +0.1585], n=149, 45 user tốt hơn / 91 bằng / 13 tệ hơn.
- **Memory sai bị bỏ qua, không gây nhiễu:** `shuffled` −0.0002 (CI [−0.037, +0.037]), `lorem` −0.0013 (CI [−0.025, +0.023]) — cả hai ≈ `empty`.
- **Tỉ lệ trùng reward:** hai `M_collab` lấy mẫu độc lập cho cùng user cho cùng vị trí gold ở **111/149 = 74%**. Trùng reward: NDCG@5 **80.5%**, NDCG@10 74.5%, MRR 74.5%.
- Facet sinh ra: trung bình 6.99/7, không mẫu nào rỗng trên 298 mẫu.

Lệch so với kế hoạch:
1. **Tiêu chí Validation B phải hiệu chỉnh.** DoD §7 M2 viết `r(thật) > r(user khác) > r(lorem) ≈ r(rỗng)`. Bất đẳng thức giữa **sai trên chính `LLM_Rec` thật** — `shuffled` 0.6090 ≈ `lorem` 0.6079 ≈ `empty` 0.6092. Model thật *bỏ qua* memory không liên quan chứ không bị đánh lừa; đòi proxy tái hiện `shuffled > lorem` là đòi proxy dễ bị lừa hơn model nó thay thế. Đổi thành `r(thật) ≥ max(các arm hỏng) + 0.02`. Đã sửa cả trong plan lẫn code, theo §10.8.
2. **Thêm `soft_weight` vào reward, mặc định tắt.** Xem quyết định dưới.

Quyết định đã ra + lý do:
- **Reward chỉ theo rank có trần trùng giá trị 74%** — hệ quả trực tiếp: trong group GRPO thì `std(r)=0` → advantage 0 → không gradient (§9.2), và dynamic sampling §6.4 sẽ lọc vượt xa ngưỡng báo động 60% của kill criteria M4. Nếu không phát hiện trước, nhiều khả năng mất vài phiên H100 để thấy đường reward phẳng. Đã thêm số hạng liên tục `soft_weight * p_gold` (`p_gold` = xác suất softmax ranker đặt lên gold), **mặc định `soft_weight = 0.0` tức đúng công thức §5**. Phần B đo tỉ lệ trùng trên ranker 1.5B thật rồi mới quyết định bật. Không tự ý đổi spec reward — đây là cùng lập luận §5.1 đã dùng để loại Hit@1, chỉ là NDCG@5 vẫn chưa đủ mịn.
- **Grounding so khớp với snippet neighbor, không phải `M_v` trong storage.** Snippet là thứ policy thực sự đọc (`SnippetPacker` dựng từ metadata tĩnh). So với memory trong storage là chấm policy trên văn bản nó chưa từng thấy, và sẽ hỏng hẳn với neighbor user không nằm trong 2350 user của snapshot (không có `M_u`). Thêm `parse_neighbor_snippets` + trường `neighbor_snippets`.
- **Mẫu số của grounding là `n_facets` yêu cầu, không phải số facet sinh ra.** Chia cho số sinh ra là lỗ hổng hiển nhiên: sinh 1 facet hoàn hảo, bỏ 6 cái còn lại, được điểm 1.0. §5.2 viết `/ N_f` và nghĩa là số mục tiêu. Có test khoá lại.
- **Output méo vẫn được chấm như "không có memory", không phải hằng số.** Hằng số sẽ khiến mọi rollout méo giống hệt nhau → group `std=0` → §9.2 quay lại qua ngả format penalty.
- **`include_instruction` giữ `True`.** Phần A đo được instruction *không* làm phẳng tín hiệu memory (+0.111 vẫn còn khi đã có instruction), nên giữ proxy trung thành với `LLM_Rec` thật. Vẫn chạy `--no_instruction` ở Phần B để đối chứng.

Việc tiếp theo:
- M2 Phần B trên GPU (gộp phiên với M3-B theo §11.5): `bash scripts/rl/02_validate_reward.sh hf`. Không còn gọi API — mọi thứ chấm lại trên cặp đã cache.
- Nếu ρ < 0.6: theo plan thử `Qwen2.5-3B-Instruct` hoặc pointwise scoring trước khi đi tiếp M4.
