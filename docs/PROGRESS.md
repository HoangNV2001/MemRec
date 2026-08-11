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

## M2 Phần B — Reward validation trên GPU — 2026-08-07

Trạng thái: DONE MỘT PHẦN. Validation A + B đạt (sát nút, và chỉ nhờ `soft_weight`); Validation C hoãn sang H100. Chi tiết số liệu đầy đủ ở `docs/RESULTS.md` mục "M2 Reward Validation → Phần B".

Máy: NVIDIA L4 24 GB (đúng tầng T1 mà `docs/DEPLOY_GPU.md` §1.6 khuyến nghị — **không** thuê H100 cho việc này). ~1.5 GPU-hour. **0 lời gọi API** — chấm lại đúng 745 cặp đã cache ở Phần A.

Đã làm:
- Verify môi trường trước khi chạy: torch 2.6.0+cu124 / CUDA available / L4, `verify_transfer` all pass, `pytest tests/rl/` 139 pass + 5 skip. Bật thêm 4 test `test_ranker_hf.py` (vốn skip vì thiếu `HF_TEST_MODEL`) bằng chính model thật → 4 pass.
- Chạy Validation A/B/C cho 1.5B (cả `instruction` on/off), rồi 3B, rồi 3B fp32.
- Thêm `--dump_pairs` + metric **within-user agreement** vào `validate_reward.py`; thêm `--dtype` cho `FrozenRanker`; viết `src/rl/backfill_baselines.py`.

Số đo chính:
- **1.5B hỏng thật, không phải sát nút:** ρ = 0.3071 (ngưỡng 0.6), và `lorem` (0.4174) **thắng** memory thật (0.4091). Khoảng cách `sample1 − empty` = +0.006 so với +0.111 trên `LLM_Rec` thật.
- **Trần không phải vấn đề:** tự-tương quan của chính gpt-4o-mini là ρ = 0.899 → ngưỡng 0.6 hợp lý.
- **3B (fp32):** ρ = 0.5861 với NDCG@5 đơn thuần (✗), **0.6051** với `NDCG@5 + 0.3·p_gold` (✓). Validation B đạt với margin **+0.0038**. Throughput 1.3 reward/s @ batch 32 trên L4.
- **Tỉ lệ trùng 71.1%** (NDCG@5) vs **0.0%** (`p_gold`).

Lệch so với kế hoạch:
1. **Đổi frozen ranker 1.5B → `Qwen2.5-3B-Instruct`.** Đây đúng là phương án dự phòng §M2 đã viết sẵn ("Nếu ρ < 0.6: thử `Qwen2.5-3B-Instruct`"). Ảnh hưởng ngân sách VRAM §4.3: ranker 8 GB → ~12 GB fp32, tổng ~69–71 GB, vẫn vừa H100 80 GB nhưng chật hơn. **7B đã loại**: +11 GB nữa thì không còn chỗ cho policy + vLLM colocate.
2. **`soft_weight` 0.0 → 0.3.** Quy tắc "trùng > 50% thì bật, khởi điểm 0.3" của chính §M2 đã kích hoạt (71.1%). Test cũ khoá mặc định = 0.0 đã được đổi thành khoá mặc định = 0.3, kèm một test riêng xác nhận `soft_weight=0.0` vẫn tái hiện đúng công thức §5 cho ablation.
3. **`FrozenRanker` mặc định fp32 thay vì bf16.** Trong bf16 reward **không tất định**: padding đổi thứ tự cộng dồn nên cùng một rollout cho điểm khác nhau tuỳ batch (2/48 user lệch NDCG@5 giữa batch 24 và batch 1; fp32 lệch 0/48). §5.1 chọn thiết kế này *vì* nó tất định, nên bf16 làm sai chính tiền đề. Giá: VRAM ×2, throughput ~½.
4. **Validation C hoãn.** 3B fp32 @ batch 64 không vừa L4 24 GB, nên con số "≥ 20 reward/s @ batch 64" **không đo được trên tầng T1**. Phải đo lại trên H100 ở phiên M3/M4. Không coi là đạt.
5. **Thêm trường `baseline_p_gold`.** `baseline_h1` nhị phân (ranker đóng băng, tất định → chỉ 0.0 hoặc 1.0) khiến dải curriculum `[0.2, 0.8]` của §6.4 khớp **0 user** và làm rỗng tập train. `baseline_p_gold` liên tục diễn đạt đúng ý định "bỏ user quá dễ và quá khó". Ghi cả ba trường; chọn trường nào lái curriculum để lại cho M4.

Quyết định đã ra + lý do:
- **Giữ `include_instruction=True`** — đã đóng câu hỏi bỏ ngỏ của §5.1 bằng số: on 0.3071 vs off 0.1411 trên 1.5B.
- **Không sửa prompt ranker.** Nghi ngờ "đọc logit ở đầu lượt assistant nên model muốn viết chữ thay vì một ký tự" đã được đo và **bác bỏ**: letter mass = 0.9998. Thêm prefix "Answer: " còn phá nó (mass → 0). Loại trừ được một nghi can trước khi đổ lỗi cho model.
- **Đo thêm within-user agreement** (không có trong DoD gốc). ρ gộp trên 745 cặp bị chi phối bởi khác biệt *giữa* các user — thứ GRPO không bao giờ thấy, vì mọi rollout trong một group là cùng một user. Đây là số quyết định M4 có học được gì không, nên phải đo.

⚠️ **Rủi ro lớn nhất phát hiện được (chưa giải quyết):**
Reward phân biệt **thô** tốt (`sample1` vs `empty`: đồng ý 72.4%) nhưng phân biệt **tinh** thì không (`sample1` vs `sample2`: 37.5%, ngẫu nhiên = 50%). Cùng một cơ chế đo, nên tương phản là thật; stub ranker cho đúng 52% nên metric được hiệu chuẩn đúng. Hai cách đọc — proxy quá thô, **hoặc** chính gpt-4o-mini cũng không phân biệt nổi hai memory tốt (nó trùng điểm 80.5%) — chưa tách được với chỉ 2 mẫu/user. Cả hai đều dẫn tới: M4 dạy được "viết memory thật, bám neighbor", khó dạy được "memory A hơn memory B", gain sẽ bão hoà sớm.

Hai bug đã bắt được lúc backfill, cả hai đều im lặng:
1. Bản đầu `backfill_baselines.py` giữ default `--ranker_model` = 1.5B nên lần chạy đầu ghi đè 3 file jsonl bằng số của ranker **chưa validate**. Lộ ra vì `r_null` (0.4109) lệch arm `empty` của Validation (0.5123) — hai đường code phải cho cùng một số.
2. Lần chạy lại (batch 32) ghi xong `train` + `val` rồi **OOM ở `test`**, để lại 1 split mang số cũ của 1.5B mà nhìn vào dữ liệu không thấy được. Chạy lại `test` riêng ở batch 8.

Đã bỏ default trùng lặp (thừa kế từ `FrozenRanker`), in cấu hình ranker mỗi lần chạy, và thêm `data/rl/baselines_provenance.json` ghi model/dtype/số bản ghi cho từng split. Hash trong `verify_transfer.py` đã cập nhật (nội dung jsonl đổi thật, số dòng không đổi).

Số backfill cuối (3B fp32): `r_null` train 0.5068 / val 0.5123 / test 0.5267. **Dải curriculum §6.4 chỉ giữ ~12% user** (141/1185 train) — không phải lỗi, nhưng M4 cần biết trước.

Việc tiếp theo (cần người dùng quyết trước khi thuê H100):
- **Đề xuất chạy trước, rẻ, không cần GPU (~$1.5, CPU + API):** sinh thêm 3 mẫu `M_collab`/user rồi chấm bằng gpt-4o-mini → số cặp so sánh trong-user tăng từ 1 lên 10 mỗi user. Đây là cách duy nhất tách "proxy quá thô" khỏi "không có tín hiệu tinh nào để học", và nó quyết định M4 có đáng thuê H100 hay không.
- Đo lại Validation C trên H100 (batch 64, fp32) ngay đầu phiên M3/M4.
- Cân nhắc lại ngân sách VRAM §4.3 với ranker 3B fp32 (~12 GB thay vì 8 GB).

## M2 Phần B (tiếp) — thí nghiệm within-user, và ĐẢO NGƯỢC hai kết luận — 2026-08-07

Trạng thái: **M2 KHÔNG ĐẠT DoD. M4 bị chặn** theo đúng luật §M2. Chi tiết số liệu: `docs/RESULTS.md` mục "M2 Reward Validation → Phần B".

Chi phí: **~$0.35 API + ~0.5 GPU-hour** (L4). Không thuê H100.

Đã làm:
- `src/rl/extend_val_reference.py`: sinh thêm 3 mẫu `M_collab`/user (tổng 5), chấm bằng chính `LLMReranker` của repo. Tái dùng nguyên `_generate_m_collab` + `_score` của `build_val_reference` để prompt trùng byte — mẫu mới không so được với mẫu cũ nếu prompt lệch.
- Tổng quát hoá `validate_reward.py`: đọc mọi arm `sampleN`, so **mọi cặp** thay vì chỉ sample1-vs-sample2, thêm khoảng tin cậy Wilson.

Vì sao phải làm: kết luận "không có phân biệt trong-user" của lần đo trước dựa trên **29 cặp**, CI ~[19%, 59%] — không phải một kết luận. Giờ có 296 cặp phân biệt được.

### Đảo ngược #1 — `soft_weight` phải TẮT (đã từng bật 0.3)

Quy tắc §M2 "trùng > 50% thì bật" đã kích hoạt (70.8%) và em đã bật. Đo lại đàng hoàng thì **quy tắc đó nhìn sai số**:

| Ai quyết định | Cặp | Đồng ý | 95% CI |
|---|---:|---:|---|
| NDCG@5 tự quyết | 99 | 60.6% | [50.8, 69.7] trên ngẫu nhiên |
| NDCG@5 hoà → `p_gold` | 197 | 40.1% | [33.5, 47.1] **dưới ngẫu nhiên** |
| Tổng | 296 | 47.0% | = ngẫu nhiên |

`p_gold` được giao đúng những cặp NDCG không phán được, và trên đúng những cặp đó nó phản tín hiệu. Không tinh chỉnh được bằng `w` (mọi `w > 0` cho kết quả y hệt). Đã trả `soft_weight` về 0.0, test khoá lại kèm lý do.

### Đảo ngược #2 — Validation A thật ra FAIL (đã từng báo "đạt 0.6051")

Con số 0.6051 chỉ có được khi (a) bật `soft_weight` và (b) chỉ có 2/5 arm là memory thật. Thêm 3 arm mẫu thật → ρ = **0.5573** (§5 nguyên bản), tối đa 0.5833 với mọi `w`. ρ gộp nhạy với tỉ lệ arm thật/hỏng; bản 5 mẫu đáng tin hơn. **Validation A fail ở mọi cấu hình.**

### Kết cục xấu nhất đã bị loại trừ

Tín hiệu tinh **có thật và lớn**: 296/1490 cặp phân biệt được, biên độ trung bình **0.3413** — gấp ~3 lần hiệu ứng thô (+0.1112) — tập trung ở 54/149 user (36.2%). Nên vấn đề là **proxy 3B không đọc được tín hiệu**, không phải "không có gì để học". Ý tưởng đồ án không bị bác bỏ.

Ghi chú: heuristic tự động ban đầu của em in ra "LLM_Rec mostly CANNOT tell these memories apart" chỉ vì 63.8% user phẳng — đó là kết luận sai, vì cái quyết định gradient là **biên độ của các cặp không hoà**, không phải tỉ lệ hoà. Đã sửa hàm báo cáo.

Việc tiếp theo — **cần người dùng quyết**, 4 hướng đã liệt kê ở cuối mục M2 trong `docs/RESULTS.md`:
pointwise scoring (~2–3 GPU-h, là phương án 2 §M2 đã ghi sẵn) · ranker 7B (chẩn đoán, không dùng được ở M4 vì VRAM) · reward = gpt-4o-mini trực tiếp (~$17–30) · thu hẹp phát biểu đồ án về trục cost của Figure 4.

## M2 Phần B (tiếp) — pointwise scoring: đã thử, KHÔNG cứu được — 2026-08-07

Trạng thái: **M2 vẫn KHÔNG ĐẠT DoD. M4 vẫn bị chặn.** Phương án dự phòng cuối cùng mà §M2 ghi sẵn ("hoặc đổi sang pointwise scoring") đã được thực hiện và loại. Số liệu đầy đủ: `docs/RESULTS.md` mục "Pointwise scoring (§M2 phương án 2)".

Chi phí: ~1.2 GPU-hour trên L4 (73 phút cho 1192 cặp), **0 lời gọi API** — chấm lại đúng bộ cặp đã cache.

Đã làm:
- `FrozenRanker` thêm tham số `scoring ∈ {listwise, pointwise}` (mặc định giữ `listwise` = §5.1 nguyên bản). Pointwise hỏi Yes/No độc lập cho từng candidate, batch phẳng theo `pointwise_chunk=64`.
- `--scoring` xuyên suốt `validate_reward.py`, `backfill_baselines.py`, `02_validate_reward.sh`. 4 test mới trong `tests/rl/test_reward_logic.py`.
- Chạy full 1192 cặp (8 arm × 149 user), 3B fp32, so sánh trực tiếp với listwise trên cùng bộ cặp.

Số đo:

| | listwise | pointwise | gpt-4o-mini |
|---|---:|---:|---:|
| Validation A — ρ gộp | **0.5573** | **0.4010** | — |
| Validation B — margin | +0.0120 | **+0.0661** | — |
| Độ nhạy thô `real − empty` | +0.0516 | **+0.0861** | +0.0994 |
| Trong-user, cặp NDCG phán được | 60.6% [50.8, 69.7] | 56.3% [46.7, 65.5] | — |
| Trong-user, gộp (296 cặp) | 47.0% [41.3, 52.6] | 48.6% [43.0, 54.3] | — |
| Tỉ lệ trùng NDCG@5 | 70.8% | 66.0% | 80.5% |
| Throughput (L4, fp32) | 1.38/s | 0.30/s | — |

Bug đã bắt trước khi nó làm hỏng số: bản đầu xếp hạng theo `softmax(P("Yes"))`. Model trả "No" cho gần như mọi candidate → `P(Yes) ≈ 1e-30` → **underflow về đúng 0.0** trong fp32, 8/10 candidate sập về cùng một giá trị, **tái tạo lại chính bài toán trùng mà pointwise sinh ra để diệt**. Đổi sang hiệu logit `Yes−No` (đơn điệu tương đương, ổn định số học) → cả 10 candidate phân biệt.

Lệch so với kế hoạch: không có lệch mới — đây đúng là nhánh dự phòng §M2 đã viết. Mặc định `scoring` giữ `listwise` vì pointwise không đạt Validation A.

Quyết định đã ra + lý do:
- **Nút thắt KHÔNG nằm ở scoring mode.** Hai thiết kế khác nhau về bản chất — một softmax 10 chiều tổng bằng 1, và mười điểm Yes/No hoàn toàn độc lập — cho **cùng một kết quả trong-user: ngẫu nhiên** (47.0% và 48.6%, CI của cả hai chứa 50%). Đổi cách hỏi không giải quyết được; giả thuyết mạnh nhất còn lại là **3B đơn giản quá yếu ở chính bài toán xếp hạng này** (NDCG@5 tuyệt đối: gpt-4o-mini 0.709 vs 3B 0.564–0.577, ngẫu nhiên 0.295).
- **Đính chính một phát biểu sai của em:** pointwise **không** "thoát trần 6 giá trị của NDCG@5". Trần đó nội tại trong *dạng* reward `f(thứ hạng gold)` — hai memory đặt gold vào cùng vị trí thì NDCG y hệt nhau bất kể scorer liên tục đến đâu. Pilot xác nhận: pointwise vẫn trùng 66%.
- **Nhưng pointwise thắng rõ ở vùng thô** (`real − empty` +0.0861 vs +0.0516, sát `LLM_Rec` thật +0.0994). Nếu sau này chọn hướng thu hẹp phát biểu về trục cost thì **dùng pointwise**, và phải giải bài throughput trước (0.30/s là không dùng được cho vòng lặp GRPO).

Căng thẳng kiến trúc mới lộ ra (quan trọng cho M4): §4.3 muốn ranker colocate cùng policy 4B + vLLM trên một H100 → trần ranker ~3B. Nhưng đo được rằng 3B quá yếu để làm proxy trung thực. Hai ràng buộc mâu thuẫn nhau; ba cách thoát (2 GPU / thu nhỏ policy / reward qua API) ở `docs/RESULTS.md`.

Việc tiếp theo — **cần người dùng quyết**, 4 hướng ở cuối mục M2 trong `docs/RESULTS.md`. Đề xuất ưu tiên: **ranker 7B/8B làm phép chẩn đoán** (~1–2 GPU-h trên GPU ≥40 GB) — rẻ nhất và loại trừ được nhiều nhất, vì nó kiểm tra trực tiếp giả thuyết mạnh nhất còn lại.

## M2 Phần B (tiếp) — đổi `LLM_Rec` sang gpt-5.6-luna: đóng họ giải pháp "nâng cấp người chấm" — 2026-08-10

Trạng thái: **M2 vẫn KHÔNG ĐẠT DoD**, nhưng đã loại trừ dứt điểm một họ giải pháp và **bác bỏ được rủi ro lớn nhất của việc đổi `LLM_Rec`**. Số liệu: `docs/RESULTS.md` mục "Đổi `LLM_Rec` sang model mạnh hơn".

Chi phí: **$1.40 API, 5.5 phút, 0 GPU.** Chấm lại đúng 1192 cặp `(user, arm)` đã cache + 298 lời gọi đối chứng nhiễu.

Đã làm:
- `src/rl/rescore_reference.py` — chấm lại các arm đã cache bằng một judge khác. Dựng lại chính xác facet của từng arm (`shuffled` phải lấy đúng cặp ghép theo thứ tự file lúc build, nếu ghép khác là so với một arm khác hẳn).
- Vá `src/models/llm_client.py`: điều kiện chọn `max_completion_tokens` / bỏ `temperature` trước đây chỉ khớp chuỗi `nano`, nay khớp cả họ `gpt-5`/`o1`/`o3`/`o4` bằng prefix. **Đường gpt-4o-mini không đổi** (§10.4).

Số đo:
- **Headroom tăng, không giảm:** `real − empty` = **+0.1759** trên Luna vs +0.1112 trên gpt-4o-mini. Lo ngại "reranker mạnh hơn thì memory thành thừa" đã bị bác bỏ bằng số.
- **ρ(Luna, gpt-4o-mini) = 0.6635** — vượt ngưỡng Validation A 0.6, và là cận dưới vì Luna tự nhiễu. Đây là thứ đầu tiên trong cả M2 đạt Validation A.
- **Validation B margin +0.1545** (proxy 3B listwise: +0.0120).
- **Tỉ lệ hoà trong-user: Luna 79.1% vs gpt-4o-mini 80.1%** — gần như y hệt.

Lệch so với kế hoạch: họ gpt-5 **không nhận `temperature=0`** (kiểm chứng trực tiếp với API). Reranker của repo chấm ở temp 0 để tất định, nên mọi judge họ gpt-5 là **ngẫu nhiên**. Đây không phải lựa chọn, là ràng buộc của API.

Quyết định đã ra + lý do:
- **Loại Luna khỏi vai reward trong vòng lặp M4.** Không phải vì tiền ($12.1/run là chấp nhận được) mà vì phép đối chứng nhiễu: chấm **cùng một memory** 2 lần đã tách 18.3% số cặp, trong khi chấm **hai memory khác nhau** tách 20.9% — **~88% khả năng phân biệt của nó là nhiễu của chính nó**. §5.1 chọn thiết kế one-forward-pass *vì* tính tất định; nhiễu reward đi thẳng vào variance của advantage khi GRPO không có critic.
- **Đóng vĩnh viễn họ giải pháp "nâng cấp người chấm".** 1.5B → 3B → pointwise → gpt-5.6-luna. Tỉ lệ hoà không nhúc nhích (80.1% → 79.1%) khi đổi sang model mạnh hơn hẳn → **trần nằm ở *bài toán*** (10 candidate, NDCG@5 chỉ 6 giá trị, hai memory tốt thường đặt gold cùng vị trí), không nằm ở người chấm. Mọi nỗ lực tiếp theo theo hướng này là lãng phí.
- **Giữ mở phương án Luna làm `LLM_Rec` cho bảng kết quả cuối** (không phải cho vòng lặp). Nhiễu triệt tiêu khi lấy trung bình trên 993 test user; chi phí chạy lại M0 chỉ ~$2.45; và mọi số tuyệt đối đều tốt hơn. Quyết định để lại M7a.

Ghi chú phương pháp luận đáng giữ: hình dạng thống kê "~20% cặp tách được, biên độ ~0.33" **là thứ nhiễu thuần tuý cũng tạo ra**, vì NDCG@5 rất thô. Kết luận "tín hiệu tinh có thật và lớn" của lần đo trước vẫn đứng (gpt-4o-mini chấm ở temp 0, tất định), nhưng từ nay mọi phát biểu về phân biệt trong-user phải kèm đối chứng nhiễu của chính judge đó.

Việc tiếp theo — **cần người dùng quyết**. Hướng duy nhất chưa thử và tấn công đúng chỗ trần thật sự nằm: **đổi *dạng* reward**, bỏ `f(thứ hạng gold)` sang đại lượng liên tục. Ứng viên rẻ nhất: `LLMReranker` vốn trả về **điểm số cho từng candidate**, nhưng `_score` hiện chỉ lưu `ranking`/`ndcg_at_5`/`hit_at_1` và **vứt điểm thô đi**. Chấm lại 1192 cặp bằng gpt-4o-mini ở temp 0 có lưu điểm thô (**~$0.52, tất định**) là đủ để trả lời: biên `điểm(gold) − max(điểm khác)` có thoát được trần hoà 80% không.

## M2 Phần B (tiếp) — đổi *dạng* reward sang `gold_margin`: blocker §9.2 đã hạ xuống dưới ngưỡng — 2026-08-10

Trạng thái: **hướng đi đã tìm được.** Trần hoà — thứ mà mọi nâng cấp người chấm không chạm tới (80.1% → 79.1%) — bị hạ bằng cách đổi *dạng* reward. Số liệu: `docs/RESULTS.md` mục "Đổi *dạng* reward: `gold_margin`".

Chi phí: **$0.85 API, 12 phút, 0 GPU** (1192 cặp + 745 cặp chấm lặp lấy nền nhiễu).

Đã làm:
- `_score` trong `build_val_reference.py` giờ lưu thêm `scores` (điểm 0–1 cho từng candidate), `gold_score`, `gold_margin`. Trước đây `LLMReranker` trả về điểm thô rồi bị **vứt đi** ngay sau khi sort — không khôi phục lại được nếu không trả tiền chạy lại.
- Chấm lại 1192 cặp bằng gpt-4o-mini @ temp 0, rồi chấm lặp 745 cặp lần hai để lấy nền nhiễu của từng đại lượng.

Số đo:

| Đại lượng | Tách 2 memory khác nhau | Nền nhiễu | Tín hiệu THẬT |
|---|---:|---:|---:|
| `ndcg_at_5` | 19.3% | 8.6% | 10.7 điểm |
| `gold_score` | 21.6% | 8.1% | 13.5 điểm |
| **`gold_margin`** | **34.1%** | 13.4% | **20.7 điểm** |

- **Group suy biến (`std(r)=0`): NDCG@5 67.8%/62.4% → `gold_margin` 42.6%/43.6%** (hai lần chạy độc lập). Từ trên ngưỡng báo động 60% của §9.2 xuống dưới, và ổn định.
- **Margin đồng hướng với NDCG@5**, không cãi nhau: 88.5% [83.2, 92.3] cùng lần chạy, **76.2% [69.1, 82.2] tái lập chéo hai lần chạy độc lập**. ρ gộp 0.8670. Tương phản với `p_gold` của `soft_weight`: 40.1%, dưới ngẫu nhiên.
- Margin mang tín hiệu chất lượng memory: headroom +0.0583 trên thang 0.1674 = **35% tương đối** (NDCG@5: 12%).

Lệch so với kế hoạch: **gpt-4o-mini @ `temperature=0` KHÔNG tất định** — 8.6% cặp đổi điểm giữa hai lần chạy y hệt. §5.1 giả định người chấm tất định; giả định đó sai kể cả với temp 0 qua API. Mọi tỉ lệ "tách được" từ nay phải báo cáo kèm nền nhiễu của chính đại lượng đó.

Quyết định đã ra + lý do:
- **`gold_margin` là dạng reward được chọn để đi tiếp.** Nó tăng nhiễu (8.6% → 13.4%) nhưng tăng khả năng tách nhanh hơn nhiều (19.3% → 34.1%), nên tín hiệu thật gần gấp đôi. Quan trọng hơn: nó **đồng hướng** với NDCG@5 chứ không thay thế nó — đây là khác biệt bản chất so với `p_gold`, thứ được giao đúng những cặp NDCG không phán được và trên đó nó phản tín hiệu.
- **Giữ NDCG@5 làm metric BÁO CÁO.** Margin là tín hiệu huấn luyện; bảng kết quả vẫn phải là H@k/N@k để so được với paper gốc.

Việc tiếp theo — **cần người dùng quyết**, hai câu hỏi độc lập:
1. **Ai tính reward trong vòng lặp M4?** (a) gpt-4o-mini qua API: ~$5.6/run, tương quan hoàn hảo theo định nghĩa, 0 VRAM, nhưng 8.6% nhiễu; (b) proxy 3B local: **chưa biết có bám được margin không** — mọi phép đo Validation A/within-user trước đây dùng NDCG@5 ở cả hai phía, mục tiêu vừa đổi nên kết luận cũ không tự động áp dụng. Trả lời được bằng ~1.4 GPU-h trên L4 (`FrozenRanker` đã trả softmax nên margin tính được ngay; `m2_pairs*.json` chỉ lưu `proxy_p_gold`, thiếu max của phần còn lại).
2. **`LLM_Rec` cuối cùng là gpt-4o-mini hay gpt-5.6-luna?** Luna cho headroom +0.1759 vs +0.1112 và mọi số tuyệt đối tốt hơn, chi phí chạy lại M0 ~$2.45 — nhưng nhiễu gấp đôi. Để lại M7a.

## M2 — ĐẠT — 2026-08-10

Trạng thái: **DONE.** Validation A ✅ 0.7726 · B ✅ · C ⏸ không đo được ở tầng T1. **M4 hết bị chặn.** Số liệu đầy đủ: `docs/RESULTS.md` mục "M2 Reward Validation".

Chi phí ngày hôm nay: **~$5.5 API + ~4 GPU-hour trên L4.** Vẫn chưa thuê H100 một giờ nào. Tổng M2: ~$6.5 + ~6.5 GPU-h rẻ.

Đã làm:
- `src/rl/rescore_reference.py` — chấm lại arm đã cache bằng judge khác (đổi người chấm rẻ vì memory là nửa đắt tiền và đã nằm trên đĩa).
- `src/rl/measure_margin.py` — đo dạng reward liên tục trên ranker local, mỗi tỉ lệ kèm nền nhiễu riêng.
- `_score` của `build_val_reference.py` lưu thêm điểm thô từng candidate (`scores`, `gold_score`, `gold_margin`) — trước đây `LLMReranker` trả về rồi bị vứt ngay sau khi sort.
- `FrozenRanker._templated()` + `letter_mass()`; `llm_client.py` nhận diện họ model bằng prefix thay vì chuỗi con `nano`.
- Nâng `transformers` 4.57.1 → 5.14.1 (cần cho `qwen3_5`), build `causal-conv1d 1.6.2.post1` từ source. `pytest tests/rl/` giữ 146 pass / 5 skip qua cả hai.

Số đo chính:
- **Ranker `Qwen/Qwen3.5-4B`**: ρ = **0.7726** (1.5B 0.3071 → 3B 0.5573 → pointwise 0.4010). NDCG@5 tuyệt đối 0.736–0.756, **cao hơn gpt-4o-mini** (0.703–0.722). Headroom +0.1310. bf16 bất biến batch 0/48 (torch fallback).
- **Reward `r_ndcg + 0.02·margin_logit`**: group suy biến **71.1% → 0.0%**, within-user **62.9%** [57.1, 68.4], phá hoà ở 58.4% [50.9, 65.5].
- **Trần hoà là của bài toán, không của người chấm**: gpt-4o-mini 80.1%, gpt-5.6-luna 79.1%.
- **Tiền đề đồ án sống sót reranker mạnh hơn**: headroom trên Luna +0.1724 vs gpt-4o-mini +0.1112 — memory KHÔNG thành thừa khi reranker khoẻ lên.

Lệch so với kế hoạch:
1. **Ranker `Qwen2.5-1.5B` → `Qwen/Qwen3.5-4B`** (§6.1 ghi 1.5B, fallback "giữ nguyên"). Ảnh hưởng §4.3: ranker bf16 8.4 GB, **không cần fp32** vì model này bất biến batch — rẻ hơn phương án 3B fp32 (~12 GB).
2. **Dạng reward đổi từ `f(thứ hạng gold)` sang `r_ndcg + w·margin_logit`.** §5.1 chỉ có `r_ndcg`. Không phải tinh chỉnh: với reward chỉ theo rank thì 71% group không có gradient, tức M4 chạy 400 step trên 29% dữ liệu.
3. **`soft_weight` bị THAY THẾ, không bật lại.** Cùng vai trò, nhưng `p_gold` phá hoà ở 40.1% (dưới ngẫu nhiên) còn `margin_logit` ở 58.4% (trên).
4. **`temperature=0` không khả dụng cho họ gpt-5**, và **gpt-4o-mini @ temp 0 cũng không tất định** (8.6%). §5.1 xây trên tiền đề "người chấm tất định" — tiền đề đó sai kể cả với API.

Quyết định đã ra + lý do:
- **Reward chạy local, không gọi API trong vòng lặp.** Đã đo `gpt-5.6-luna` làm reward và loại: nó tách 94.0% cặp trên nền nhiễu **92.9%** → 1.1 điểm tín hiệu thật. Nó là evaluator giỏi nhất đã đo nhưng reward tồi, vì API không cho tắt sampling. Local ranker gỡ luôn cả nhiễu, chi phí và VRAM.
- **Mọi tỉ lệ "tách được" từ nay phải kèm nền nhiễu của chính nó.** Hai ứng viên bị loại *chỉ nhờ* phép này, cả hai đều cho "0% group suy biến" và đều trông như chiến thắng: `luna + margin` và `margin_prob`.
- **Validation C ghi ⏸ chứ không ❌.** L4 24 GB OOM ở batch 64 và bão hoà từ batch 32 — không phải chỗ để kiểm ngưỡng viết cho H100. §11.5 đã lên lịch đo ở đầu phiên M3-B/M4.

Bug đã bắt (đều im lặng, đều suýt cho kết luận ngược):
1. **Chat template của reasoning model.** Qwen3.5 kết thúc generation prompt trong khối `<think>` đang mở → scorer đọc token đầu của chuỗi suy luận. Letter mass 0.000020. ρ 0.1232 → 0.7726 sau khi đóng khối. Hiện ra y hệt "model quá yếu".
2. **`margin_prob` bão hoà.** Ranker dồn ~99% mass vào một token → margin xác suất bão hoà ±1, phần nhúc nhích là kernel. Chỉ nền nhiễu bắt được.
3. **`llm_client` nhận diện model bằng chuỗi con `nano`** → bắt được gpt-5-nano và không gì khác trong họ.

Việc tiếp theo: **M3.** Phần A (CPU + API, ~$7): sinh 8 mẫu `M_collab`/user cho 1185 train user bằng gpt-4o-mini. Phần B (~4 GPU-h): chấm bằng ranker Qwen3.5-4B đã validate, giữ top-1/user nếu `r > r_null`, SFT LoRA. Đo lại Validation C ở đầu phiên GPU.

## M3 + ĐÓNG DỰ ÁN — 2026-08-11

Trạng thái: **M3 XONG (chưa eval). M4 KHÔNG THỰC HIỆN. Dự án đóng.** Tổng kết đầy đủ ở đầu `docs/RESULTS.md`.

### M3 đã làm

- **Phần A** (CPU + API): `src/rl/build_sft_teacher.py` — 1185 train user × 8 mẫu `M_collab` từ gpt-4o-mini @ temp 1.0. **$2.88, 18.8 phút.** 0/9480 mẫu rỗng, 0/1185 user có cả 8 mẫu giống hệt, median 276 token.
- **Phần B bước 1** (GPU): `src/rl/select_sft_data.py` — chấm 9480 mẫu bằng reward M2 đầy đủ, giữ top-1/user nếu thắng `r_null`. **457/1185 user (38.6%)**; 728 bị loại, phần lớn là số học vì **40.8% user có `r_null` = 1.0** (gold đã đứng nhất khi không có memory, nên `r > r_null` không thể thoả).
- **Phần B bước 2** (GPU): `src/rl/sft.py` — LoRA r=32/α=64 all-linear, 2 epoch, lr 1e-5, `enable_thinking=False`, loss chỉ trên completion. **116 step, train_loss 0.3902, 24.8 phút, VRAM peak 15.1 GB** → `checkpoints/rl/sft_books`.
- `src/rl/eval_sft.py` đã viết (DoD M3: JSON hợp lệ ≥95%, NDCG@5 không tụt so với base) nhưng **chưa chạy**.

### Số đo quyết định việc đóng dự án

- **Dự báo group suy biến trên tập train thật**, đúng hình dạng 8 mẫu/group mà M4 sẽ lấy: `std(r) = 0` ở **0/1185 = 0.0%** (ngưỡng báo động §M4: 60%). M4 hết bị chặn về kỹ thuật.
- **Nhưng phân rã ra thì yếu hơn nhiều:** **734/1185 = 61.9%** group **không có khác biệt thứ hạng nào** (NDCG@5 phẳng). Ở những group đó toàn bộ hướng gradient do `margin_logit` quyết định, mà nó chỉ đồng ý với `LLM_Rec` thật **58.4%** [50.9, 65.5]. Chỉ **38.1%** group có tín hiệu rank thật.
- **Thí nghiệm cuối — nới candidate list của reward lên N=26** (`src/rl/measure_wider_candidates.py`, 5.7 phút GPU, 0 API). Giả thuyết: trần trùng thuộc về protocol N=10 nên nới độ phân giải sẽ phá được. **Bác bỏ:** tỉ lệ hoà giảm thật (MRR@26 hoà 67.0% vs NDCG@5@10 hoà 83.6%) nhưng độ chính xác giảm tương ứng, và tín hiệu ròng không đổi (+43 vs +42) hoặc tệ hơn khi cộng margin (+56 vs +71). Các cặp hoà **không giấu tín hiệu dùng được**.

### Quyết định đã ra + lý do

- **Đóng dự án.** Chủ đồ án đặt điều kiện cứng: đóng góp phải là accuracy tốt hơn MemRec gốc **≥ +0.05 NDCG@5**, và nói rõ đóng góp về cost không đáp ứng yêu cầu. Trần oracle đo được của Stage-R synthesis là **+0.06→+0.08**, reward tốt nhất đúng 62.9% within-user, ước lượng thực tế **+0.02→+0.035**, xác suất chạm +0.05 khoảng **15–20%**. Không đủ để cam kết 25 GPU-hour còn lại.
- **Dừng trước khi thuê H100.** Tổng tiêu: **~$16 API + ~13 GPU-h trên L4**, ~11% ngân sách §11, trong 5 ngày / 14 tuần. Đây đúng là việc chế độ LEAN §2.5 và cổng M2 được thiết kế để làm — mua kết luận chặn đường bằng tiền lẻ thay vì bằng vài phiên H100.
- **Không thử tiếp ranker 9B hay G=16.** Đã có ba điểm dữ liệu cho thấy nâng cấp scorer không chạm được trần (3B→4B đưa within-user 47%→63%, không phải 80%; và gpt-5.6-luna mạnh hơn hẳn gpt-4o-mini vẫn hoà 79%). Thử tiếp là kéo dài mà không đổi bản chất.

### Giá trị còn lại

Hai đóng góp phương pháp luận, đúng và dùng được độc lập với kết quả accuracy — chi tiết ở `docs/RESULTS.md`:
1. **ρ gộp là metric sai** để validate reward proxy cho GRPO (bị chi phối bởi phương sai giữa các user, thứ group không bao giờ thấy).
2. **Mọi tỉ lệ "phân biệt được X%" phải kèm nền nhiễu của chính nó.** Ba ứng viên bị loại *chỉ nhờ* phép này, cả ba đều cho "0% group suy biến" và đều trông như chiến thắng.

Ba bug im lặng cũng đã ghi lại (chat template của reasoning model; default trùng lặp hai lần trong cùng một file; `StageRReward` chưa bao giờ batch ranker).

### Việc còn dang dở

`eval_sft` chưa chạy (~20 phút GPU) · Validation C chưa đo được ở tầng T1 · bug RNG của M0 chưa sửa · bảng chính §8 để trống vì không có M4.
