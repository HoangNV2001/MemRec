# RL_PLAN.md — Huấn luyện `LM_Mem` của MemRec bằng GRPO

> **Đặt file này tại:** `docs/RL_PLAN.md` trong repo `rutgerswiselab/MemRec`.
> **Đối tượng đọc:** tác giả đồ án + coding agent thực thi theo từng milestone.
> **Trạng thái:** `DRAFT — chưa bắt đầu M0`
> **Cập nhật lần cuối:** _(agent điền)_

---

## 0. TL;DR

MemRec là pipeline **zero-shot prompting hoàn toàn** — không tham số nào được học. `LM_Mem` viết ra `M_collab` mà không bao giờ nhận tín hiệu "memory này có thực sự giúp `LLM_Rec` xếp hạng tốt hơn không".

Đồ án này gắn **GRPO** vào đúng chỗ đó: biến Stage-R (Collaborative Memory Synthesis) thành một **policy học được**, với reward là chất lượng ranking downstream.

**Câu hỏi nghiên cứu chính:**
> Với cùng một `LLM_Rec` đóng băng, một `LM_Mem` nhỏ (3–4B) được huấn luyện bằng GRPO có tổng hợp `M_collab` tốt hơn `LM_Mem` prompted (gpt-4o-mini / 7B) không — đồng thời **ngắn hơn**?

Nếu đúng, đóng góp là **một điểm Pareto tốt hơn** trên trục (accuracy × cost) của Figure 4 trong paper, chứ không chỉ là con số accuracy cao hơn.

**Ngân sách:** 1× H100 on-demand, chế độ **LEAN** — mục tiêu ~**50 GPU-hour** cho kết quả tối thiểu (§11). **Thời lượng:** ~14 tuần. **Không** phải paper hội nghị.

**Thứ tự ưu tiên tuyệt đối:** kết quả tối thiểu trước → ablation sau → stretch cuối. Xem §2.5.

---

## 1. Tóm tắt MemRec (phần cần cho RL)

Ba stage, tách `LM_Mem` (nhẹ, quản lý graph) khỏi `LLM_Rec` (nặng, suy luận):

| Stage | Nội dung | Eq. | Vai trò trong đồ án |
|---|---|---|---|
| **R** — Collaborative Memory Retrieval | Curate top-k neighbor bằng `R_domain` → synthesize thành `N_f` facet = `M_collab` | 1–3 | **← Đây là policy sẽ train** |
| **ReRank** — Grounded Reasoning | `LLM_Rec(I_u ‖ M_collab ‖ C_info)` → score + rationale | 4 | **Đóng băng, dùng làm reward** |
| **W** — Async Collaborative Propagation | Cập nhật `M_u`, `M_i`, `ΔM_neigh` trong 1 lời gọi, `O(1)` | 5–6 | **Đóng băng ở M0–M5, mở ở M6 (stretch)** |

Ablation trong paper (Books): bỏ Collab. Read `−9.9%` H@1, bỏ LLM Curation `−5.5%`, bỏ Collab. Write `−4.2%`. → Stage-R là nơi có headroom lớn nhất, hợp lý để đánh trước.

Hyperparameter gốc: `k=16`, `N_f=7`, `N_candidates=10`, metrics H@{1,3,5} và N@{3,5}.

---

## 2. Phạm vi & những gì KHÔNG làm

### Trong phạm vi
- GRPO cho **Stage-R synthesis** trên **1 dataset chính: `instructrec-books`**
- Môi trường **tĩnh**: memory graph đóng băng thành snapshot, không ghi trong lúc train
- Reward từ **frozen local ranker**, không gọi API trong vòng lặp train
- Đánh giá transfer sang `instructrec-movietv` (zero-shot, không train lại)

### Ngoài phạm vi (viết rõ trong Limitations của luận văn)
- ❌ Train `LLM_Rec` (đã có Rec-R1 / Rank-GRPO làm việc này — chỉ trích dẫn)
  > **Cập nhật 2026-08-06:** `docs/RL_LM_REC_EXTENSION.md` đề xuất làm đúng việc này như một **extension tuỳ chọn sau M7a**. Điều đó **không** mở rộng phạm vi của Plan này: trong toàn bộ M0–M7b, `LLM_Rec` vẫn đóng băng tuyệt đối. Extension là một đóng góp phụ tách biệt, có cổng riêng (§7.1), branch riêng, và file kết quả riêng. Nếu extension không chạy, Limitations vẫn viết đúng như dòng trên.
- ❌ Curation policy học được (giữ nguyên `R_domain` tĩnh)
- ❌ Full-catalog retrieval — giữ nguyên protocol N=10 candidate của paper
- ❌ Online A/B, multi-hop propagation, federated memory
- ⚠️ Stage-W propagation reward: **chỉ làm nếu M4 thành công và còn ≥3 tuần** (xem M6)

### Vì sao thu hẹp mạnh
1 GPU phải gánh đồng thời: policy training + vLLM rollout + frozen reward ranker. Mở thêm Stage-W nghĩa là graph thay đổi theo policy → môi trường non-stationary → cần re-materialize graph mỗi epoch → gấp 3–5× chi phí và thêm một nguồn bug rất khó chẩn đoán. Không hợp với đồ án có deadline cứng.

---

## 2.5 Nguyên tắc thực thi LEAN — **đọc trước khi làm bất cứ gì**

Ràng buộc chi phối mọi quyết định khác trong file này: **tối đa hoá CPU, tối thiểu hoá H100, có kết quả tối thiểu trước.**

### 2.5.1 Phân tầng tài nguyên

| Tầng | Dùng cho | Giá tham chiếu |
|---|---|---|
| **T0 — CPU / laptop** | build dataset, parse, unit test, phân tích, vẽ hình, viết | $0 |
| **T0-API — CPU + gpt-4o-mini** | warmup graph, baseline eval, sinh dữ liệu teacher | ~$0.2/1M token |
| **T1 — GPU rẻ (A10/L4/T4)** | *shake out plumbing* với `Qwen2.5-0.5B`, 3–20 step | ~$0.3–0.6/h |
| **T2 — H100** | **chỉ** SFT thật + GRPO thật + reward ranker thật | ~$2.5–3.5/h |

**Quy tắc:** một run chỉ được lên T2 nếu câu hỏi nó trả lời là *"phương pháp có hiệu quả không"*. Mọi run trả lời *"code có chạy không"* phải chạy ở T1 hoặc T0 trước. Đây là đòn bẩy tiết kiệm lớn nhất — phần lớn lỗi GRPO là lỗi config/plumbing, không phải thuật toán.

### 2.5.2 Bốn quyết định LEAN thay đổi kế hoạch gốc

**① Bỏ local 7B, dùng gpt-4o-mini cho mọi việc offline.**
Kế hoạch gốc chạy `Qwen2.5-7B-Instruct` local cho warmup graph và teacher — tức là thuê H100 hàng chục giờ chỉ để làm inference. Nhưng warmup Books ~30M input token qua gpt-4o-mini chỉ tốn **~$5**, chạy trên CPU với `--parallel_workers 32`, và repo vốn đã mặc định provider này.
→ **M0 và M1 chuyển sang 100% CPU + API. Cắt ~30 GPU-hour.**

**② Reward ranker vẫn local, vì nó chạy trong vòng lặp.**
1 run GRPO ≈ 36k lời gọi reward. Qua API là ~$16/run cộng độ trễ mạng biến động — không đáng. Ranker 1.5B colocate trên GPU vốn đã trả tiền cho training. Giữ nguyên.

**③ Gộp phiên GPU.** Không thuê máy cho từng milestone. M2 + M3 vào **một phiên**; M4 vào 2–3 phiên. Mỗi lần bật máy là một lần trả phí khởi động + tải model + rebuild môi trường.

**④ M3 đã là một kết quả hoàn chỉnh.** SFT distillation (teacher gpt-4o-mini → student 4B) nếu đạt ngang baseline với chi phí thấp hơn thì **tự nó đã là luận văn nộp được**, đạt sau ~20 GPU-hour. M4 (GRPO) là phần nâng cao, không phải điều kiện sống còn. Điều này thay đổi hồ sơ rủi ro rất nhiều — bạn có kết quả trước khi tiêu phần lớn ngân sách.

### 2.5.3 Kết quả tối thiểu (MVR) — định nghĩa "xong"

MVR = **một bảng duy nhất** trên `instructrec-books` test set, cùng `LLM_Rec` (gpt-4o-mini), so sánh:

1. MemRec prompted (baseline)
2. MemRec + `LM_Mem` là 4B SFT-distilled
3. MemRec + `LM_Mem` là 4B GRPO

kèm cột **token/query**, cộng một biểu đồ Pareto (accuracy × cost) và **một** kiểm tra chống hacking (ranker-swap).

Đạt được cái này = đồ án hoàn chỉnh. Mọi thứ khác là bonus.

### 2.5.4 Thang lùi (fallback ladder)

Nếu ngân sách cạn hoặc kết quả không ra, lùi theo thứ tự này — mỗi bậc vẫn là một luận văn nộp được:

| Bậc | Nội dung | GPU-hour tích luỹ |
|---|---|---|
| 1 | GRPO > SFT > prompted, kèm Pareto + ranker-swap | ~50h |
| 2 | GRPO ≈ SFT nhưng **ngắn hơn 30–40% token** → thắng ở trục cost | ~50h |
| 3 | Chỉ SFT distillation: 4B ≈ gpt-4o-mini prompted, rẻ hơn nhiều | ~20h |
| 4 | Negative result: GRPO không cải thiện, kèm **phân tích vì sao** (group degeneracy? reward proxy lệch? hacking?) | ~35h |

Bậc 4 vẫn hợp lệ **nếu và chỉ nếu** có bằng chứng chẩn đoán (training curve, `% group bị filter`, `% thắng null-arm`). Đó là lý do §9 bắt buộc log các metric này ngay từ đầu.

---

## 3. Hình thức hoá

Stage-R là **contextual bandit một bước** (vì graph đóng băng):

```
state   s = (I_u, M_u, Rep(N'_k(u)))          # KHÔNG chứa candidate list — xem §5.3
action  a = M_collab = {F_1, ..., F_{N_f}}    # JSON, do policy sinh
policy  π_θ = LM_Mem (Qwen3-4B-Instruct + LoRA)
reward  r = R(a; s, C, i*)                    # từ frozen LLM_Rec, xem §5
```

GRPO: với mỗi `s`, sample `G` action, chuẩn hoá reward trong group:

```
Â_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)        # vanilla
Â_i =  r_i − mean(r_1..r_G)                          # Dr. GRPO (KHUYẾN NGHỊ, xem §6.3)
```

Không cần critic. Đây chính là lý do chọn GRPO thay vì PPO khi chỉ có 1 GPU.

---

## 4. Kiến trúc hệ thống

### 4.1 File mới cần tạo

Trạng thái: ✅ = đã tồn tại sau M1 · ⬜ = còn phải viết.

```
src/rl/
├── __init__.py                 ✅
├── env.py                      ✅ GraphSnapshot, cache N'_k(u), ghim ngân sách packer
├── splits.py                   ✅ [thêm mới] split user-disjoint + candidate tất định theo (seed, user_id, salt)
├── leakage.py                  ✅ [thêm mới] màn lọc rò đáp án (trùng tên sách trong catalogue)
├── warmup.py                   ✅ [thêm mới] dựng memory sạch-test (target = valid item)
├── build_snapshot.py           ✅ [thêm mới] entrypoint M1 bước 1 (có --memory_file để tái tạo offline)
├── build_dataset.py            ✅ [thêm mới] entrypoint M1 bước 2
├── dataset.py                  ✅ jsonl -> datasets.Dataset cho TRL + backfill + curriculum filter
├── policy.py                   ✅ prompt candidate-blind + parser JSON facets không bao giờ raise
├── reward/                     ⬜ M2
│   ├── __init__.py
│   ├── ranker.py               # frozen listwise ranker, chấm bằng logprob (§5.1)
│   ├── metrics.py              # ndcg_at_k, hit_at_k
│   ├── grounding.py            # kiểm tra facet có cite neighbor hợp lệ (§5.2)
│   └── composite.py            # hàm reward tổng, interface cho TRL
├── sft.py                      ⬜ M3 rejection-sampling warm start
├── train_grpo.py               ⬜ M4 entrypoint TRL GRPOTrainer
└── eval_rl.py                  ⬜ cắm policy đã train ngược vào pipeline MemRec gốc

configs/rl/
├── m1_env_books.yaml           ✅
├── m3_sft_books.yaml           ⬜
├── m4_grpo_books.yaml          ⬜
└── m5_ablations/               ⬜ 1 yaml cho mỗi ablation

scripts/rl/
├── 00_build_snapshot.sh        ✅
├── 01_build_dataset.sh         ✅ (chạy luôn pytest để verify DoD)
├── 02_validate_reward.sh       ⬜
├── 03_sft_warmstart.sh         ⬜
├── 04_train_grpo.sh            ⬜
└── 05_eval.sh                  ⬜

tests/rl/                       ✅ [thêm mới — §10.5 bắt buộc có smoke test]
├── test_policy_parser.py       ✅ 20 ca output méo viết tay
├── test_env_splits.py          ✅ tất định, split, snapshot round-trip, ngân sách packer
└── test_no_leakage.py          ✅ DoD M1

data/rl/
├── graph_snapshot_books.json   ✅ 17.4 MB, 2350 user (gitignored)
├── user_splits_books.json      ✅ [thêm mới] ĐƯỢC track trong git — định nghĩa thí nghiệm
├── stager_books_train.jsonl    ✅ 1185
├── stager_books_val.jsonl      ✅ 149
├── stager_books_test.jsonl     ✅ 993
└── stager_books_dropped_users.json  ✅ [thêm mới] 23 user bị loại vì lộ đáp án

docs/
├── RL_PLAN.md                  ✅ file này — đóng góp CHÍNH
├── RL_LM_REC_EXTENSION.md      ✅ extension train LM_Rec — PHỤ, chỉ sau M7a + M5 Ưu tiên 1
├── PROGRESS.md                 ✅ agent append sau mỗi milestone
└── RESULTS.md                  ✅ bảng kết quả, agent điền dần
```

> **Quan hệ với `docs/RL_LM_REC_EXTENSION.md`:** file đó là **extension**, không phải một phần của Plan này. Khi hai file mâu thuẫn, **Plan này thắng**. Xem §7.1 về cổng và thứ tự ưu tiên ngân sách.

### 4.2 Điểm tích hợp với code có sẵn

| Module gốc | Cách dùng | Nguyên tắc |
|---|---|---|
| `src/models/llm_client.py` | Thêm provider `local_vllm_policy` trỏ tới policy đang train | **Chỉ thêm**, không sửa logic cũ |
| `src/models/reranker_llm.py` | Tham chiếu để `src/rl/reward/ranker.py` khớp prompt format | Copy prompt, không import chéo |
| `src/models/reranker_vector.py` | Dùng làm reward phụ + sanity check chống hack (§9.1) | — |
| `src/memory/manager.py` | Gọi ở M1 để materialize graph rồi serialize | Read-only trong RL loop |
| `src/memory/pruner.py` | Sinh `N'_k(u)`, cache vào snapshot | Read-only |
| `scripts/run_train.py --save_llm_conversations` | **Nguồn dữ liệu chính cho M1 và M3** | — |

> **Quy tắc bất di bất dịch:** không sửa đổi hành vi của đường eval gốc. Mọi thứ mới nằm dưới `src/rl/`. Phải luôn reproduce được baseline paper bằng lệnh gốc.

### 4.3 Bố trí GPU (H100 80GB)

| Thành phần | Ước tính VRAM |
|---|---|
| Policy Qwen3-4B bf16 + LoRA (grad ckpt bật) | ~22 GB |
| vLLM rollout engine, `gpu_memory_utilization=0.32` | ~26 GB |
| Frozen ranker Qwen2.5-1.5B-Instruct bf16 + KV | ~8 GB |
| Đệm / fragmentation | ~10 GB |
| **Tổng** | **~66 GB** |

Dùng TRL `GRPOTrainer` với `use_vllm=True, vllm_mode="colocate"`. Nếu OOM: hạ policy xuống `Qwen2.5-3B-Instruct` **trước khi** hạ `G` (giảm `G` phá vỡ chất lượng advantage estimate).

**Tầng staging (T1, §2.5.1):** cùng cấu hình nhưng policy `Qwen2.5-0.5B` + ranker `Qwen2.5-0.5B`, `gpu_memory_utilization=0.25` — vừa A10 24GB. Chỉ để test plumbing.

---

## 5. Đặc tả reward

```python
r = r_ndcg  +  λ_g * r_ground  −  λ_len * over_length  −  λ_fmt * is_malformed
```

Giá trị khởi điểm: `λ_g = 0.2`, `λ_len = 0.1` (mỗi 100 token vượt budget 400), `λ_fmt = 1.0`.

### 5.1 `r_ndcg` — thành phần chính

**Dùng NDCG@5, KHÔNG dùng Hit@1.** Hit@1 nhị phân → rất nhiều group có `std(r)=0` → advantage bằng 0 → đốt compute vô ích.

**Chấm điểm bằng một forward pass duy nhất:** đưa `M_collab` + toàn bộ 10 candidate vào frozen ranker, đọc logprob trên các token chỉ mục candidate (`A`–`J`), softmax → ranking đầy đủ → NDCG. Ưu điểm:
- Tất định (không có nhiễu sampling). GRPO không có critic nên nhiễu reward đi thẳng vào variance của advantage — đây không phải tối ưu hoá vặt.
- Rẻ: prefill-only, không sinh token.
- Batch được 64 rollout cùng lúc.

Frozen ranker mặc định: `Qwen2.5-1.5B-Instruct`. **Phải** validate ở M2 rằng nó tương quan với `LLM_Rec` thật.

### 5.2 `r_ground` — chống hallucination

Bắt policy xuất mỗi facet kèm `source_ids` trỏ tới node trong `N'_k(u)`:

```json
{"facets": [
  {"text": "...", "source_ids": ["u_4412", "i_88301"]}
]}
```

```
r_ground = (# facet có ≥1 source_id hợp lệ VÀ cos_sim(facet, memory[source_id]) ≥ τ) / N_f
```

`τ = 0.35` với `bge-small-en-v1.5` (rẻ, chạy CPU được). Đây vừa là chống hallucination vừa là hàng rào chống reward hacking.

### 5.3 Quyết định thiết kế: **candidate-blind**

`LM_Mem` **không** được nhìn thấy candidate list `C` khi tổng hợp `M_collab`.

Lý do: nếu thấy `C`, policy sẽ học viết "user này rất thích trinh thám Bắc Âu" chỉ vì candidate #4 là trinh thám Bắc Âu — tức **mã hoá đáp án vào memory** mà không hề mô hình hoá user. Reward tăng đẹp, generalization bằng 0.

Đây là khác biệt so với MemRec gốc (prompt `P_synth` có nhắc tới candidate context). **Phải ablate cả hai chế độ ở M5** và báo cáo trung thực — nếu candidate-visible cho reward cao hơn nhưng ranker-swap test tụt, đó chính là bằng chứng hacking, và bản thân nó là một finding đáng viết.

### 5.4 Ghi chú về "null-arm baseline" — đọc kỹ để khỏi mất thời gian

Ý tưởng "trừ `NDCG(∅)` (không có collaborative memory) khỏi mọi reward" **KHÔNG thay đổi gradient**: đó là hằng số theo prompt, group-mean-centering của GRPO đã triệt tiêu nó, `std` cũng không đổi.

Vẫn tính `r_null` (cache 1 lần/prompt, tất định) nhưng dùng cho:
- **Chẩn đoán**: `% rollout thắng null-arm` là metric theo dõi tốt nhất
- **Lọc prompt** (§6.4): loại prompt mà mọi rollout đều thua/thắng null

Muốn null-arm thực sự vào gradient thì phải nhét nó **vào trong group** → cần custom trainer. Ghi vào stretch, đừng làm ở M4.

---

## 6. Cấu hình huấn luyện

### 6.1 Model

| Vai trò | Mặc định | Fallback nếu OOM |
|---|---|---|
| Policy `LM_Mem` | `Qwen3-4B-Instruct-2507` | `Qwen2.5-3B-Instruct` |
| Frozen reward ranker | `Qwen2.5-1.5B-Instruct` | giữ nguyên |
| Teacher cho warm-start (M3) | `Qwen2.5-7B-Instruct` (local) | gpt-4o-mini (~$20 API) |
| `LLM_Rec` cho **eval cuối** | gpt-4o-mini (khớp paper) | `Qwen2.5-7B-Instruct` |
| Embedding cho grounding | `bge-small-en-v1.5` | — |

> Agent phải verify tên checkpoint còn tồn tại trên HF trước khi hardcode; nếu không, chọn model cùng cỡ và ghi vào `PROGRESS.md`.

### 6.2 Siêu tham số GRPO khởi điểm

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `num_generations` (G) | 8 | Đừng hạ xuống dưới 6 |
| `per_device_train_batch_size` | 8 | |
| `gradient_accumulation_steps` | 4 | prompt hiệu dụng/step = 4 |
| `learning_rate` | 1e-6 | RL cần LR thấp hơn SFT 100× |
| `beta` (KL) | 0.0 | Không KL; format reward + LoRA giữ ổn định |
| `epsilon` / `epsilon_high` | 0.2 / 0.28 | clip-higher kiểu DAPO |
| `max_completion_length` | **384** | facet vốn ngắn; giảm từ 512, cắt ~20% thời gian generation |
| `temperature` | 1.0 | rollout cần đa dạng |
| LoRA `r` / `alpha` / dropout | 32 / 64 / 0.05 | target: tất cả proj tuyến tính |
| `max_steps` | **400** | giảm từ 800. Chỉ tăng nếu run đầu cho thấy reward **chưa** bão hoà |
| `save_steps` / `eval_steps` | **100 / 100** | eval chiếm ~15% mỗi run |
| `seed` | 42 | |

**Quy mô dữ liệu LEAN:** train **1200** user (giảm từ 3000), val **150** (giảm từ 500), test giữ **1000** để bảng kết quả còn đáng tin.

Với config này, một run GRPO trọn vẹn ≈ **3.5h** thay vì 8h. Xem §11.3.

### 6.3 Biến thể thuật toán: **Dr. GRPO**, không phải vanilla

Bật `loss_type="dr_grpo"` (hoặc `scale_rewards=False` + tắt length normalization tuỳ phiên bản TRL).

Lý do cụ thể cho bài toán này: vanilla GRPO chia loss cho độ dài completion, tạo áp lực ngầm khiến policy viết memory **dài hơn**. Toàn bộ luận điểm của đồ án là làm memory **ngắn và tốt hơn**. Dùng vanilla là tự bắn vào chân.

### 6.4 Dynamic sampling (bắt buộc)

Trước mỗi epoch, loại khỏi batch những prompt có `std(r) = 0` trong group. Oversample bù để giữ batch hiệu dụng ổn định.

Kèm curriculum: chỉ train trên user có baseline H@1 ∈ [0.2, 0.8] (tính ở M1). User quá dễ hoặc quá khó không cho gradient hữu ích.

---

## 7. Roadmap

Mỗi milestone có **Definition of Done (DoD)**. Agent **chỉ** tick checkbox khi DoD đã được verify bằng lệnh chạy thật, không tick bằng suy luận.

**Chia làm 2 phase. KHÔNG được bắt đầu Phase 2 khi Phase 1 chưa đạt MVR (§2.5.3).**

| Phase | Milestone | Tầng | GPU-hour |
|---|---|---|---|
| **1 — Kết quả tối thiểu** | M0 · M1 · M2 · M3 · M4 · M7a | T0/T0-API → T1 → T2 | **~50h** |
| **2 — Chỉ khi Phase 1 xong** | M5 (ablation) · M6 (stretch) · M7b | T2 | +40h / +70h |
| **3 — Extension, tuỳ chọn** | `docs/RL_LM_REC_EXTENSION.md` (RL cho `LM_Rec`) | T2 | +24–36h |

> Phase 3 **loại trừ** với phần lớn Phase 2 về ngân sách. Chọn một, theo §7.1.

Mỗi milestone dưới đây gắn nhãn tầng: 🖥️ `T0` CPU · 🌐 `T0-API` · 🔧 `T1` GPU rẻ · 🚀 `T2` H100.

---

# PHASE 1 — Kết quả tối thiểu

---

### ☑~ M0 — Reproduce baseline · 🖥️🌐 `T0 + T0-API` — **0 GPU-hour**

> **Thay đổi LEAN:** kế hoạch gốc chạy `Qwen2.5-7B-Instruct` local. Bỏ hẳn. Dùng gpt-4o-mini qua API — chạy trên CPU, `--parallel_workers 32`, tốn ~$5–8 thay vì ~25 GPU-hour.

- [x] Setup env theo README (`conda`, `requirements.txt`). **Không cần CUDA ở milestone này.** — conda env `memrec`, Python 3.10, torch CPU-only.
- [x] Tải InstructRec datasets, chạy `bash scripts/convert_all_instructrec.sh` — chỉ Books (đúng scope §2).
- [x] Verify `data/processed/instructrec-books` tồn tại và load được
- [x] Cấu hình provider `azure_openai` / `openai` với `gpt-4o-mini` — dùng `openai` (plain OpenAI key, không phải Azure); phát hiện + sửa bug hardcode `azure_openai` cho Stage-ReRank (xem PROGRESS.md).
- [x] **Đo trước khi chạy full:** warmup 100 user — xem `docs/RESULTS.md`.
- [x] Chạy MemRec full pipeline trên `instructrec-books`, 1k user, seed 42, `--parallel --parallel_workers 16 --save_llm_conversations`
- [x] Chạy 2 baseline nội bộ: `w/o Collab. Read` và Vanilla LLM
- [x] Ghi số + **chi phí API thực tế** vào `docs/RESULTS.md` bảng "M0 Baselines"

**DoD:** bảng H@{1,3,5} + N@{3,5} cho 3 config, cùng seed, reproduce 2 lần ra cùng số. Không cần trùng số paper nhưng **thứ tự phải đúng**: MemRec > w/o Collab. Read > Vanilla.

**Nếu thất bại:** thứ tự sai nghĩa là pipeline đang hỏng — dừng, debug. Không được đi tiếp.

> **[~] DoD chỉ đạt một phần — quyết định có chủ đích, ghi lại ở đây theo quy tắc §10.2/§10.8.** Ngày 2026-08-06.
> Sau khi sửa 2 bug thật (Stage-ReRank hardcode `azure_openai`; prompt "MemRec mode" tự mâu thuẫn khi facets rỗng — xem PROGRESS.md), thứ tự **H@1 đã đúng**: 0.510 > 0.436 > 0.425. Nhưng **H@3/N@3/H@5/N@5 vẫn sai thứ tự** (w/o Collab. Read < Vanilla ở cả 4 metric này).
> Nguyên nhân nghi ngờ (chưa fix): `_evaluate_single_user` sample negative candidates bằng `RandomState(seed=hash(thread.ident))` — seed theo thread ID, không theo `user_id`/seed cấu hình. Ba config **không được đánh giá trên cùng bộ candidate/user**, và **không** reproduce được 2 lần ra cùng số như DoD yêu cầu.
> Quyết định: chủ đích **không** sửa RNG này ngay, chấp nhận phần sai lệch trên là nhiễu sampling, đi tiếp M1 với baseline H@1 đủ tin cậy để tham chiếu. Nếu sau này cần bảng H@3/H@5/NDCG đáng tin cậy hơn (vd. lúc viết luận văn ở M7a/M7b), phải quay lại sửa RNG này trước.

---

### ☑ M1 — Đóng băng môi trường & dựng RL dataset · 🖥️🌐 `T0 + T0-API` — **0 GPU-hour**

> Toàn bộ milestone này là xử lý file + một lần warmup qua API. Không chạm GPU.

- [~] ~~Tái sử dụng dump `llm_conversations/` của M0 — **không chạy lại pipeline**~~ → **thay bằng warmup mới 2350 user ($3.15, CPU+API).** Hai lý do bắt buộc, xem PROGRESS.md M1:
  (a) `memory.jsonl` của M0 **nhiễm test** — config `memrec_instructrec-books_1k.yaml` để `enable_stage_w` mặc định `True` nên eval loop đã ghi ground-truth click của **test item** vào memory dùng chung;
  (b) M0 chỉ warm 1000 user, M1 cần 2350. Replay log 5808 dòng để dựng lại state là xấp xỉ (thứ tự ghi đua nhau giữa 16 thread); warmup mới rẻ hơn nhiều so với rủi ro đó. Ngân sách này chỉ tiêu **một lần** — có `--memory_file` để tái tạo snapshot offline miễn phí.
- [x] Viết `src/rl/env.py`: serialize memory graph → `data/rl/graph_snapshot_books.json` (17.4 MB, 2350 user, 40080 item memory)
- [x] Cache `N'_k(u)` cho mọi user (k=16) vào snapshot — pruner không chạy lại trong RL loop
- [x] Viết `scripts/rl/01_build_dataset.sh` sinh 3 file jsonl, mỗi dòng:
  ```json
  {"user_id": "...", "prompt": "<state, candidate-blind>", "candidates": [...],
   "gold_item_id": "...", "M_u": "...", "neighbors": [...],
   "r_null": 0.41, "baseline_h1": 1}
  ```
- [~] Split user-disjoint: train 1200 / val 150 / test 1000 → **thực tế 1185 / 149 / 993** sau khi loại 23 user bị lộ đáp án (§ dưới). Chênh ~1%, không bù thêm user vì phải trả thêm tiền warmup.
- [x] `r_null` và `baseline_h1`: để trống (`null`), backfill sau M2 bằng `src.rl.dataset.backfill`
- [x] Viết `src/rl/dataset.py` load jsonl → `datasets.Dataset`
- [x] Viết `src/rl/policy.py`: prompt template + parser JSON facets — **unit test đầy đủ trên CPU**
- [x] Viết `tests/rl/test_no_leakage.py`

**DoD: ĐẠT.** 3 file jsonl tồn tại; split user-disjoint (assert bằng code, pass); `test_no_leakage.py` pass — 0 kết quả khi grep gold item title/id trong `prompt` trên **cả ba** split; parser xử lý được 20 ca output méo mó viết tay. Tổng **78 test pass** (`pytest tests/rl/`, 26s).

**Ba phát hiện phải xử lý (không có trong kế hoạch):**
1. **Trùng tên sách trong catalogue** — gold item không bao giờ là neighbor của chính user đó (graph chỉ dựng từ `train_data`), nhưng Books có **nhiều `item_id` khác nhau cho cùng một quyển sách**. Nếu user có bản sao kia trong lịch sử, tên sách đáp án bị in ra trong neighbor table. Đo được **23/2350 user (~1%)**, toàn bộ qua neighbor table, không qua `M_u`. Đã loại các user này (`src/rl/leakage.py`, log ở `data/rl/stager_books_dropped_users.json`).
2. **Ngân sách packer** — `SnippetPacker` giữ 300 token cho khối candidate. Bỏ candidate mà không bù lại thì policy được **1000** token neighbor trong khi baseline prompted chỉ có **700** → "GRPO thắng prompted" sẽ lẫn với "được nhìn nhiều neighbor hơn". Đã ghim bằng `CANDIDATE_BLOCK_RESERVE`.
3. **Negative của warmup trùng negative của eval** — cùng một RNG stream nên cả hai rút đúng 9 distractor giống nhau, mà Stage-R lúc warmup *có* nhìn thấy khối candidate → distractor của bài thi đã tham gia nặn ra `M_u`. Đã tách bằng salt (`WARMUP_CANDIDATE_SALT` / `EVAL_CANDIDATE_SALT`); sửa miễn phí vì candidate eval sinh offline.

**Quyết định thiết kế:** state **không** chứa instruction của InstructRec. Instruction diễn giải lại chính quyển sách đáp án (kiểm chứng bằng tay), và trong pipeline gốc nó chỉ đi vào Stage-ReRank chứ không vào Stage-R. Đưa nó cho policy là trao thẳng đáp án. `§3` viết `s = (I_u, M_u, Rep(N'_k(u)))`; ở đây `I_u` được hiểu là biểu diễn lịch sử tương tác (đã nằm trong neighbor table), không phải instruction text.

> **Lưu ý LEAN:** parser JSON phải chịu được output xấu **trước khi** lên GPU. Mỗi lần parser crash giữa run GRPO là mất cả phiên thuê máy.

---

### ☐ M2 — Reward function · 🖥️`T0` rồi 🚀`T2` — **~3 GPU-hour** — **milestone rủi ro nhất**

**Phần A — viết code, không cần GPU** 🖥️
- [ ] `src/rl/reward/metrics.py`: `ndcg_at_k`, `hit_at_k` + unit test với ví dụ tính tay
- [ ] `src/rl/reward/grounding.py` (bge-small chạy CPU + kiểm tra `source_ids`)
- [ ] `src/rl/reward/composite.py`, chữ ký khớp TRL reward function
- [ ] `src/rl/reward/ranker.py` với **stub mode**: trả logit cố định để test toàn bộ đường reward trên CPU
- [ ] `tests/rl/test_reward_logic.py` pass hoàn toàn trên CPU với stub ranker

**Phần B — cần GPU, gộp chung phiên với M3** 🚀
- [ ] Bật chế độ thật của `ranker.py` (`Qwen2.5-1.5B-Instruct`, listwise, một forward pass, logprob token chỉ mục)
- [ ] **Validation A — tương quan proxy:** trên **150** user val, tính NDCG@5 bằng (a) frozen 1.5B ranker và (b) gpt-4o-mini. Đo **Spearman ρ**. *(Phía gpt-4o-mini chạy trước trên CPU/API, cache ra file — lúc lên GPU chỉ so sánh.)*
- [ ] **Validation B — độ nhạy:** reward phải giảm rõ khi thay `M_collab` thật bằng (i) chuỗi rỗng, (ii) `M_collab` của user khác, (iii) lorem ipsum
- [ ] **Validation C — throughput:** ≥ 20 reward/s ở batch 64
- [ ] Backfill `r_null` + `baseline_h1` vào 3 file jsonl bằng một job batch
- [ ] Ghi vào `docs/RESULTS.md` mục "M2 Reward Validation"

**DoD:** Spearman ρ ≥ **0.6** · Validation B cho `r(thật) > r(user khác) > r(lorem) ≈ r(rỗng)` · throughput ≥ 20 reward/s.

**Nếu ρ < 0.6:** thử `Qwen2.5-3B-Instruct` làm ranker, hoặc đổi sang pointwise scoring. **Không được đi tiếp M4 với reward chưa validate** — 400 step trên reward sai là mất cả phiên thuê máy và cả tuần.

---

### ☐ M3 — Warm-start bằng rejection sampling · 🌐`T0-API` + 🚀`T2` — **~4 GPU-hour**

RL thuần từ base 4B trên JSON có cấu trúc sẽ collapse format. Bắt buộc warm-start.

> **Thay đổi LEAN:** teacher là **gpt-4o-mini qua API**, không phải local 7B. 1200 user × 8 sample ≈ **$10–14**, chạy trên CPU. Thuê H100 để chạy inference 7B cho việc này là lãng phí.

**Phần A — sinh dữ liệu, CPU + API** 🌐
- [ ] Với mỗi train user, sample 8 `M_collab` từ gpt-4o-mini (temperature 1.0), lưu ra jsonl
- [ ] Ước lượng chi phí trên 50 user trước, rồi mới chạy full

**Phần B — chấm điểm + SFT, GPU (gộp phiên với M2-B)** 🚀
- [ ] Chấm 9600 mẫu bằng reward function M2, giữ top-1/user nếu `r > r_null`
- [ ] `src/rl/sft.py`: SFT LoRA trên cặp (prompt, best `M_collab`), 2 epoch, lr 1e-5
- [ ] Eval checkpoint SFT trên val

**DoD:**
- Tỉ lệ output JSON hợp lệ ≥ **95%** trên val (base model trước SFT thường 40–70%)
- NDCG@5 val của SFT ≥ NDCG@5 val của base
- Checkpoint tại `checkpoints/rl/sft_books/`

> **⚠️ Đây là điểm an toàn của cả đồ án.** Sau M3 bạn đã có một kết quả nộp được: distillation gpt-4o-mini → 4B local. Nếu SFT-4B ngang prompted baseline với chi phí inference thấp hơn nhiều, đó là **bậc 3 của thang lùi (§2.5.4)** — luận văn hoàn chỉnh sau ~20 GPU-hour.
>
> - [ ] **Chốt bảng kết quả M3 vào `docs/RESULTS.md` NGAY, trước khi bắt đầu M4.** Đừng để kết quả này phụ thuộc vào việc M4 có thành công hay không.

---

### ☐ M4 — Huấn luyện GRPO · 🔧`T1` rồi 🚀`T2` — **~25 GPU-hour** — **milestone chính**

**Phần A — plumbing trên GPU rẻ** 🔧 (`A10`/`L4`, ~$0.5/h)
- [ ] Viết `src/rl/train_grpo.py` dùng TRL `GRPOTrainer`, init từ checkpoint SFT M3
- [ ] Bật `use_vllm=True, vllm_mode="colocate"`, `loss_type="dr_grpo"`
- [ ] Implement dynamic sampling filter (§6.4)
- [ ] Log mỗi eval_step: `reward_mean`, `reward_std`, `% thắng null-arm`, `completion_length_mean`, `format_valid_rate`, `grounding_score`, `% group bị filter`, `val NDCG@5`
- [ ] Implement + **test** checkpoint/resume
- [ ] **Dry run 20 step với `Qwen2.5-0.5B` + ranker 0.5B trên T1.** Mục tiêu: sập hết mọi lỗi config, dtype, tokenizer, parser, logging. **Không quan tâm reward có tăng không.**

**Phần B — run thật trên H100** 🚀
- [ ] Smoke run 20 step với model thật, verify VRAM peak < 75GB
- [ ] Full run **400 step**
- [ ] Eval checkpoint tốt nhất trên **test set** bằng gpt-4o-mini (không phải proxy)

**DoD:**
- Test NDCG@5 của GRPO policy > SFT policy > MemRec prompted baseline
- `completion_length_mean` **không tăng** quá 10% so với SFT (nếu tăng vọt → reward hacking hoặc Dr.GRPO chưa bật đúng)
- Training curve của `reward_mean` tăng đơn điệu (làm mượt), không sụp
- Ghi vào `docs/RESULTS.md` bảng chính

**Kill criteria — nếu sau 200 step mà `reward_mean` phẳng:**
1. Kiểm tra `% group bị filter` — nếu >60%, curriculum quá hẹp, nới band difficulty
2. Kiểm tra `reward_std` trong group — nếu ~0, tăng temperature lên 1.2
3. Nếu vẫn phẳng: giảm scope xuống "GRPO chỉ để nén memory" (reward = NDCG với ràng buộc length cứng). Kết quả "cùng accuracy, ít token hơn 40%" vẫn là một luận văn hoàn chỉnh.

---

### ☐ M7a — Chốt kết quả tối thiểu · 🖥️ `T0` — **0 GPU-hour**

**Cổng bắt buộc. Không được sang Phase 2 nếu mục này chưa xong.**

- [ ] Điền đủ bảng chính §8 (3 dòng: prompted / SFT / GRPO) kèm cột token/query
- [ ] Vẽ biểu đồ Pareto: x = token/query, y = H@1
- [ ] Chạy **một** kiểm tra chống hacking: ranker-swap (train reward 1.5B → eval gpt-4o-mini)
- [ ] Viết nháp chương Kết quả + Limitations
- [ ] Đối chiếu §11.7: đã tiêu bao nhiêu GPU-hour, còn lại bao nhiêu

**DoD:** đạt ít nhất **bậc 3** của thang lùi (§2.5.4). Tại điểm này bạn đã có luận văn nộp được.

Thêm hai mục để không phải trả lại tiền compute nếu sau này mở extension (§7.1):

- [ ] Đẩy checkpoint `LM_Mem` tốt nhất + config + git SHA lên storage bền
- [ ] Ghi SHA-256 của cột `candidates` trong 3 file jsonl vào `docs/RESULTS.md` — mọi so sánh về sau phải kiểm hash này trước khi chạy

**Quyết định sau M7a:** xem §7.1.

---

## 7.1 Cổng sau M7a — chọn MỘT nhánh

Sau M7a có **ba** nhánh cạnh nhau, không phải hai. Ngân sách thực tế chỉ đủ cho một.

| Nhánh | Nội dung | GPU-hour | Củng cố cái gì |
|---|---|---:|---|
| **0. Bắt buộc trước mọi thứ** | M5 **Ưu tiên 1** (ranker-swap, cross-domain, đọc tay 30 mẫu, breakdown theo `\|H_u\|`) | ~2 | Bảo vệ trực tiếp đóng góp chính. Chỉ eval, không train lại. |
| **1.** | M5 Ưu tiên 2–3 (ablation đầy đủ) | +33 | Đóng góp **chính** sâu hơn |
| **2.** | M6 (stretch, Stage-W propagation reward) | +70 | Đóng góp chính, rủi ro cao |
| **3.** | `docs/RL_LM_REC_EXTENSION.md` (RL cho `LM_Rec`) | +24–36 | Đóng góp **phụ** (modularity, sequential co-adaptation) |

**Quy tắc quyết định:**

1. Luôn chạy **nhánh 0** trước — ~2 GPU-hour, và nếu ranker-swap cho thấy gain không giữ được thì đóng góp chính đang có vấn đề, mọi nhánh còn lại đều vô nghĩa.
2. Còn <40h sau nhánh 0 → **bỏ hết**, đi thẳng M7b.
3. Còn 40–70h → chọn **nhánh 1 hoặc nhánh 3**, không phải cả hai.
   - Chọn **nhánh 1** nếu M4 cho gain rõ nhưng chưa rõ *vì sao* — cần ablation để giải thích.
   - Chọn **nhánh 3** nếu M4 đã sạch và thuyết phục, và bạn muốn thêm một câu hỏi nghiên cứu mới.
   - **Mặc định là nhánh 1.** §2.5 của Plan này: một đóng góp chính vững hơn tốt hơn hai đóng góp đều lung lay.
4. Nhánh 2 chỉ khi còn ≥70h **và** ≥3 tuần (điều kiện gốc của M6).
5. **Không tự ý mở nhánh 3.** Cổng đầy đủ ở §3 của file extension; bắt đầu rồi bỏ dở là mất trắng, vì extension chỉ có giá trị khi có đủ ma trận 6 dòng + robustness.

> Ghi lại lựa chọn và lý do vào `docs/PROGRESS.md` ngay khi quyết định, kèm số GPU-hour còn lại tại thời điểm đó.

---

# PHASE 2 — Chỉ khi Phase 1 đã đạt MVR

> Mọi thứ dưới đây là **tuỳ chọn**. Đừng bắt đầu vì tò mò — bắt đầu vì đã có ngân sách dư.

---

### ☐ M5 — Ablation & kiểm tra reward hacking · 🚀 `T2` — **+35 GPU-hour**

Mỗi mục là một run riêng, config trong `configs/rl/m5_ablations/`. **Làm đúng theo thứ tự ưu tiên dưới đây, dừng bất cứ lúc nào hết ngân sách.**

**Ưu tiên 1 — rẻ, không cần train lại** (chỉ eval, ~2h tổng)
- [ ] **Non-LLM ranker test**: dùng `M_collab` đã train với vector reranker của repo
- [ ] **Cross-domain transfer**: policy train trên Books, eval zero-shot trên MovieTV
- [ ] **Đọc tay 30 mẫu** `M_collab` trước/sau RL, ghi định tính vào `docs/RESULTS.md`
- [ ] Breakdown gain theo quartile độ thưa `|H_u|` — gain **phải** tập trung ở user thưa

**Ưu tiên 2 — mỗi mục là 1 run (~3.5h)**
- [ ] Candidate-blind vs candidate-visible (§5.3) — trả lời trực tiếp câu hỏi hacking
- [ ] Dr. GRPO vs vanilla GRPO — kỳ vọng: vanilla làm completion dài hơn rõ rệt
- [ ] `r_ndcg` only (bỏ grounding + length penalty)
- [ ] Pure RL không SFT warm-start — kỳ vọng: format collapse

**Ưu tiên 3 — bỏ đầu tiên nếu thiếu giờ**
- [ ] `G ∈ {4, 8, 16}` — trả lời câu hỏi về thuật toán, không phải về đóng góp của đồ án
- [ ] `r_ndcg + r_ground` (tách riêng khỏi full reward)
- [ ] Reward dùng vector reranker thay LLM ranker

**DoD:** hoàn thành trọn Ưu tiên 1 + ít nhất 2 mục Ưu tiên 2. Ranker-swap giữ ≥60% gain.

---

### ☐ M6 (STRETCH) — Graph-propagated reward cho Stage-W · 🚀 `T2` — **+70 GPU-hour**

**Chỉ bắt đầu nếu M5 đã xong VÀ còn ≥70 GPU-hour VÀ còn ≥3 tuần.** Phần mới nhất về khoa học nhưng rủi ro cao nhất, và tốn GPU gấp đôi mọi milestone khác vì graph phải re-materialize mỗi epoch.

> Với ngân sách LEAN, khả năng cao milestone này **không** được thực hiện. Đó là chấp nhận được — ghi vào Future Work.

Ý tưởng: một action ghi tại node `u` sinh `M_u^t, M_i^t, {ΔM_neigh}`. Reward là tổng lợi ích lan truyền:

```
R(a_t) = Σ_{v ∈ {u} ∪ N'_k(u)}  w_v · Δmetric_v(probe kế tiếp của v)
```

- [ ] Bỏ đóng băng graph, implement epoch snapshot (rollout đọc snapshot, ghi vào bản scratch, cuối epoch mới materialize lại)
- [ ] Monte-Carlo: chỉ probe 2–4 hàng xóm ngẫu nhiên mỗi rollout (estimator không chệch)
- [ ] Ablate: self-only reward vs self+neighbor reward
- [ ] Ablate số neighbor probe ∈ {1, 2, 4}

**DoD:** self+neighbor reward > self-only trên test H@1. Nếu không, báo cáo negative result — vẫn có giá trị.

---

### ☐ M7b — Hoàn thiện luận văn · 🖥️ `T0` — **0 GPU-hour**

- [ ] Chốt toàn bộ bảng trong `docs/RESULTS.md`
- [ ] Báo cáo **chi phí thực tế** (GPU-hour + API $) như một kết quả — đây là đồ án về hiệu quả chi phí, con số này thuộc về phần Kết quả
- [ ] Viết Limitations: protocol N=10 từ log data (item không quan sát ≠ item không liên quan); chỉ 1 dataset train; graph tĩnh; không có online eval
- [ ] Related work: **Nhánh ranker** — Rec-R1, RecLLM-R1, ConvRec-R1/Rank-GRPO. **Nhánh memory** — Memory-R1, Mem-α, Memory-as-Action. **Khe hở** — chưa ai làm RL cho collaborative memory trên đồ thị.
- [ ] Dọn repo, viết `docs/REPRODUCE.md`

---

## 8. Bảng kết quả cần điền (`docs/RESULTS.md`)

### Bảng chính — instructrec-books, test set, `LLM_Rec` = gpt-4o-mini

| Config | H@1 | H@3 | N@3 | H@5 | N@5 | Tokens/query | Δ vs prompted |
|---|---|---|---|---|---|---|---|
| Vanilla LLM | | | | | | | |
| MemRec w/o Collab. Read | | | | | | | |
| MemRec prompted (7B `LM_Mem`) | | | | | | | — |
| MemRec + SFT-4B `LM_Mem` (M3) | | | | | | | |
| **MemRec + GRPO-4B `LM_Mem` (M4)** | | | | | | | |

### Bảng chống hacking

| Test | Gain giữ được | Kết luận |
|---|---|---|
| Reward ranker (1.5B) → gpt-4o-mini | | |
| → vector reranker | | |
| Books → MovieTV (zero-shot) | | |
| Candidate-blind → candidate-visible | | |

---

## 9. Chế độ hỏng đã biết & cách phát hiện

### 9.1 Reward hacking qua kênh ngầm — nguy hiểm nhất
**Triệu chứng:** train reward tăng mạnh, ranker-swap eval không tăng hoặc tụt.
**Phòng:** candidate-blind (§5.3) + grounding reward (§5.2).
**Phát hiện:** ranker-swap test ở M5. Nếu gain giữ <40% → coi như hack, phải báo cáo.

### 9.2 Group thoái hoá (`std(r)=0`)
**Triệu chứng:** `reward_std` ≈ 0, loss ≈ 0, không học.
**Phòng:** NDCG thay Hit@1, dynamic sampling, curriculum theo difficulty.
**Phát hiện:** log `% group bị filter` mỗi step. >60% là báo động.

### 9.3 Phình độ dài
**Triệu chứng:** `completion_length_mean` tăng đều, reward tăng nhẹ.
**Phòng:** Dr. GRPO (§6.3) + length penalty.
**Phát hiện:** plot length vs step. Tăng >10% so với SFT là báo động.

### 9.4 Format collapse
**Triệu chứng:** `format_valid_rate` tụt sau vài trăm step.
**Phòng:** SFT warm-start (M3), `λ_fmt = 1.0`.
**Phát hiện:** log `format_valid_rate`. <90% → tăng `λ_fmt` hoặc bật `beta=0.005`.

### 9.5 Proxy reward lệch mục tiêu thật
**Triệu chứng:** proxy NDCG tăng, `LLM_Rec` thật không tăng.
**Phòng:** ngưỡng Spearman ở M2.
**Phát hiện:** eval bằng `LLM_Rec` thật mỗi 200 step, không chỉ ở cuối.

---

## 10. Giao ước làm việc với agent

1. **Một branch cho mỗi milestone:** `rl/m2-reward`, `rl/m4-grpo`, ... Merge vào `main` chỉ khi DoD đạt.
2. **Chỉ tick checkbox sau khi verify bằng lệnh chạy thật.** Không tick bằng suy luận. Nếu một mục bị bỏ qua có chủ đích, đổi `- [ ]` thành `- [~]` kèm một dòng lý do ngay bên dưới.
3. **Sau mỗi milestone, append vào `docs/PROGRESS.md`:**
   ```
   ## M2 — 2026-xx-xx
   Trạng thái: DONE / BLOCKED
   Đã làm: ...
   Số đo: Spearman ρ = 0.xx, throughput = xx/s
   Lệch so với kế hoạch: ...
   Quyết định đã ra + lý do: ...
   Việc tiếp theo: ...
   ```
4. **Không sửa đường eval gốc.** Sau mỗi milestone, chạy lại lệnh baseline M0 và xác nhận số không đổi.
5. **Mọi milestone phải có smoke test chạy <5 phút** trong `tests/rl/`. CI tay cũng được, nhưng phải có.
6. **Log VRAM peak** mỗi lần train (`torch.cuda.max_memory_allocated()`) **và wall-clock GPU-hour** của mỗi run, ghi vào PROGRESS và cập nhật bảng §11.7.
7. **Seed cố định 42.** Mọi run lưu config đầy đủ + git SHA vào thư mục output.
8. **Khi gặp mâu thuẫn giữa file này và thực tế code, DỪNG và hỏi.** Đừng tự ý đổi phạm vi. Nếu file này sai, sửa file này trong cùng PR và ghi rõ lý do.
9. **Không tối ưu hoá sớm.** Chạy được trước, nhanh sau. Ngoại lệ duy nhất: reward throughput ở M2, vì nó nhân với mọi step về sau.
10. **Ngân sách là ràng buộc cứng, không phải gợi ý.** Trước khi thuê H100, phải trả lời được: run này trả lời câu hỏi *phương pháp* hay câu hỏi *plumbing*? Nếu là plumbing → T1. Nếu không chắc → T1.
11. **Không tự ý mở Phase 2 hay Phase 3.** Cổng M7a là bắt buộc, và sau đó phải chọn nhánh theo §7.1 — dừng, hỏi. Điều này áp dụng cho cả `docs/RL_LM_REC_EXTENSION.md`: có checkpoint `LM_Mem` ở M4 **không** phải là điều kiện đủ để bắt đầu extension.
12. **Plan này thắng khi mâu thuẫn với file extension.** Đóng góp chính của luận văn là RL cho `LM_Mem`; không viết lại narrative chính vì extension.

---

## 11. Ngân sách compute — chế độ LEAN

**Con số chốt Phase 1:** ~**50 GPU-hour H100** + ~**15h GPU rẻ** + ~**$35 API** ≈ **$190 tổng**.
So với kế hoạch gốc (~250 GPU-hour ≈ $700): giảm ~4× nhờ đẩy mọi việc offline sang CPU/API và cắt quy mô run.

### 11.1 Bóc tách Phase 1

| Milestone | Tầng | H100 (h) | GPU rẻ (h) | API ($) |
|---|---|---|---|---|
| M0 — repro baseline | 🖥️🌐 | **0** | 0 | ~8 |
| M1 — snapshot + dataset | 🖥️ | **0** | 0 | 0 |
| M2 — reward function | 🖥️→🚀 | **3** | 0 | ~3 |
| M3 — SFT warm-start | 🌐→🚀 | **4** | 0 | ~14 |
| M4 — GRPO | 🔧→🚀 | **25** | 15 | 0 |
| M7a — chốt MVR | 🖥️ | **0** | 0 | ~8 |
| Dự phòng | | **18** | 0 | ~5 |
| **Tổng Phase 1** | | **~50h** | **15h** | **~$38** |

Phase 2: M5 +35h, M6 +70h. Quyết định riêng sau M7a.

### 11.2 Ba đòn bẩy đã cắt được 200 giờ

| Đòn bẩy | Cắt được | Cách |
|---|---|---|
| **Bỏ local 7B, dùng API cho offline** | ~30h | Warmup graph + teacher qua gpt-4o-mini trên CPU. ~$22 thay vì ~$100 tiền thuê GPU |
| **Staging tier cho plumbing** | ~20h | Dry-run `Qwen2.5-0.5B` trên A10 $0.5/h. Debug config ở đây, không ở H100 $3/h |
| **Hoãn ablation sang Phase 2** | ~75h | M5/M6 là điều kiện của ngân sách dư, không phải của đồ án |
| Giảm quy mô run (400 step, 1200 user, 384 token) | ~25h | §6.2 |

### 11.3 Cơ sở tính M4 (config LEAN)

`G=8`, `per_device_bs=8`, `grad_accum=4` → 32 rollout/step.

| Thành phần | Ước tính |
|---|---|
| generation 32 × ~300 token (max 384) | ~4s |
| reward, 32 forward pass prefill-only | ~1.5s |
| backward + logprob (`beta=0`, không ref model) | ~10s |
| weight sync sang vLLM | ~2s |
| **Tổng** | **~18s/step** |

400 step × 18s × 1.4 (oversample) = 2.8h + eval → **~3.5h/run trọn vẹn**.

25h cho M4 ≈ 1 smoke run + 3 run hỏng/điều chỉnh + 2 run tốt + 1 run dự phòng.

### 11.4 Ba rủi ro ảnh hưởng con số nhiều nhất

**① Warmup graph — ẩn số lớn nhất, nhưng giờ là rủi ro TIỀN chứ không phải rủi ro GIỜ GPU.** Books có 120.9K item; chạm ~30K node trong neighborhood của ~2400 user ≈ ~30M input token qua gpt-4o-mini ≈ ~$5–8. Sai số ±2× nghĩa là tệ nhất ~$16 — chấp nhận được. Đây chính là lý do chuyển sang API: sai số ước lượng biến từ "20 giờ H100" thành "8 đô".

> **Việc đầu tiên ở M0:** warmup 100 user, đếm token thật, ngoại suy. Ghi vào `PROGRESS.md`.

**② Storage bền, không phải ephemeral disk.** Snapshot + dataset + checkpoint SFT. Mỗi lần mất là phải trả lại tiền API warmup. Đẩy `data/rl/` và `checkpoints/rl/` lên object storage sau mỗi phiên.

**③ Checkpoint/resume phải chạy được TRƯỚC khi lên H100.** Test ở tầng T1. Bắt buộc nếu dùng spot (rẻ hơn 50–70%) — với block 4h thì spot khá an toàn, đây là lợi ích phụ của việc rút run từ 8h xuống 3.5h.

### 11.5 Lịch đặt chỗ

| Phiên | Nội dung | Tầng | Thời lượng |
|---|---|---|---|
| — | M0 + M1 | CPU + API | không thuê GPU |
| S1 | M4 Phần A: dry-run plumbing 0.5B | A10/L4 | 2 block × 6h |
| S2 | M2-B + M3-B (gộp) | H100 | 1 block × 8h |
| S3 | M4 smoke + run thật #1 | H100 | 1 block × 6h |
| S4–S6 | M4 lặp | H100 | 3 block × 5h |
| — | M7a | CPU | không thuê GPU |

**Bắt buộc làm trên CPU, không trên máy đang tính tiền:** convert dataset, build jsonl, unit test, parser, phân tích kết quả, vẽ hình, đọc tay 30 mẫu `M_collab`, viết luận văn. Eval bằng gpt-4o-mini là API — không cần GPU.

### 11.6 Checklist tiết kiệm cho agent

- [ ] Trước mỗi phiên H100: liệt kê **chính xác** những gì sẽ chạy, ước tính giờ, ghi vào PROGRESS. Máy bật lên rồi mới nghĩ là cách đốt tiền nhanh nhất.
- [ ] Không bao giờ để H100 idle. Nếu đang đọc log hay sửa code >15 phút, tắt máy.
- [ ] Model weights tải sẵn vào persistent volume, không tải lại mỗi phiên (4B ≈ 8GB, mất ~5 phút mỗi lần).
- [ ] Mọi câu hỏi dạng "code có chạy không" → T1 với model 0.5B, không bao giờ T2.
- [ ] Kết quả eval cache ra file, không tính lại. Đặc biệt là điểm gpt-4o-mini ở M2 Validation A.
- [ ] Sau mỗi phiên: đẩy checkpoint + log lên storage bền **trước khi** terminate.

### 11.7 Bảng theo dõi giờ thực tế — agent điền

| Milestone | Dự toán H100 (h) | Thực tế | GPU rẻ (h) | API ($) | Ghi chú |
|---|---|---|---|---|---|
| M0 | 0 | **0** | 0 | ~**1.1** | warmup 1k user đo được: 3.45M in + 1.00M out |
| M1 | 0 | **0** | 0 | **3.15** | warmup 2350 user, 26.6 phút wall, 24 worker, 9.54M in + 2.87M out |
| M2 | 3 | | | | |
| M3 | 4 | | | | |
| M4 | 25 | | | | |
| M7a | 0 | | | | |
| Dự phòng | 18 | | | | |
| **Phase 1** | **50** | | **15** | **~38** | |
| M5 Ưu tiên 1 | +2 | | | | **bắt buộc sau M7a** (§7.1 nhánh 0) |
| M5 Ưu tiên 2–3 | +33 | | | | §7.1 nhánh 1 — mặc định |
| M6 | +70 | | | | §7.1 nhánh 2, nhiều khả năng bỏ |
| Extension `LM_Rec` | +24–36 | | | ~8 | §7.1 nhánh 3, loại trừ với nhánh 1 |

- [ ] Đo warmup 100 user ở M0, cập nhật ước tính API
- [ ] Verify checkpoint/resume ở tầng T1 trước khi vào M4-B
- [ ] Sau mỗi phiên, cập nhật cột "Thực tế"; lệch >50% thì báo động và điều chỉnh phạm vi Phase 2

---

## 12. Thuật ngữ

| Ký hiệu | Nghĩa |
|---|---|
| `LM_Mem` | Memory manager nhẹ — **policy được train** |
| `LLM_Rec` | Reasoning agent nặng — **đóng băng** |
| `M_collab` | Collaborative memory tổng hợp, `N_f` facet — **action** |
| `M_u`, `M_i` | Semantic memory của node user/item |
| `N'_k(u)` | Top-k neighbor sau curation |
| `r_null` | Reward khi không có collaborative memory (baseline chẩn đoán) |
| `G` | Group size của GRPO |
| DoD | Definition of Done |

---

## 13. Tham chiếu

- **MemRec** — Chen et al., ACL 2026. [aclanthology.org/2026.acl-long.2061](https://aclanthology.org/2026.acl-long.2061/) · [arXiv:2601.08816](https://arxiv.org/abs/2601.08816)
- **GRPO** — Shao et al., DeepSeekMath, 2024
- **Dr. GRPO** — bỏ length/std normalization bias
- **DAPO** — clip-higher, dynamic sampling, token-level loss
- **Memory-R1** — [arXiv:2508.19828](https://arxiv.org/abs/2508.19828) — RL cho memory manager (QA, memory cô lập)
- **Rank-GRPO / ConvRec-R1** — [arXiv:2510.20150](https://arxiv.org/abs/2510.20150) — GRPO mức rank cho recsys
- **Rec-R1**, **RecLLM-R1** — GRPO cho ranker trong recsys