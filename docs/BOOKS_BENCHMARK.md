# InstructRec Books — benchmark và kết quả full MemRec

**Cập nhật 2026-09-30.** Đây là nguồn chuẩn cho thesis về InstructRec Books. Số liệu dưới đây không thay thế per-user predictions, manifests và hashes trong `results/full_memrec_books_baselines/`; hồ sơ rải rác trước đây đã gộp vào đây.

## 1. Task và ranh giới

MemRec trong paper xếp hạng **candidate set có sẵn**, không retrieve toàn catalog. InstructRec Books gốc có 7.377 user, mỗi user đúng 10 candidate riêng biệt và một target; ordered candidate/target manifest SHA-256 `a13f7435b5f788503bb1f66608fd072bbc9b48d252bbdd0284126d79c2f87cb6`. Đã audit 7.377/7.377 list. Với 10 candidate và một positive, Hit@10 là tất yếu (trừ malformed); NDCG@5 là primary, Hit@1 và Hit@5 là secondary.

Fork trước đây bỏ `ranked_lists` gốc để lấy random negatives và có thể dùng `eval_feedback=gt`, gây sai lệch task/leakage. Run cũ 1.000 user NDCG@5 `0.665572` không thể dùng làm baseline sạch. Benchmark mới dùng original fixed lists, graph từ train history, warm-up Stage-W từ lịch sử trước test, **không ghi test label vào shared memory**. `.inter` chỉ có vị trí trong từng user, không có global clock. Core upstream được pin ở commit `58d9031ed91c623b8034d2bf04f39aa937424c33`; fork giữ graph/pruner/packer và full Stage-R/ReRank/W nhưng đổi provider, candidate/feedback/cohort guards và logging. Đây là **upstream-aligned architecture với self-host LLM**, không phải exact GPT-4o-mini paper replication.

**Fidelity caveat mới (code audit 2026-09-29):** `src/memory/pruner_llm_rules.py` ở cả fork và pinned upstream có `metadata_overlap=0.5` placeholder cho item, `memory_sim=0.0` và recency “days” suy từ điểm 0–1; Books rule boost metadata chỉ khi `>0.6`. `src/memory/packer.py` dùng token estimate và dừng greedy loop ở snippet đầu không vừa. Vì vậy vài tín hiệu curation mô tả trong paper có thể chưa hoạt động như kỳ vọng ở implementation này. Chưa biết author run dùng đúng code path này hay preprocessing khác; không suy ra paper result sai. Trước luận điểm method mới vượt MemRec, cần report nguyên trạng upstream **và** một đối chứng feature-complete có cách tính công khai/frozen, tách fidelity fix khỏi đóng góp học thuật và không tự nhận exact paper replication. Xem [thiết kế CM-IRank](CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md).

Split user-disjoint trên original test rows: development 2.000 user, SHA-256 sorted IDs `ee83fe12d61df57c5678337458c8a2b556d5bb01c2d39542a52832fb4118f2b0`; held-out 5.377 user, SHA-256 `b7751e0ae8f4f23709736ed9b19c457e570aa3edf2a6e0db19c6da938c4af8be`. Seed split `full-memrec-books-v1-20260925` ở `src/data/books_protocol.py`. 1.085 user đã lộ outcome lịch sử (sorted-ID SHA-256 `d94c17508805b64c709ec625f832e43139b714de59c2aa76fce797316aa84171`) đều nằm trong dev; held-out có 0 user overlap. Chạy full 7.377 user kiểu paper là mục tiêu thứ cấp sau khi khóa method, không phải kết quả đã có.

## 2. Đối chứng và cohort 700/200 đã khóa

SASRec được chọn tự động trong grid preregistered trên **dev 2.000**: dimension `64/128` × max sequence `50/100`, 2 blocks/heads, dropout `0.2`, Adam LR `0.001`, batch `512`, tối đa 50 epoch, patience 5, seed `20260925`. Kết quả dev NDCG@5 theo d64_s50/d64_s100/d128_s50/d128_s100 là `0.303372/0.295173/0.321127/0.302800`; d128_s50, epoch 1 được chọn tự động. Trên đúng **200 user** bên dưới, SASRec NDCG@5 `0.343320`. Trên 2.000 dev: NDCG@5 `0.321127`, Hit@1 `0.1255`, Hit@5 `0.5275`. Checkpoint SHA-256 `0c59d0c0563816e729729574d3675ff83d04e9fffd594a35b67e0152522eb8fd`; prediction 2.000-row SHA-256 `f4827100061c3bccb9a27d91813cc4fd365b8cbdff47b2dcf330e7fb87a20c69`. Run ID `sasrec-books-dev-v3-hnv`, artifacts dưới `results/full_memrec_books_baselines/`; chưa chấm held-out và không retrain checkpoint theo held-out.

Vì chi phí, full-MemRec thực nghiệm hiện dùng **700 warm-up/200 evaluation dev users**. Eval là 200 ID đầu của dev 2.000 (sorted-ID SHA-256 `5395ee7775d9e5d0add11ed386715c34a002138c282f4f910f775ea90871a837`); warm-up ID SHA-256 `e8b9e14032df6e2ea8d0389c62de13d0e8fb087edac127aacbfba8b4062050c9`. Warm-up lấy 700 ID thấp nhất, thay ID non-eval cao nếu cần để chứa đủ 200 eval user, rồi duyệt sorted-ID. Graph vẫn xây từ train interactions được phép của toàn bộ users; chỉ memory warm-up downsample. Fixed original 10 candidates, seed 42, `k=16`, `τ=1800`, Stage-R/ReRank/Stage-W warm-up, không có eval-label Stage-W. Model `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, revision `5a5a776300a41aaa681dd7ff0106608ef2bc90db`, vLLM `0.10.2`, một H100, tensor parallel 1, GPU-memory fraction `0.60`, temperature 0. Cấu hình giới hạn 2.500 logical calls, 2.750 physical reservations và timeout 570 phút. LLM 30-user smoke đã qua trước full run.

| Arm, cùng 200 user/candidates | NDCG@5 | Hit@1 | Hit@5 | Malformed | Diễn giải |
|---|---:|---:|---:|---:|---|
| SASRec | 0.343320 | — | — | — | Trained sequential control; cùng evaluation cohort, khác training/warm-up budget |
| **Full MemRec baseline, 700 warm-up** | **0.747918** | **0.595** | **0.890** | **2/200** | Full Stage-R/ReRank/W architecture; reduced-memory dev |
| Full MemRec + directed one-step evidence **trong Stage-R** | 0.742968 | 0.575 | 0.900 | 1/200 | Actual full-agent method; primary improvement gate thất bại |

Baseline NDCG@5 chính xác `0.7479179811`; Stage-R method `0.7429683083`. Method − baseline `−0.0049496728`, paired bootstrap CI95% `[−0.0329741157,+0.0233556961]` (10.000 resamples, seed `20260928`). Target improved/worsened/unchanged: `19/23/158`; 163 full candidate permutations thay đổi. CI cắt 0 nên không khẳng định method gây hại chắc chắn, nhưng **không có bằng chứng cải thiện full MemRec**. Các malformed output đều giữ trong mẫu số, tính như miss. Baseline dùng 2.052 physical attempts; Stage-R method 1.504, cả hai hoàn tất và GPU đã nhả. Baseline artifacts `results/full_memrec_books_baselines/books-memrec-llm-dev700-v1-hnv/`; method artifacts `results/full_memrec_books_baselines/books-memrec-transition-stage-r-dev700-v1-hnv/`; method prediction SHA-256 `ec61dd656303624cfa34fa94875f08c44b34169cd2786f2f43c6606cdfc8dcd4`. Intervention xây 185.628 cạnh directed từ cặp `train_data` liên tiếp của 7.377 user; tối đa sáu train seeds gần nhất. Stage-R pruner giữ bốn user và sáu item neighbor đầu theo baseline, dùng slot dư cho successor mới, vẫn `k≤16`, `τ≤1800`. Không dùng candidate/gold để chọn successor; không đổi Stage-W, LLM hay prompt. 176/200 user có successor mới nhưng chỉ 3/200 candidate lists overlap; đây là coverage audit, không phải causal explanation.

### Thử nghiệm chuyển giao phụ: post-ranking residual

Đây **không phải** thay đổi Stage-R/Stage-W: cộng graph score sau khi full MemRec đã trả ranking. One-step primary NDCG@5 `0.748919`, Δ `+0.001001`, CI95% `[−0.005536,+0.008540]`; PPR secondary `0.733789`, Δ `−0.014129`, CI95% `[−0.036241,+0.009521]`. Chỉ 8/200 candidate lists có one-step support; 1 user tốt lên, 1 xấu đi. PPR có support ở 53/200 nhưng cải thiện/xấu đi `6/20`; không có LLM/GPU call mới. Kết luận: **không xác nhận gain**, không được gộp với Stage-R method hay dùng để tune trên 200 label. Artifacts dưới `results/full_memrec_books_baselines/books-memrec-transition-transfer-v2-hnv/`; `full_predictions.jsonl` SHA-256 `7e92ea4896da40023c4e84526e7cf555b7fbdc0a6c199cd4d4f8c8481cc524bd`, result JSON SHA-256 `4c57c5a9218395c72ac2532b152ac26343050f3841d65481ed61428e42d6f2d3`.

Run full-dev all-user warm-up cũ đã dừng theo yêu cầu chi phí ở 129/7.377 warm-up, **0/2.000 scored users**; không có metric full-dev. Full 700/200 ở trên là run riêng, không được gọi là paper-style full-data. Số paper Books NDCG@5 MemRec `0.6601`, SASRec `0.2824` dùng model/protocol khác; không so trực tiếp với bảng này.

## 3. Tiến độ, giới hạn và handoff

| Hạng mục | Trạng thái |
|---|---|
| Original candidates, split, leakage guards và baseline smoke | Hoàn tất |
| SASRec trên dev 2.000 / paired 200 subset | Hoàn tất; chỉ là dev |
| Full MemRec self-host 700/200 | Hoàn tất; exploratory reduced-memory |
| One-step residual transfer và Stage-R integration | Hoàn tất; không đạt improvement gate |
| Full 7.377 warm-up / 2.000 dev full-MemRec | Dừng sớm do chi phí; không có score |
| Full-MemRec headroom/ablation có preregistration | Chưa chạy |
| Method mới và held-out 5.377 user | Chưa chạy; held-out niêm phong |

Các 200 eval label đã được xem; không thủ công tune quota, depth, prompt, alpha hay chọn seed theo chúng. Hướng tiếp theo là [CM-IRank](CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md): nghiên cứu Stage-ReRank theo policy loại dần trên memory cố định; phương pháp **chưa có kết quả thực nghiệm**. Với mọi experiment: smoke vài chục mẫu trước full run, lưu per-user predictions/failures/hashes, trả GPU khi kết thúc.

## 4. Tái lập và artifact ledger

**Data contract.** `data/processed/instructrec-books/booksAll_recagent.pkl` chứa candidate/target; `.inter` chứa chuỗi interaction; `.instruction` và `.meta` cung cấp prompt/item metadata. SHA-256 của bốn input theo thứ tự `.pkl/.inter/.instruction/.meta` là:

```text
5bdc05f44db5f7c4fa2ca05db70bdaf9adf31c68fa2d7425695e83f63cbb7dbc
891d4007af3e63eef40493520218f644478250ec35d4443faa1d2b96384efc24
44a3ae1b3ab49a4328eebd50ed9e5c1e7bbe2819e454a44c6d50662477916150
69b46870c62e207fca11b6d695d203efef5ee1925f33428f43c1e6eb6ccec88c
```

Metadata có 190.756 item IDs; `.inter` thấy 120.925 interacted items. Original target position gần đều ở cả 10 vị trí, không có shortcut “target luôn đứng đầu”; chưa xác minh candidate construction khớp *chính xác* paper. Các lệnh kiểm tra CPU, không cần GPU:

```bash
python scripts/audit_instructrec_books_candidates.py
python scripts/audit_instructrec_books_candidates.py --full
python scripts/smoke_full_memrec_cpu.py --users 30
```

`smoke_full_memrec_cpu.py` dùng fake schema-shaped LLM, chỉ kiểm wiring; score của nó **không có giá trị ranking**. Full benchmark config là `configs/memrec_instructrec-books_full_benchmark.yaml`; candidate list được trainer dùng thật khi bật `--use_pregenerated_candidates`. Fixed-list run chạy serial, `eval_feedback: none`, count malformed/failure là miss, lưu JSONL per user; không dùng `RecDataset.valid_data` có negative distribution khác. Trước full GPU run phải qua [quy tắc H100](../internal_docs/H100_RESOURCE_RULES.md) và real-LLM smoke 20–30 user, đối chiếu model/revision/config/cohort/prediction hashes, rồi mới promote. Không tự chạy lại 7.377/2.000 full-dev; dry-run budget gốc là 26.131 logical và cap 28.745 physical requests, run cũ đã dừng ở 129 warm-up/0 scored.

| Run/artifact | Dấu vết cần giữ | Kiểm định |
|---|---|---|
| `sasrec-books-dev-v3-hnv` | checkpoint/predictions ở `results/full_memrec_books_baselines/` | Chọn tự động trên dev; same candidate manifest; GPU đã nhả |
| `books-memrec-llm-dev700-v1-hnv` | `completion.json`, prediction JSONL, metrics, journal, GPU snapshots | 700 warm-up/200 scored; 8/8 completion hashes match; local `scripts/check_books_memrec_full_dev.py --protocol dev700` pass; 2.052 physical attempts; GPU về baseline |
| `books-memrec-transition-stage-r-dev700-v1-hnv` | cùng loại artifact, source commit `d0026f5` | 8/8 hashes match; checker với `--expected-pruner-mode transition_one_step` pass; 1.504 attempts; GPU về baseline |
| `books-memrec-transition-transfer-v2-hnv` | `full_predictions.jsonl`, `full_result.json` | 200/200 predictions, baseline failures giữ lại, zero LLM/GPU |

Baseline 700/200 chấm trên **tất cả 200** user: Hit@1/3/5/10 `0.595/0.770/0.890/0.990` và NDCG@1/3/5/10 `0.595/0.6989/0.7479/0.7803`. Hit@10 dưới 1 chỉ vì 2 malformed rankings bị tính miss. Run baseline source commit `bda9ed2`, vLLM smoke đầu tiên `books-memrec-llm-smoke-v2-hnv` có 30/30 ranking hợp lệ, 150 physical calls, 0 failures; smoke `smoke700` sau đó được exact-input cache hit 150/150 nên không phát request mới. Stage-R method smoke có 30/30 hợp lệ, 88 physical requests. Smoke metrics không dùng chọn/tune method. Mọi luận điểm thesis phải lấy artifact/prediction gốc để recompute paired statistics, không suy từ bảng aggregate một mình.
