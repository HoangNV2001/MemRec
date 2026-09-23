# Temporal Transition-PPR for next-item recommendation

Đây là tài liệu canonical của hướng nghiên cứu hiện tại. Nó mô tả phương pháp,
protocol, kết quả discovery P7-v2, internal replication 200 event và frozen
cross-domain transfer trên MovieLens 32M; các hướng đã đóng chỉ còn bản tóm tắt trong
[ARCHIVED_DIRECTIONS.md](ARCHIVED_DIRECTIONS.md). Execution record đầy đủ nằm
trong [REPLICATION_PLAN.md](REPLICATION_PLAN.md).

## 1. Câu hỏi nghiên cứu

Các graph user–item đồng sở thích trước đây tăng reachability khi thêm hop nhưng
đồng thời khuếch đại hub và negative evidence. Transition-PPR đổi semantics của
graph: thay vì hỏi “ai cũng thích item nào”, nó hỏi “sau item A, người dùng
thường chuyển sang item nào”. Multi-hop khi đó mô hình hóa chuỗi chuyển tiếp xa
hơn nhưng vẫn bám objective dự đoán item kế tiếp.

Primary hypothesis đã khóa trước fresh test:

> Residual fusion giữa frozen local ranker và directed temporal transition-PPR
> cải thiện ít nhất +0,02 NDCG@5, với paired bootstrap 95% CI lower > 0.

## 2. Dữ liệu và temporal contract

Nguồn hiện tại là Kaggle Amazon Books:

- `data/amazon_books/raw/Books_rating.csv`: review, user, item, score và Unix
  `review/time`.
- `data/amazon_books/raw/books_data.csv`: metadata; không có item ID, chỉ được
  join exact title.
- 2.438.194 interaction có user/item/timestamp hợp lệ; 561.787 dòng userless và
  19 timestamp lỗi bị loại.
- Timeline theo ngày từ 1996-08-17 đến 2013-03-04. Split global 80/10/10 tạo
  train cutoff và validation cutoff; mọi event cùng timestamp là một unordered
  batch. Event ở `t` chỉ được đọc state có timestamp `< t`.
- Exact-title metadata coverage trên usable review là 99,992%; không fuzzy join.

Candidate protocol cố định: một positive novel target và chín negative lấy
deterministically, uniform từ positive-item pool đã xuất hiện trước train
cutoff. Negative phải unseen trong strict-past history của target user. Candidate
order cố định theo hash; graph scorer không đọc gold label.

## 3. Frozen local ranker

Local baseline dùng `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, bf16, greedy decoding, seed
`20260921`:

1. Stage-R đọc tối đa sáu review strict-past gần nhất và sinh tối đa năm
   preference facets.
2. Listwise reranker nhận cùng facets và base memory của mười candidate, rồi
   trả permutation A–J.
3. Base memory gồm title và exact-title metadata, giới hạn 420 ký tự.

P7-v2 schema buộc array đúng mười label thuộc A–J. Nếu model lặp label, parser
giữ lần xuất hiện đầu tiên rồi nối label thiếu theo A–J. Đây chỉ là syntax
completion, không dùng item ID, graph score hoặc gold. Toàn run đặt cap năm
completion; thực tế cần **0**.

## 4. Directed transition graph

Với mỗi user `u`, gom strict-past interactions thành chuỗi timestamp batch:

```text
B(u, t1), B(u, t2), ..., B(u, tk),  t1 < t2 < ... < tk
```

Với hai batch liên tiếp, mỗi item `a ∈ B(u, ti)` trỏ tới destination group
`B(u, ti+1)`. Không tạo cạnh bên trong cùng ngày. Một source có thể giữ nhiều
destination group; occurrence lặp qua user/thời điểm được giữ để thể hiện tần
suất chuyển tiếp. Group representation tránh giả định thứ tự nội batch và tránh
materialize Cartesian product.

Graph chỉ dùng interaction trước global validation cutoff. Snapshot P7 có:

| Statistic | Value |
|---|---:|
| Users | 1.008.972 |
| Source items | 142.572 |
| Consecutive batch pairs | 511.907 |
| Source-to-group links | 876.669 |

Không dùng rating filter, hub penalty, semantic filter, beam search hoặc
target-aware path selection.

## 5. Multi-hop terminal-state PPR estimator

Seed distribution là uniform trên tối đa sáu unique item strict-past gần nhất
của target user. Mỗi event chạy 50.000 deterministic Monte Carlo walks:

1. Chọn một seed uniform.
2. Ở mỗi step, dừng với probability `r = 0,15`.
3. Nếu tiếp tục, chọn một observed destination group của current item uniform,
   rồi chọn một item trong group uniform.
4. Dừng khi restart coin kích hoạt, node không có outgoing group, hoặc đủ 64
   step; chỉ terminal item được đếm nếu nó là candidate.

Vì vậy implementation là personalized geometric-stop/terminal-state estimator,
không phải phép tính exact stationary vector. Seed, walk RNG và event order đều
được hash cố định; deterministic replay được smoke-test.

Với candidate `i`, graph score là `g_i = terminal_count_i / 50.000`. One-step
transition score được tính riêng để audit reachability, không tham gia primary
ranking.

## 6. Residual fusion

Local rank `r_i` được đổi thành reciprocal-log score:

```text
l_i = 1 / log2(r_i + 1),  r_i bắt đầu từ 1
```

PPR score được min–max trong đúng mười candidate:

```text
g'_i = (g_i - min(g)) / (max(g) - min(g))
s_i  = l_i + alpha * g'_i
```

Nếu mọi graph score bằng nhau, giữ nguyên local ranking. Calibration grid được
khóa là `[0, 0.05, 0.10, 0.20, 0.40, 0.80]`; prior cohort chọn `alpha = 0,80`.
Fresh-test không retune alpha/restart/walk/depth.

## 7. Cohort và leakage control

- Calibration: 100 event P6 đã được quan sát; chỉ dùng chọn alpha và admission.
- Primary: 100 positive event mới sau validation cutoff, user-disjoint với P6
  train/test và validation cũ.
- Replication: 200 positive event mới, mỗi user đúng một event và user-disjoint
  với toàn bộ 1.500 user từng xuất hiện trong các cohort P3–P7 đã khóa.
- 0/100 calibration gold và 0 candidate từng xuất hiện trong strict-past user
  history của event tương ứng.
- Graph, candidate pool, local output và alpha được hash trước khi mở primary
  labels. Evaluation manifest ghi `test_tuning_after_labels=false`.
- Post-hoc best-of local/PPR là oracle không deployable và không đổi decision.

## 8. Tiến độ thực nghiệm

| Milestone | Trạng thái | Kết quả |
|---|---|---|
| Dataset audit | Complete | Strict-past timestamp-batch replay hợp lệ |
| Graph smoke | Pass | 20 calibration + 20 fresh; deterministic replay |
| Calibration | Pass | Local 0,5897; fused 0,7088; +0,1191; alpha 0,8 |
| P7-v1 LLM | Invalid, sealed | Một duplicate-label response; không evaluate |
| P7-v2 LLM smoke | Pass | 21 event, 42/42 success, 0 error/retry/repair |
| P7-v2 full local ranker | Complete | 200/200 primary requests, 0 retry/repair |
| Fresh evaluation | **Pass** | +0,0752 NDCG@5; CI lower dương |
| Replication R0/R1 | Pass | 200 user-disjoint event; graph deterministic/leakage-safe |
| Replication LLM | Complete | 400/400 primary requests, 0 retry/repair |
| Replication evaluation | **Pass** | +0,1371 NDCG@5; CI [+0,0904; +0,1838] |
| MovieLens frozen transfer | Complete | 500 event; 1.000/1.000 LLM requests, 0 retry/repair |
| MovieLens primary evaluation | **Pass** | Exact-PPR +0,3663; CI [+0,3305; +0,4007] |
| MovieLens secondary evaluation | **Pass** | Session-PPR +0,3774; CI [+0,3430; +0,4115] |

P7-v1 journal dừng ở 130 attempts và không được trộn vào v2. Regression event
gây lỗi v1 được thêm vào v2 smoke trước full run.

## 9. Discovery result — P7-v2

| Arm | NDCG@5 | Hit@5 |
|---|---:|---:|
| Frozen local | 0,6533 | 0,80 |
| Transition-PPR residual | **0,7285** | **0,87** |

- Delta NDCG@5: **+0,0752**.
- Paired bootstrap 95% CI: **[+0,0179; +0,1375]**.
- Improved / worsened / unchanged event: 22 / 16 / 62.
- Primary gate (+0,02 và CI lower > 0): **pass**.
- Gold reachability: one-step 21/100; PPR 63/100.
- Negative reachability: one-step 1/900 (0,11%); PPR 97/900 (10,78%).
- Post-hoc oracle best-of-two: 0,7740, +0,1207 vs local; không deployable.

Kết quả cho thấy graph sâu có ích khi edge semantics bám temporal next-item;
gain không đến từ mở rộng co-preference graph bằng nhiều rule hơn. Replication
được khóa trước để kiểm tra liệu effect có lặp lại ngoài 100 event này hay không.

## 10. Internal replication result — final Amazon Books number

Replication giữ nguyên toàn bộ method parameter và alpha 0,80, nhưng dùng 200
event/user mới, không overlap với các cohort trước.

| Arm | NDCG@5 | Hit@5 |
|---|---:|---:|
| Frozen local | 0,624675 | 0,820 |
| Transition-PPR residual | **0,761807** | **0,905** |

- Delta NDCG@5: **+0,137133**.
- Paired bootstrap 95% CI: **[+0,090385; +0,183835]**.
- Improved / worsened / unchanged event: 64 / 36 / 100.
- Pre-registered replication gate (+0,03 và CI lower > 0): **pass**.
- Gold reachability: one-step 36/200; PPR 123/200.
- Negative reachability: one-step 8/1.800 (0,44%); PPR 172/1.800
  (9,56%).
- Post-hoc oracle best-of-two: 0,803821, +0,179147 vs local; không deployable.

Đây là số final cho Amazon Books. Không tiếp tục tune alpha/restart/depth trên
200 labels này; mọi thay đổi tiếp theo phải dùng validation/cohort/dataset khác.

## 11. Frozen cross-domain result — MovieLens 32M

MovieLens study giữ nguyên model, revision, graph parameters và alpha 0,80 từ
Amazon, nhưng đổi rendering sang prior `title + genres + rating` và candidate
`title + genres`. Cohort gồm 500 singleton-five-minute-session rating events;
graph cutoff luôn trước target và candidate protocol vẫn là một positive cộng
chín uniform strict-past-unseen negatives.

| Arm | NDCG@5 | Hit@5 |
|---|---:|---:|
| Frozen local | 0,471944 | 0,702 |
| Exact Transition-PPR residual — primary | **0,838235** | **0,952** |
| Session-300s Transition-PPR residual — secondary | **0,849310** | **0,960** |

- Primary exact delta: **+0,366290**, CI95%
  **[+0,330511; +0,400667]**, gate pass.
- Secondary session delta: **+0,377365**, CI95%
  **[+0,343037; +0,411524]**, gate pass.
- Preregistered conclusion: **robust cross-domain transfer**.
- Retry/parser/permutation repair: 0/0/0 across 1.000 physical requests.

The effect should be interpreted as transition-graph transfer, not proof that
multi-hop is intrinsically superior. Exact/session one-step graph-only reached
0,861449/0,869054 NDCG@5, both above their PPR graph-only variants. Graph-only
PPR also slightly exceeded residual PPR. The easy ten-candidate uniform-negative
setting is highly graph-separable; harder retrieval-scale negatives remain a
new-study requirement. The full protocol, all arms, coverage, compute audit and
sealed hashes are in [MOVIELENS32M_PROTOCOL.md](MOVIELENS32M_PROTOCOL.md).

## 12. Compute và request audit

- Smoke: 42 requests, 39.810 token.
- Full invocation: reuse smoke và chạy thêm 158 requests, 140.778 token.
- Tổng: 200 physical requests; 169.901 input + 10.687 output = 180.588 token.
- Peak VRAM: 7,8325 GiB trên đúng một H100 80 GB; tensor parallel = 1.
- Sau smoke/full, process thoát và cả bốn GPU về baseline 1 MiB.

Replication:

- Smoke: 40 requests; 34.635 input + 2.127 output = 36.762 token.
- Full invocation: reuse smoke, thêm 360 requests; 307.258 input + 19.291
  output = 326.549 token.
- Tổng: 400 physical requests; 341.893 input + 21.418 output = 363.311 token.
- Retry/parser/permutation repair: 0/0/0.
- Peak VRAM: smoke 7,829 GiB; full 7,863 GiB trên đúng một H100.
- Sau smoke/full, VRAM của GPU đã chọn về baseline lần lượt 4 MiB và 1 MiB.

MovieLens 32M frozen transfer:

- Smoke: 40 requests; 17.272 input + 1.767 output = 19.039 token.
- Full invocation reuse smoke, thêm 960 requests; 415.165 input + 41.733
  output = 456.898 token.
- Tổng: 1.000 physical requests; 432.437 input + 43.500 output = 475.937
  token; retry/parser/permutation repair = 0/0/0.
- Peak VRAM: smoke 7,702 GiB; full 7,707 GiB trên đúng một H100; GPU về
  baseline 1 MiB sau full run.

## 13. Reproducibility surface

Source còn active:

- `src/temporal_books/p7_transition_ppr.py`: graph, scoring, calibration/eval.
- `src/temporal_books/p7_selfhost_local.py`: smoke-first frozen LLM runner.
- `src/temporal_books/current_support.py`: shared data/ranking/journal utilities.
- `src/temporal_books/p0_audit.py`: raw temporal dataset audit.
- `configs/temporal_amazon_books_2014/p7v2_transition_ppr.yaml`: sealed config.
- `configs/temporal_amazon_books_2014/transition_ppr_replication_200.yaml`:
  sealed replication config và là default active config.
- `src/temporal_movielens/`: MovieLens audit, preparation, graph scorer, frozen
  local ranker và sealed evaluator.
- `configs/temporal_movielens32m/m1_frozen_transfer.yaml`: sealed MovieLens
  transfer config.
- `docs/MOVIELENS32M_PROTOCOL.md`: canonical MovieLens protocol/result record.

P7-v2 artifacts là sealed historical record. Active commands dưới đây trỏ tới
replication config mới, không overwrite run P7-v2:

```bash
python -m src.temporal_books.p0_audit --config configs/temporal_amazon_books_2014/dataset_audit.yaml
python -m src.temporal_books.p7_transition_ppr --prepare
python -m src.temporal_books.p7_transition_ppr --graph-smoke
python -m src.temporal_books.p7_selfhost_local --smoke-only
python -m src.temporal_books.p7_selfhost_local --request
python -m src.temporal_books.p7_transition_ppr --evaluate
```

GPU commands phải tuân theo local-only `internal_docs/H100_RESOURCE_RULES.md`:
smoke 20–30 mẫu, dynamic idle-GPU selection, đúng một visible GPU cho contract
hiện tại, và unload ngay khi xong.

## 14. Canonical hashes

Replication final:

| Artifact | SHA256 |
|---|---|
| Config | `4a0aa4fee584983eaa4dedacd0633c1293e187da66739ab4ba621d7f43b62098` |
| Prepared cohort | `6b1726a620313051bdf0ea488d4a87f7c8dc42bf259cb87dea9b0e816e54a41e` |
| Locked method | `4e3d6e98cfed44143b43ce33c0fd6363352f494a353a069d40bd8b8e47d3936a` |
| Local LLM manifest | `77aec295a6745c9259720d98d1baddadd2b3dc5ade91869fa8bc15a2c814f36c` |
| Test scores, 200 rows | `58c550c5dc67dcc6a26d1e0f8cf3695e91b4669120b9de693e801894c7ebe219` |
| Metrics | `956cb632e00b7a1b45ae7307cbe06cd08c95953b3fd9e030cdd9eb6a99cefaf6` |
| Evaluation manifest | `688fb5c1fee96e047c7e3905334f389edbe861c246acc3e829ffb87be7efc3a7` |

P7-v2 discovery record:

| Artifact | SHA256 |
|---|---|
| Prepared fresh cohort | `4ebe1a8a10f33239d472bb64e7119de057ffc4e3dcc56183b2bb5591310597a9` |
| Locked alpha selection | `1c40ea0966c3e528f9243547edd712076d1de056a757136cdeda9c3123bfc9fc` |
| Local LLM manifest | `e6315870a2fcdd389bb5338593e494fcfcfcccdb2adbd1f1c4b6b4e39dc28102` |
| Test scores, 100 rows | `2fd3fd94897e0138cd9dbc0c116f1e5bb7df50087045e94de690b72940260e2e` |
| Metrics | `07bba2e2f222f1a9665a09d190ad53f9262c64b5bbd974bb3ecb3ecc138b93e3` |
| Evaluation manifest | `0c33effce9056237a186f6c83353c14b40757b06eaea5484b4873c4d0c6c4d1c` |

## 15. Hạn chế

- Discovery có 100 và internal replication có 200 event; cả hai vẫn dùng
  candidate task 1-positive/9-uniform-negative khá dễ.
- Dataset cũ, timestamp chỉ tới độ phân giải ngày.
- Graph snapshot lớn được dựng lại per command, chưa tối ưu production latency.
- Frozen local ranker là Qwen3-4B listwise, chưa chứng minh gain transfer sang
  ranker mạnh hơn hoặc retrieval candidate pool lớn.
- Alpha được calibration trên Amazon và frozen-transfer sang MovieLens đã pass,
  nhưng cả hai dataset vẫn dùng task 1-positive/9-uniform-negative; external
  validity cho candidate retrieval khó hơn hoặc domain khác chưa được chứng minh.
