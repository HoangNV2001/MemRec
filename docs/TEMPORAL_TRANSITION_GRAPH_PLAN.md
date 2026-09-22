# TEMPORAL_TRANSITION_GRAPH_PLAN.md — P7 directed transition PPR

> **Protocol khóa trước result:** 2026-09-21.
> **Trạng thái:** P7-v2 complete, primary fresh-test gate **pass**.
> P7-v1 LLM run đã bị niêm phong do output-contract failure; P7-v2 được khóa
> trước khi chạy lại và được evaluate đúng một lần, không post-hoc tuning.

## Câu hỏi

P3–P6 dùng user–item co-preference graph: đi xa tăng reachability nhưng cũng
tăng hub/noise. P7 thay đổi semantics của graph, không tune path rule: cạnh có
hướng `item A → item B` khi một user tương tác B ở timestamp batch kế tiếp sau
A. Multi-hop Personalized PageRank (PPR) vì vậy đo khả năng chuyển tiếp tới
candidate, gần objective “item tiếp theo” hơn.

## Graph và scorer

- Chỉ dùng interaction strict trước global validation cutoff để dựng graph.
- Interaction cùng timestamp là một unordered batch, không tạo cạnh nội bộ.
- Giữa hai batch liên tiếp: mỗi source item trỏ tới cả destination batch;
  destination được lấy uniform trong batch. Cách biểu diễn group transition
  tránh giả định thứ tự trong cùng ngày và tránh materialize tích Descartes.
- Restart distribution uniform trên tối đa 6 item strict-past gần nhất của
  target user.
- PPR được ước lượng deterministic bằng 50.000 random walk/event, restart 0,15,
  tối đa 64 step. Secondary arm là exact one-step transition trên cùng graph.
- Không dùng rating/path length/hub threshold, semantic filter hay beam search.

Graph score được min-max trong 10 candidate rồi residual-fuse với frozen local
rank score. P6 fresh test đã mở được dùng làm **calibration**; alpha chỉ chọn từ
grid khóa `[0, 0.05, 0.10, 0.20, 0.40, 0.80]`, tie chọn alpha nhỏ hơn.

## Fresh cohort và staged gate

- P7 primary test gồm 100 positive novel event sau validation cutoff, disjoint
  với toàn bộ P6 train/test và validation cũ.
- Candidate protocol giữ nguyên: gold + 9 deterministic uniform unseen item
  đã có positive interaction trước train cutoff.
- Graph smoke bắt buộc trên 20 calibration + 20 fresh event, dùng 5.000 walk để
  kiểm determinism/schema/strict-past; full calibration vẫn dùng 50.000 walk.
- Chỉ chạy thêm 200 self-host LLM generation nếu calibration PPR residual tăng
  ít nhất +0,01 NDCG@5 và alpha được chọn >0. Nếu fail, hard stop trước GPU.
- Nếu admitted: LLM smoke 20 trước full 100, cùng Qwen revision/cap P6; primary
  pass khi fresh-test delta >=+0,02 và paired CI lower >0.

### P7-v2 LLM output contract

P7-v1 yêu cầu model sinh strict permutation A–J. Full run dừng ở event
`APZAJ6LGXH02F:1359590400` vì structured decoder cho phép array đủ 10 phần tử
nhưng không ép `uniqueItems`; model lặp một label và bỏ một label. Retry
byte-identical cũng fail. Run v1 được niêm phong ở 130 physical attempts
(127 successful keys, 2 error rows), không evaluate và không tái sử dụng
journal trong v2.

P7-v2 giữ nguyên model, prompt, graph, cohort, alpha và generation budget; chỉ
khóa lại output contract:

- JSON schema buộc đúng 10 phần tử, mỗi phần tử thuộc A–J.
- Nếu label lặp, giữ lần xuất hiện đầu tiên rồi nối các label còn thiếu theo
  thứ tự A–J. Đây là deterministic syntax completion, không dùng item ID,
  gold label hay graph/test signal.
- Event làm v1 fail bắt buộc được thêm vào LLM smoke; tổng smoke vẫn trong
  20–30 event.
- Tối đa 5 rerank response được phép cần permutation completion trong toàn run;
  vượt cap thì run fail trước evaluation.
- V2 dùng run ID và journal mới. Chỉ manifest smoke đúng config/prepared/
  selection hash mới mở gate cho full run.

Offline gate P7-v2 đã pass trước GPU: 21 unit/regression test pass; graph smoke
deterministic và calibration tái lập đúng score P7-v1. Config SHA256 là
`7fda73bb88281b2197a156e9dfec55fb0a9d96b510aba66cb597585d0db529fd`,
graph-smoke SHA256 là
`84dbc5531ccaf66244322502523c8cc329d1d90f56728df6612abc44438411ac`,
locked-selection SHA256 là
`1c40ea0966c3e528f9243547edd712076d1de056a757136cdeda9c3123bfc9fc`.
Calibration manifest ghi nhận `fresh_test_labels_accessed=false`, `gpu_used=false`.

P7 test label không được dùng trước khi graph/config/alpha và local LLM output
đã khóa. Post-hoc oracle không thay đổi primary decision.

## PROGRESS & RESULTS

Protocol trên được khóa khi chưa có result. Sau đó:

- Unit test synthetic xác nhận same-timestamp batching và PPR reach được node
  2-hop mà one-step không reach.
- Materialize 100 fresh event, zero user overlap với P6 train/test, mỗi prompt
  có 5–6 review strict-past. Prepared SHA256:
  `4ebe1a8a10f33239d472bb64e7119de057ffc4e3dcc56183b2bb5591310597a9`.
- Graph smoke 20 calibration + 20 test pass và deterministic replay; one-step/
  PPR có evidence ở 6/19 trên 400 candidate slot. Smoke SHA256:
  `7da86ebf5375f010a2f07ebf2c45d8770f508717440baba2bfa01384100a5177`.
- Leakage audit: 0/100 calibration gold và 0 candidate từng có trong toàn bộ
  strict-past user history.
- Full calibration: PPR cover 53% gold và 8,67% negative slot; graph-only
  NDCG@5 0,6483. Residual chọn alpha 0,8, đạt 0,7088 vs local 0,5897,
  **+0,1191**; gate admit fresh test. Locked-selection SHA256:
  `61e108c92dedcf17d7552a290680d68de145724aee483727a1f6bbf5342e201e`.
- Self-host smoke chạy 21 event (20 deterministic + regression event), 42/42
  success, zero error/retry/parser repair/permutation completion. Full journal
  hoàn tất 200/200 primary request, zero retry/error/repair; 42 smoke request
  được reuse nên invocation full chỉ chạy thêm 158 request. Tổng hai invocation
  là 169.901 input + 10.687 output = 180.588 token. Qwen3-4B revision
  `cdbee75f…0554`, bf16, một H100, peak VRAM 7,8325 GiB; process thoát và cả
  bốn GPU về 1 MiB. Local-manifest SHA256:
  `e6315870a2fcdd389bb5338593e494fcfcfcccdb2adbd1f1c4b6b4e39dc28102`.
- Fresh test được mở đúng một lần sau khi local output và alpha 0,8 đã khóa.
  Local NDCG@5/Hit@5 = 0,6533/0,80; transition-PPR residual = 0,7285/0,87,
  **delta NDCG@5 +0,0752**, paired bootstrap 95% CI
  **[+0,0179; +0,1375]**. Primary gate (+0,02 và CI lower >0) pass.
- Residual cải thiện 22 event, làm tệ 16, giữ nguyên 62. Gold coverage one-step
  21/100 và PPR 63/100; negative coverage tương ứng 1/900 (0,11%) và 97/900
  (10,78%). Oracle post-hoc đạt 0,7740 (+0,1207 vs local), chỉ dùng đo
  headroom, không thay primary decision.
- Test-score có 100 record; manifest xác nhận `test_tuning_after_labels=false`.
  Metrics/test-score/evaluation-manifest SHA256 lần lượt là
  `07bba2e2f222f1a9665a09d190ad53f9262c64b5bbd974bb3ecb3ecc138b93e3`,
  `2fd3fd94897e0138cd9dbc0c116f1e5bb7df50087045e94de690b72940260e2e`,
  `0c33effce9056237a186f6c83353c14b40757b06eaea5484b4873c4d0c6c4d1c`.

**Kết luận:** temporal transition graph + multi-hop PPR transfer thành công sang
fresh cohort và vượt gate thống kê đã khóa. Không retune alpha/walk/restart trên
test này; bước kế tiếp hợp lệ phải là replication trên cohort/dataset mới hoặc
ablation đã định trước, không khai thác thêm chính 100 labels vừa mở.
