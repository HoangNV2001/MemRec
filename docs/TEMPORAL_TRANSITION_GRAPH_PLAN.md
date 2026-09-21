# TEMPORAL_TRANSITION_GRAPH_PLAN.md — P7 directed transition PPR

> **Protocol khóa trước result:** 2026-09-21.
> **Trạng thái:** graph smoke + calibration complete; fresh-test admitted,
> chưa chạy LLM/evaluation.

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
- Fresh-test labels chưa được mở. Bước kế tiếp: self-host smoke 20, full local
  baseline rồi chạy đúng locked alpha một lần.
