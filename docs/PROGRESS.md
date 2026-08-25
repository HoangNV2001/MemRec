# PROGRESS.md — Nhật ký Selective Multi-Hop

> **Kế hoạch active:** [MULTIHOP_PLAN.md](MULTIHOP_PLAN.md). Tổng kết hướng đã
> đóng: [RL_WORK_SUMMARY.md](RL_WORK_SUMMARY.md).
>
> Quy tắc: ghi entry ngay sau mỗi milestone/run; không diễn giải kết quả chưa có
> artifact. Mọi command, commit, hash và cost phải đủ để tái lập run.

## Trạng thái hiện tại

| Milestone | Trạng thái | Quyết định / đầu ra |
|---|---|---|
| MH0 — Freeze protocol | 🟡 in progress | Docs đã chuyển sang multi-hop; cần verification, topology freeze và materialize control trước API run. |
| MH1 — 2-hop pool + bundles | ⏳ | Chưa bắt đầu implementation. |
| MH2 — Oracle headroom (val) | ⏸ | Chờ MH0/MH1; là gate bắt buộc. |
| MH3 — Selective selector | ⏸ | Chỉ nếu MH2 pass. |
| MH4 — Locked test | ⏸ | Chỉ nếu MH3 config được khóa. |
| MH5 — Selective propagation | ⏸ | Chỉ nếu read-side thắng. |

## MH0 — Freeze protocol — 2026-08-25

**Đã làm**

- Đọc code/datasets/docs của repo và đóng hướng SFT/RL trong
  RL_WORK_SUMMARY.md.
- Viết experiment contract, oracle protocol và gate cho multi-hop trong
  MULTIHOP_PLAN.md.
- Khởi tạo bảng kết quả trống không trộn baseline M0 RNG theo thread với candidate
  set fixed của protocol mới.

**Cần chạy trước MH1**

- [ ] Chạy python -m src.rl.verify_transfer hoặc equivalent check snapshot/jsonl.
- [ ] Chạy pytest tests/rl/ -q.
- [ ] Dựng topology từ pre-target interaction history, rồi ghi SHA256 topology,
  snapshot và input jsonl vào data/multihop/mh0_manifest.json.
- [ ] Re-materialize exact one_hop control và lưu per-user K_u, T_u,
  candidate-order hash.

**Gate:** chỉ chuyển MH1 sau khi control reproducible và mọi guard leakage pass.

## MH1 — 2-hop pool + bundle generator — pending

**Mục tiêu:** implement C2(u), budget controller và naive_two_hop/oracle bundles
theo §3–5 của plan.

| Field | Điền sau |
|---|---|
| Ngày / commit | — |
| Input manifest hash | — |
| Command | — |
| Unit tests | — |
| Users smoke / val | — |
| Pool coverage / shortfall | — |
| Budget assertion | — |
| Compute / API cost | — |
| Quyết định | — |

## MH2 — Oracle headroom validation — pending

| Field | Điền sau |
|---|---|
| Ngày / commit / run ID | — |
| Validation records analysed | — |
| Reranker + settings | — |
| Bundles/user, quota q | — |
| Selection pass artifact | — |
| Independent report pass count | — |
| Delta_naive NDCG@5 + 95% CI | — |
| Delta_oracle NDCG@5 + 95% CI | — |
| Budget/coverage audit | — |
| Cost / wall time | — |
| **Gate decision** | — |

## MH3 — Selective selector — pending

| Field | Điền sau |
|---|---|
| Frozen config hash (alpha, beta, gamma, q) | — |
| Validation delta vs 1-hop / naive | — |
| Oracle capture | — |
| Qualitative audit | — |
| Leakage/budget checks | — |
| Test admission decision | — |

## MH4 — Locked test — pending

| Field | Điền sau |
|---|---|
| Locked config/manifest hash before run | — |
| Test users analysed | — |
| Paired Delta NDCG@5 + 95% CI | — |
| Secondary metrics | — |
| Token/latency delta | — |
| Failure-bucket analysis | — |
| Final conclusion | — |

## MH5 — Selective propagation — pending / conditional

| Field | Điền sau |
|---|---|
| Admission evidence from MH4 | — |
| Endpoint cap / coverage | — |
| Temporal leakage checks | — |
| Ranking and cost results | — |
| Conclusion | — |
