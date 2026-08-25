# RESULTS.md — Selective Multi-Hop Collaborative Memory

> **Active protocol:** [MULTIHOP_PLAN.md](MULTIHOP_PLAN.md).
>
> Các số của hướng SFT/RL cũ nằm trong [RL_WORK_SUMMARY.md](RL_WORK_SUMMARY.md)
> và không được dùng làm baseline trực tiếp ở đây: M0 cũ sample negative theo
> thread, còn bảng này chỉ dùng candidate set deterministic đã materialize.

## 1. Protocol registry

| Field | Giá trị |
|---|---|
| Dataset | InstructRec-Books |
| Frozen memory source | data/rl/graph_snapshot_books.json (cần ghi hash ở MH0) |
| Frozen topology source | data/multihop/mh0_topology_books.json, xây từ pre-target history (pending MH0) |
| Split | train 1,185 / val 149 / test 993 |
| Candidate protocol | 10 fixed candidates/user; cùng thứ tự trong mọi arm của user |
| Primary metric | Paired NDCG@5, 10,000 bootstrap user resamples |
| Context control | Per-user K_u node slots và T_u token cap từ 1-hop control |
| Active result status | Chưa có run multi-hop |

## 2. MH0 control verification

| Check | Status | Evidence / value |
|---|---|---|
| Snapshot/jsonl integrity | ⏳ | — |
| Topology frozen from pre-target history | ⏳ | — |
| Split disjoint | ⏳ | — |
| Selector/context has no instruction/candidate/gold; ranker input is expected | ⏳ | — |
| Candidate-order hash equal across arms | ⏳ | — |
| one_hop re-materialized from snapshot | ⏳ | — |
| Per-user K_u, T_u persisted | ⏳ | — |

## 3. MH2 validation — bounded oracle headroom

All rows must use the **independent report pass**, not the oracle selection pass.
Delta is paired against one_hop on exactly the same users.

| Arm | N users | NDCG@5 | Delta vs 1-hop | 95% CI | H@1 | H@3 | H@5 | Nodes | Context tokens | Remote ratio | Status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| one_hop | — | — | — | — | — | — | — | — | — | 0 | pending |
| naive_two_hop | — | — | — | — | — | — | — | — | — | — | pending |
| oracle_two_hop | — | — | — | — | — | — | — | — | — | — | pending |

### Oracle audit

| Field | Value |
|---|---|
| Bundles/user and quota distribution | — |
| Selection-pass reranker calls | — |
| Independent report-pass calls/arm | — |
| Delta_oracle gate | — |
| Pool coverage / remote shortfall | — |
| Reranker variability estimate | — |
| API/GPU cost and wall time | — |

### Gate decision

| Criterion | Result |
|---|---|
| Stop: Delta_oracle <= +0.02 or CI upper <= +0.03 | — |
| Borderline: CI crosses 0 or +0.02 < Delta_oracle < +0.05 | — |
| Proceed MH3: Delta_oracle >= +0.05 and CI lower > +0.01 | — |
| Decision and rationale | — |

## 4. MH3 validation — selector after config lock

| Arm | Selector config hash | N users | NDCG@5 | Delta vs 1-hop | Delta vs naive | 95% CI vs 1-hop | Oracle capture | Budget pass | Status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| one_hop | control | — | — | — | — | — | — | — | pending |
| naive_two_hop | fixed traversal | — | — | — | — | — | — | — | pending |
| selective_two_hop | — | — | — | — | — | — | — | — | pending |

### Selector diagnostics

| Measure | Value |
|---|---|
| alpha, beta, gamma, q | — |
| Semantic/path/redundancy score distributions | — |
| Remote item/user mix | — |
| Degree/hub distribution vs 1-hop | — |
| Duplicate/redundant node rate | — |
| 30-user qualitative audit | — |

## 5. MH4 locked test

> Fill only after validation config and manifest hash are frozen. No parameter may
> be changed after looking at this table.

| Arm | N users | H@1 | H@3 | NDCG@3 | H@5 | **NDCG@5** | Paired Delta vs 1-hop | 95% CI | Tokens/query | Latency/query | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| one_hop | — | — | — | — | — | — | — | — | — | — | pending |
| naive_two_hop | — | — | — | — | — | — | — | — | — | — | pending |
| selective_two_hop | — | — | — | — | — | — | — | — | — | — | pending |

### Test robustness and breakdown

| Slice / check | 1-hop | naive 2-hop | selective 2-hop | Finding |
|---|---:|---:|---:|---|
| Short vs long history | — | — | — | — |
| Low vs high graph degree | — | — | — | — |
| Remote quota | — | — | — | — |
| Remote path type | — | — | — | — |
| Budget equality / shortfall | — | — | — | — |
| Candidate/gold leakage audit | — | — | — | — |

## 6. MH5 selective propagation (conditional)

| Arm | Endpoint cap | Coverage | Duplicate/stale rate | NDCG@5 Delta | Tokens/latency delta | Temporal leakage | Status |
|---|---:|---:|---:|---:|---:|---|---|
| one_hop_write | — | — | — | — | — | — | pending |
| naive_two_hop_write | — | — | — | — | — | — | pending |
| selective_two_hop_write | — | — | — | — | — | — | pending |

## 7. Final conclusion

**Pending.** State one of the following only after the relevant gate/run exists:

- No meaningful multi-hop headroom under equal budget; stop after MH2.
- Oracle headroom exists but current selector cannot capture it; report selection gap.
- Selective multi-hop improves ranking under equal budget; report locked-test effect
  size, confidence interval, cost and limits.
