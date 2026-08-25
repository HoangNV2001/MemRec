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
| Frozen memory source | `data/rl/graph_snapshot_books.json`, SHA256 `baa83950968699af9e26b0695a8cc5dca82265c43047643643e6b9646a2d5c3d` |
| Frozen topology source | `data/multihop/mh0_topology_books.json`, xây từ pre-target history; SHA256 `75744eb2bae77bbcc463208d557eb528fcaa3475d86d2914b2b19456840b3180` |
| Split | train 1,185 / val 149 / test 993 |
| Candidate protocol | 10 fixed candidates/user; cùng thứ tự trong mọi arm của user |
| Primary metric | Paired NDCG@5, 10,000 bootstrap user resamples |
| Context control | Per-user K_u node slots và T_u token cap từ 1-hop control |
| Active result status | MH2 validation complete: bounded oracle không có headroom thực dụng; protocol dừng sau MH2 |

## 2. MH0 control verification

| Check | Status | Evidence / value |
|---|---|---|
| Snapshot/jsonl integrity | ✅ | Recognized `legacy_pre_m2_backfill` profile; all input hashes in `data/multihop/mh0_manifest.json` |
| Topology frozen from pre-target history | ✅ | 7,377 users / 111,084 items / 193,005 edges; SHA256 `75744e…b3180` |
| Split disjoint | ✅ | train 1,185 / val 149 / test 993 |
| Selector/context has no instruction/candidate/gold; ranker input is expected | ✅ | `stage_r_context` and `ranking_context` physically separated; unit guard passed |
| Candidate-order hash equal across arms | ✅ (control) | One immutable candidate-order hash/user is persisted; MH1 will assert equality when extra arms exist |
| one_hop re-materialized from snapshot | ✅ | 2,327 source prompts match snapshot-rendered prompt byte-for-byte; gold leakage = 0 |
| Per-user K_u, T_u persisted | ✅ | `K_u`: 8–16; serialized neighbor `T_u`: 241–727 (train), 403–726 (val), 258–727 (test) |

MH0 was rerun with `--force`: topology and all three controls retained their
SHA256 values. Its manifest records 0 LLM calls and USD 0 API cost. The legacy
profile is accepted only as a frozen input bundle; no retired RL reward field is
used by the multi-hop protocol.

## 3. MH1 pool and bundle verification

| Check | Status | Evidence / value |
|---|---|---|
| Candidate-blind bounded C2 construction | ✅ | Peer user from MH0 only; cap = 128 remote item, 64 remote user, 8 remote user/item |
| Pool materialization | ✅ | train/val/test = 1,185 / 149 / 993; pool SHA256 `5cdc2e…6498` / `415f58…ed7f` / `6f78ce…36af` |
| Validation bundle count | ✅ | 3 naive + 12 oracle bundle/user; 149 records, bundle SHA256 `7b846a…1458` |
| Exact budget | ✅ | Every bundle has exact `K_u` and `T_actual ≤ T_u`; one-hop fallback is persisted |
| Remote coverage | ✅ with shortfall | No val pool shorter than quota 6; 763 / 2,235 bundles have remote shortfall due token cap, never budget overflow |
| Candidate/gold leakage audit | ✅ guard | Full-C2 and bundle screen removes 8 users from every MH2 arm; remaining locked cohort = 141/149 |
| Determinism | ✅ | Rerun with `--force` preserved all pool/bundle SHA256; 9 MH0/MH1/Azure-config tests passed |

No ranking result is implied by these artifacts. MH2 must use exactly the
141-user cohort and report remote shortfall by quota/arm.

## 4. MH2 validation — bounded oracle headroom

All rows must use the **independent report pass**, not the oracle selection pass.
Delta is paired against one_hop on exactly the same users.

| Arm | N users | NDCG@5 | Delta vs 1-hop | 95% CI | H@1 | H@3 | H@5 | Nodes | Context tokens | Remote ratio | Status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| one_hop | 141 | 0.8079 | reference | — | 0.6844 | 0.8333 | 0.9220 | 15.66 | 614.84 (T_u) | 0.000 | complete |
| naive_two_hop (q=4) | 141 | 0.7885 | −0.0193 | [−0.0476, +0.0067] | 0.6383 | 0.8440 | 0.9149 | 15.66 | 590.65 (T_actual) | 0.219 (3.39 nodes) | complete |
| oracle_two_hop | 141 | 0.7984 | **−0.0094** | **[−0.0362, +0.0176]** | 0.6489 | 0.8617 | 0.9255 | 15.66 | 600.43 (T_actual) | 0.122 (1.90 nodes) | complete — stop |

Cột node và token là trung bình trên đúng 141 user được phân tích. Mọi bundle
2-hop đều giữ exact K_u của control và T_actual ≤ T_u; thiếu remote được fill
bằng node 1-hop thay vì tăng context.

### Oracle audit

| Field | Value |
|---|---|
| Bundles/user and quota distribution | 12 candidate-blind bundles/user: 4 each for q={2,4,6}. Chosen oracle: q2=124, q4=7, q6=10 (mean requested q=2.38). |
| Selection-pass reranker calls | 1,692 = 141 × 12; used only to select each user's oracle bundle by NDCG@5. |
| Independent report-pass calls/arm | 282/arm = 141 × 2 repeats; total 846. No selection score appears in the result table. |
| Delta_oracle gate | ΔNDCG@5 = −0.0094, 10,000 paired-user bootstrap 95% CI [−0.0362, +0.0176]. |
| Pool coverage / remote shortfall | Naive q=4: 42/141 users, 86 remote slot shortfall. Selected oracle: 41/141, 68 slot shortfall. Both use one-hop fallback and retain equal budget. |
| Reranker variability estimate | Mean absolute repeat difference in NDCG@5: 0.0775 (1-hop), 0.0746 (naive), 0.0722 (oracle); report uses each user's two-repeat mean. |
| API/GPU cost and wall time | Canonical raw cache: 4,512 successful records, SHA256 `b019ba…39f0`; Azure price unavailable (USD unknown). Last resumed invocation: 1,188.3 s / 3,306,725 reranker tokens only; do not interpret this as full-run token total. |

### Gate decision

| Criterion | Result |
|---|---|
| Stop: Delta_oracle <= +0.02 or CI upper <= +0.03 | **Thỏa điều kiện dừng:** −0.0094 ≤ +0.02 (CI upper +0.0176 ≤ +0.03). |
| Borderline: CI crosses 0 or +0.02 < Delta_oracle < +0.05 | CI cắt 0, nhưng hard-stop được ưu tiên vì point estimate thấp hơn +0.02. |
| Proceed MH3: Delta_oracle >= +0.05 and CI lower > +0.01 | **Không đạt.** |
| Decision and rationale | **Dừng sau MH2.** Với cùng K_u/token budget, context xa hơn nhưng candidate-blind không cải thiện frozen ranker ngay cả khi dùng bounded oracle. MH3/MH4/MH5 không được admission. |

## 5. MH3 validation — selector after config lock

| Arm | Selector config hash | N users | NDCG@5 | Delta vs 1-hop | Delta vs naive | 95% CI vs 1-hop | Oracle capture | Budget pass | Status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| one_hop | control | — | — | — | — | — | — | — | not admitted |
| naive_two_hop | fixed traversal | — | — | — | — | — | — | — | not admitted |
| selective_two_hop | — | — | — | — | — | — | — | — | not admitted |

### Selector diagnostics

| Measure | Value |
|---|---|
| alpha, beta, gamma, q | — |
| Semantic/path/redundancy score distributions | — |
| Remote item/user mix | — |
| Degree/hub distribution vs 1-hop | — |
| Duplicate/redundant node rate | — |
| 30-user qualitative audit | — |

## 6. MH4 locked test

> Fill only after validation config and manifest hash are frozen. No parameter may
> be changed after looking at this table.

| Arm | N users | H@1 | H@3 | NDCG@3 | H@5 | **NDCG@5** | Paired Delta vs 1-hop | 95% CI | Tokens/query | Latency/query | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| one_hop | — | — | — | — | — | — | — | — | — | — | not admitted |
| naive_two_hop | — | — | — | — | — | — | — | — | — | — | not admitted |
| selective_two_hop | — | — | — | — | — | — | — | — | — | — | not admitted |

### Test robustness and breakdown

| Slice / check | 1-hop | naive 2-hop | selective 2-hop | Finding |
|---|---:|---:|---:|---|
| Short vs long history | — | — | — | — |
| Low vs high graph degree | — | — | — | — |
| Remote quota | — | — | — | — |
| Remote path type | — | — | — | — |
| Budget equality / shortfall | — | — | — | — |
| Candidate/gold leakage audit | — | — | — | — |

## 7. MH5 selective propagation (conditional)

| Arm | Endpoint cap | Coverage | Duplicate/stale rate | NDCG@5 Delta | Tokens/latency delta | Temporal leakage | Status |
|---|---:|---:|---:|---:|---:|---|---|
| one_hop_write | — | — | — | — | — | — | not admitted |
| naive_two_hop_write | — | — | — | — | — | — | not admitted |
| selective_two_hop_write | — | — | — | — | — | — | not admitted |

## 8. Final conclusion

**Không có multi-hop headroom có ý nghĩa dưới equal budget; dừng sau MH2.**
Independent report pass đặt oracle two-hop ở −0.0094 NDCG@5 so với one-hop
(95% CI [−0.0362, +0.0176]), nên protocol này không bảo đảm để làm selector,
locked test hoặc propagation claim.
