# RESULTS.md — Selective Multi-Hop Collaborative Memory

> **Protocol registry:** [MULTIHOP_PLAN.md](MULTIHOP_PLAN.md),
> [CANDIDATE_EVIDENCE_PLAN.md](CANDIDATE_EVIDENCE_PLAN.md), và
> [BUFFERED_PROPAGATION_PLAN.md](BUFFERED_PROPAGATION_PLAN.md), và
> [AMAZON_BOOKS_2014_TEMPORAL_PLAN.md](AMAZON_BOOKS_2014_TEMPORAL_PLAN.md).
> Amazon Books 2014 P2-v1 was stopped for a reranker output-contract error.
> Its separate P2-v2 smoke-first rerun completed and failed the oracle headroom
> gate. P3 self-host 3-hop cũng đã complete và fail gate; no candidate-blind
> implementation is admitted. P4 filtered 3-hop tăng rất nhẹ nhưng cũng fail.
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
| Active result status | InstructRec MH2/CE1 và write-side feasibility đều closed. Amazon Books P4 filtered 3-hop chỉ +0.0186 vs local và +0.0031 vs raw 3-hop; hard stop. |

MH2 kết luận riêng cho graph expansion. Candidate-conditioned 1-hop evidence là
một protocol ranking-time mới, được preregister tại
[CANDIDATE_EVIDENCE_PLAN.md](CANDIDATE_EVIDENCE_PLAN.md); không được gộp số với
bảng multi-hop bên dưới.

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

**Read-side multi-hop không có headroom có ý nghĩa dưới equal budget; dừng sau
MH2.** Independent report pass đặt oracle two-hop ở −0.0094 NDCG@5 so với
one-hop (95% CI [−0.0362, +0.0176]), nên protocol này không bảo đảm để làm
selector hoặc locked test. Kết luận đó không được diễn giải tự động thành
write-side propagation failure; feasibility của write-side được audit tách ở
section 10.

## 9. Candidate-conditioned 1-hop evidence (CE)

| Arm | Pilot users | NDCG@5 | Delta vs baseline | 95% CI | Evidence slots / cap | Status |
|---|---:|---:|---:|---|---|---|
| baseline_one_hop | 100 | 0.7847 | reference | — | 0 / 384 | complete |
| request_one_hop | 100 | 0.7877 | +0.0030 | [−0.0329, +0.0370] | 10 total / 384 | complete |
| candidate_one_hop | 100 | 0.7767 | **−0.0080** | **[−0.0434, +0.0281]** | 1 per candidate / 384 | complete — stop |

CE uses only frozen 1-hop snippets and exact MH2 one-hop Stage-R facets. It is
not evidence that multi-hop expansion works or fails beyond the MH2 conclusion.
CE0 artifact SHA256 is `b7b296df71769da578f8191f4274c44c53dc4b073e0dd64a0b9c7bfc8504a266`.

| CE1 gate / audit | Result |
|---|---|
| Candidate vs request | −0.0110 NDCG@5, 95% CI [−0.0464, +0.0245] |
| Admission requirement | Candidate − request ≥ +0.03 with CI lower > 0, and candidate − baseline ≥ +0.02 |
| Decision | **Fail; CE2 not admitted.** No post-hoc lexical score/budget/selector tuning. |
| Integrity | 600 canonical unique successful reranks; 100/100 analysed. Two model format errors were archived then retried by identical key/prompt; retry ledger retained. |

## 10. Buffered item-side propagation feasibility

Protocol chi tiết: [BUFFERED_PROPAGATION_PLAN.md](BUFFERED_PROPAGATION_PLAN.md).
Đây là pivot Stage-W khác MH2: Stage-R/ranker 1-hop không đổi; chỉ một
candidate-blind, source-only item overlay mới có thể được xét sau feasibility.
Không có LLM call hoặc item-memory write nào đã chạy.

| Probe | Result | Gate / decision |
|---|---|---|
| P0 temporal audit | 207,759 events / 7,377 users; 1,479 timestamp values shared across users; 207,012 events cross-user ambiguous; 7,376 file-order inversions; all per-user histories monotonic | **Dynamic hard stop.** Không có strict global cross-user event clock, nên không replay/claim asynchronous propagation. |
| P1 static source-only ledger | 1,185 source vs 149 val users; route cap 8 anchor × 16 peer × 16 remote item; 10,455 endpoint items | Candidate/gold only read posthoc; ledger candidate-blind, no LLM. |
| P1 support >=2 coverage | 60/1,490 candidate slots (4.03%); 9/149 gold (6.04%); 46/149 users with any candidate endpoint (30.87%) | **Static hard stop.** Gold coverage misses 30% feasibility gate by a large margin; no P2 oracle/implementation. |

P1 is an offline reconnaissance run, not a preregistered positive/negative
ranking test. The decision gate was recorded before any P2 LLM request. Further
progress requires a dataset/protocol with a reliable global event order, then a
new P0 and candidate-blind item-side oracle; raising hop/path caps after this
audit is not an admitted continuation.

## 11. Amazon Books 2014 temporal P0/P1

Protocol: [AMAZON_BOOKS_2014_TEMPORAL_PLAN.md](AMAZON_BOOKS_2014_TEMPORAL_PLAN.md).
This is a new global-time, item-only write-side pilot; it does not revise the
negative InstructRec findings above. It uses strict-past *timestamp batches*,
because the source only identifies the review date and has tied events.

| P0 check | Result |
|---|---|
| Raw / usable events | 3,000,000 / 2,438,194 (81.27%) |
| Dropped rows | 561,787 missing `User_id`; 19 invalid timestamp; no imputation |
| Time range / resolution | 1996-08-17 to 2013-03-04; 5,736 timestamps, 2,438,160 tied events |
| Frozen absolute splits | train `<2011-10-30`: 1,950,481; val: 242,896; test: 244,817 |
| Metadata guard | 212,403 exact-unique nonempty titles; exact title coverage 99.992%; 195 usable review rows titleless |
| Decision | **Pass only for strict-past daily-batch causal replay; no within-day event ordering claim.** |

| P1 candidate-blind source ledger | Result |
|---|---|
| Pilot / source cohort | 21,182 hash-sampled users; 17,037 train-history; 1,282 with >=5 train events; 560 selected sources |
| Route / output | source -> <=3 anchors -> <=16 peers -> <=16 remote items; 4,357 endpoint items |
| Fixed future evaluation cohort | 100 deterministic novel validation events; selected after ledger materialization |
| Gold coverage support >=1 / >=2 / >=3 | 24% / 17% / 12% |
| P1 decision | **Pass:** source count 560, support>=1 >=20%, support>=2 >=10% |
| LLM/API use | **0 calls** |

P2 preparation locks 100 ten-item candidate lists and strict-past user histories
without using an LLM. All lists are unique and contain gold exactly once; no
history is at/after target time and no gold is in that history. It also shows a
material coverage constraint: direct-anchor 1-hop packets alter 13 candidate
slots across 12 events (3 gold), while support>=2 routes reach 17 gold
endpoints. The two-hop arm may enrich only those gold endpoints after labels are
read, so it is a non-deployable oracle headroom test rather than a selector.

The P2 execution journal was stopped at 768 attempts: 560 semantic packet and
100 Stage-R calls succeeded, but 90 rerank outputs were truncated while the
old strict schema required both ranking and free-text rationale under a
180-token cap. Only 10 reranks parsed, which is not a complete arm/event matrix
and is deliberately excluded from every metric and gate. The remaining original
allocation must not be repurposed to tune or resume the run.

P2-v2 uses that corrected ranking-only contract, with a 20-event smoke cohort
executed before its full request run. It is a new run ID/journal and does not
reuse v1 ranking outputs.

| P2-v2 arm (100 fixed events) | NDCG@5 | Δ vs local | 95% paired bootstrap CI | H@5 |
|---|---:|---:|---:|---:|
| local | 0.6474 | reference | — | 0.84 |
| direct 1-hop | 0.6353 | −0.0121 | [−0.0432, +0.0179] | 0.84 |
| target-aware oracle 2-hop | 0.6799 | **+0.0324** | **[+0.0073, +0.0592]** | 0.87 |

P2-v2 used 960 primary and one identical retry (961/1,000 attempts); all 300
final rankings were valid. The retry corrected a duplicate/missing label for a
single rerank key. The paired bootstrap uses 10,000 resamples with seed
`20260826`; its artifact is `p2_v2_analysis.json`, SHA256
`2808d5e491105637d005810f73576f1917a3c932680059e99601af1450991d30`.

**Decision: hard stop.** The target-aware oracle is statistically positive but
is only +0.0324, below the pre-recorded +0.05 gate, and reaches only the 17
support>=2 gold endpoints. It is not a candidate-blind method. No router,
write-buffer implementation, selector tuning, or locked test is admitted.

## 12. Amazon Books 2014 bounded 3-hop oracle

Protocol và execution audit: [THREE_HOP_ORACLE_PLAN.md](THREE_HOP_ORACLE_PLAN.md).
P3 rerun cả ba arm bằng cùng self-host checkpoint, candidate order và Stage-R;
không so score tuyệt đối với Azure P2-v2.

| Arm (100 fixed events) | NDCG@5 | Δ vs local | 95% paired bootstrap CI | H@5 |
|---|---:|---:|---:|---:|
| local | 0,5752 | reference | — | 0,76 |
| target-aware oracle 2-hop | 0,5893 | +0,0140 | [−0,0007; +0,0338] | 0,79 |
| target-aware oracle 3-hop | 0,5907 | **+0,0155** | **[−0,0117; +0,0465]** | 0,80 |

Oracle 3-hop vs oracle 2-hop là **+0,0014**, CI [−0,0188; +0,0208]. Nó cải
thiện 4, làm tệ 4 và giữ nguyên 92 event so với 2-hop. Structural support>=2
tăng 17%→24%, nhưng bảy gold chỉ có ở 3-hop có mean delta −0,0609; coverage xa
hơn tạo noise thay vì headroom. Diagnostic post-hoc chọn arm tốt nhất cho từng
event chỉ đạt +0,0279 vs local, vẫn dưới +0,05 và không phải selector hợp lệ.

Execution dùng `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`: smoke 116/116; full journal
960 primary, 0 retry, 0 repair. Full invocation peak VRAM 7,85 GiB trên một
H100; GPU được nhả về 1 MiB ngay khi process thoát. Metrics SHA256 là
`3ab1d1a6f754473bb24d8190347bff9f023c4a49a16d3c908bc9c50a40fd2784`.

**Decision: hard stop.** Không tăng tiếp 4-hop/n-hop bằng cùng
source-packet/item-overlay. Một hướng multi-hop mới chỉ đáng xét nếu thay đổi
candidate-level representation/scoring và có residual gate giữ local signal,
không phải chỉ mở rộng depth.

## 13. Amazon Books 2014 filtered 3-hop

Protocol: [FILTERED_THREE_HOP_PLAN.md](FILTERED_THREE_HOP_PLAN.md). Filter chỉ
giữ edge rating>=4, thêm recency half-life 730 ngày, hub penalty và bridge-path
diversity. Structural support>=2 giảm hợp lý từ raw 24% xuống 19%.

| Arm | NDCG@5 | Δ vs local | 95% paired bootstrap CI | H@5 |
|---|---:|---:|---:|---:|
| local frozen P3 | 0,5752 | reference | — | 0,76 |
| raw 3-hop frozen P3 | 0,5907 | +0,0155 | [−0,0117; +0,0465] | 0,80 |
| filtered 3-hop | 0,5939 | **+0,0186** | **[−0,0021; +0,0425]** | 0,80 |

Filtered vs raw chỉ +0,0031, CI [−0,0160; +0,0238]. Trên 19 supported event,
filter tốt hơn raw +0,0165, nhưng vẫn cải thiện 5 và làm tệ 4 event. Ngay cả
post-hoc best-of local/raw/filtered chỉ +0,0370 vs local, dưới point gate +0,05.

P4 reuse frozen P3 output và chỉ tạo 100 filtered ranking: 100/100 primary
success, zero retry/repair. Peak VRAM 7,85 GiB trên một H100; card được nhả về
1 MiB ngay sau task. Metrics SHA256:
`4ec4305354b186f94e0d92c74ba4e01407a33ac847ccb9ec691c2ea1ef79a910`.

**Decision: hard stop.** Filter quality có tín hiệu nhỏ nhưng không đủ headroom
và không ổn định. Không tune threshold post-hoc; packet overlay không được
admit. Nếu tiếp tục graph, cần candidate-level scoring/retrieval protocol mới.
