# PROGRESS.md — Nhật ký Selective Multi-Hop

> **Kế hoạch active:** [MULTIHOP_PLAN.md](MULTIHOP_PLAN.md). Tổng kết hướng đã
> đóng: [RL_WORK_SUMMARY.md](RL_WORK_SUMMARY.md).
>
> Quy tắc: ghi entry ngay sau mỗi milestone/run; không diễn giải kết quả chưa có
> artifact. Mọi command, commit, hash và cost phải đủ để tái lập run.

## Trạng thái hiện tại

| Milestone | Trạng thái | Quyết định / đầu ra |
|---|---|---|
| MH0 — Freeze protocol | ✅ complete (offline) | Topology và 2,327 one-hop controls đã materialize, có manifest/hash và guard candidate-blind. Chưa có API call. |
| MH1 — 2-hop pool + bundles | ✅ complete (offline) | Bounded C2 pools train/val/test và 12 oracle bundle/validation user đã materialize; MH2 dùng cohort 141 user eligible. |
| MH2 — Oracle headroom (val) | ✅ complete — hard stop | Oracle 2-hop không vượt 1-hop: ΔNDCG@5 = −0.0094, 95% CI [−0.0362, +0.0176]. |
| MH3 — Selective selector | ⛔ not admitted | MH2 không qua headroom gate; không tune selector sau khi đã thấy kết quả. |
| MH4 — Locked test | ⛔ not admitted | Không có config MH3 được admission trên validation. |
| MH5 — Selective propagation | ⛔ not admitted | Read-side 2-hop không có headroom thực dụng dưới budget cố định. |

## MH0 — Freeze protocol — 2026-08-25

**Đã làm**

- Đọc code/datasets/docs của repo và đóng hướng SFT/RL trong
  RL_WORK_SUMMARY.md.
- Viết experiment contract, oracle protocol và gate cho multi-hop trong
  MULTIHOP_PLAN.md.
- Khởi tạo bảng kết quả trống không trộn baseline M0 RNG theo thread với candidate
  set fixed của protocol mới.
- Thêm `src/multihop/mh0.py`: dựng lại topology từ interaction history trước
  target, check split/leakage/prompt, rồi materialize control candidate-blind.
- Chạy `python -m src.multihop.mh0 --config configs/multihop/mh0_books.yaml`:
  profile input `legacy_pre_m2_backfill`; topology 7,377 users / 111,084 items /
  193,005 edges; control train/val/test = 1,185 / 149 / 993.
- Ghi artifact gitignored ở `data/multihop/`: `mh0_topology_books.json`, ba
  `mh0_control_*.jsonl` và `mh0_manifest.json`. Topology SHA256:
  `75744eb2bae77bbcc463208d557eb528fcaa3475d86d2914b2b19456840b3180`.
- Thêm config active `configs/multihop/mh0_books.yaml` và adapter Azure cho
  `LLM__*`; test unit bảo đảm `azure/gpt-5.4-mini` được gửi qua Azure SDK với
  deployment `gpt-5.4-mini`.
- Chạy `pytest tests/test_llm_client_azure.py tests/multihop/test_mh0.py -q`:
  **5 passed**. `MH0` không tạo LLM request và API cost = 0.
- `pytest tests/rl -q` chưa collect được vì environment Python 3.13 hiện thiếu
  package `torch` (lỗi import ở `src.rl.validate_reward`); đây không chặn MH0
  vì command và test của MH0 không import torch.

**Đã pass gate MH0**

- [x] Validate bundle nhận diện được hash profile, user split disjoint, fixed
  10-candidate list và source prompt không leak gold.
- [x] Dựng/hash topology từ pre-target interaction history; ghi snapshot, jsonl,
  config và output hash vào manifest.
- [x] Re-materialize exact one_hop control; lưu per-user `K_u`, `T_u`, node IDs
  và candidate-order hash; Stage-R context tách hẳn ranking context.
- [x] Rerun bằng `--force`; topology và ba control file giữ nguyên SHA256.

**Lưu ý cấu hình trước remote run:** endpoint, API version và model mới đã có
trong `.env`; `LLM__API_KEY` cố ý để trống, nên
`python scripts/check_llm_config.py` dừng an toàn trước khi gửi request. Cần điền
key thật; sau đó chạy script này và chỉ dùng `--request` khi muốn thực hiện smoke
call có chủ đích.

**Gate:** pass. Có thể bắt đầu MH1 offline; bất kỳ call LLM nào vẫn phải qua
configuration smoke check ở trên.

## MH1 — 2-hop pool + bundle generator — 2026-08-25

**Đã làm**

- Thêm `src/multihop/mh1.py`: mở rộng từ peer user đã được MH0 pack, giữ witness
  path, path strength, static snippet bằng chính `SnippetPacker`, và tách hoàn
  toàn Stage-R construction khỏi ranking context.
- Khóa structural cap candidate-blind trong config: tối đa 128 remote item, 64
  remote user và 8 remote user/remote item. Thống kê pool trung bình:
  train 120.79 item + 57.57 user; val 120.53 + 59.32; test 122.73 + 59.10.
- Materialize `mh1_pools_{train,val,test}.jsonl` cho 1,185 / 149 / 993 user;
  materialize `mh1_bundles_val.jsonl` với 3 naive bundle và **12 oracle
  bundle/user** (4 cho mỗi quota 2/4/6).
- Pass all exact `K_u`/`T_u` assertions. Remote node không vừa token cap được
  thay bằng one-hop fallback: 763 / 2,235 bundle có shortfall, được ghi per
  bundle; không có arm nào vượt budget.
- Leakage audit hai pha (toàn C2 rồi bundle) loại **cùng lúc mọi arm** cho 8/149
  validation user (8 gold-ID event, 7 duplicate-gold-title event); cohort MH2
  bị khóa còn 141 user. Không có candidate/instruction/gold được dùng để xây
  pool; guard không xóa target-aware từng node.
- Rerun `--force` giữ nguyên SHA256 cho 3 pool và bundle validation. Chạy
  `pytest tests/multihop tests/test_llm_client_azure.py -q`: **9 passed**;
  LLM calls = 0, API cost = USD 0.

| Field | Điền sau |
|---|---|
| Ngày / commit | 2026-08-25 / working tree trước commit |
| Input manifest hash | MH0 topology `75744e…b3180`; MH1 bundle `7b846a…1458` |
| Command | `python -m src.multihop.mh1 --config configs/multihop/mh0_books.yaml --force` |
| Unit tests | 9 passed |
| Users smoke / val | 149 val; 141 eligible all arms |
| Pool coverage / shortfall | 0 val pool dưới quota 6; 763/2,235 bundle remote-shortfall, đều fallback one-hop |
| Budget assertion | K exact, T_actual ≤ T_u for all 2,235 bundle |
| Compute / API cost | CPU-only / USD 0 |
| Quyết định | **Pass MH1.** Chỉ MH2 trên 141-user locked cohort; report quota/shortfall. |

## MH2 — Oracle headroom validation — 2026-08-25

**Kết luận:** dừng hướng selective multi-hop trong protocol hiện tại. Ngay cả
bounded oracle (chọn sau selection pass bằng NDCG@5) không vượt 1-hop trong
report pass độc lập, nên không có lý do hợp lệ để tune MH3, chạy test, hay mở
write-side routing.

| Field | Kết quả đã materialize |
|---|---|
| Ngày / commit / run ID | 2026-08-25 / `4268390d626bbd094901174f4b0f4d853814c486` / `mh2_val_books_gpt54mini` |
| Validation records analysed | 141 / 141 locked users; không drop user |
| Reranker + settings | Azure `azure/gpt-5.4-mini` (deployment `gpt-5.4-mini`), API `2024-05-01-preview`; JSON schema, candidate order fixed, max completion 2,048 |
| Bundles/user, quota q | 1 one-hop + naive q=4 + 12 oracle bundles (4 mỗi q∈{2,4,6}) |
| Selection pass artifact | 1,692 rerank = 141 × 12; chỉ để chọn oracle, không dùng làm score report |
| Independent report pass count | 846 rerank = 141 × 3 arm × 2 repeats; đây là nguồn duy nhất của bảng kết quả |
| Delta_naive NDCG@5 + 95% CI | −0.0193 [−0.0476, +0.0067] |
| Delta_oracle NDCG@5 + 95% CI | **−0.0094 [−0.0362, +0.0176]** |
| Budget/coverage audit | Mọi arm giữ exact K_u (mean 15.66) và T_actual ≤ T_u. Naive q=4: mean remote 3.39, shortfall 42 user / 86 slot; oracle chọn: mean remote 1.90, shortfall 41 / 68. Shortfall luôn fill one-hop. |
| Cost / wall time | Canonical cache: 4,512 successful calls, SHA256 `b019ba…39f0`; Azure price không được cung cấp nên USD = unknown. Resume cuối mất 1,188.3 s và ghi 3,306,725 reranker token; số này **không gồm** calls của invocation trước. 75 selection response trùng do một resume chồng đã bị loại khỏi cache trước phân tích. |
| **Gate decision** | **Hard stop:** Δ_oracle ≤ +0.02. Không thực hiện MH3–MH5. |

## MH3 — Selective selector — not admitted

| Field | Điền sau |
|---|---|
| Frozen config hash (alpha, beta, gamma, q) | — |
| Validation delta vs 1-hop / naive | — |
| Oracle capture | — |
| Qualitative audit | — |
| Leakage/budget checks | — |
| Test admission decision | — |

## MH4 — Locked test — not admitted

| Field | Điền sau |
|---|---|
| Locked config/manifest hash before run | — |
| Test users analysed | — |
| Paired Delta NDCG@5 + 95% CI | — |
| Secondary metrics | — |
| Token/latency delta | — |
| Failure-bucket analysis | — |
| Final conclusion | — |

## MH5 — Selective propagation — not admitted

| Field | Điền sau |
|---|---|
| Admission evidence from MH4 | — |
| Endpoint cap / coverage | — |
| Temporal leakage checks | — |
| Ranking and cost results | — |
| Conclusion | — |
