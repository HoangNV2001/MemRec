# PROGRESS.md — Nhật ký Selective Multi-Hop

> **Các protocol đã audit:** [MULTIHOP_PLAN.md](MULTIHOP_PLAN.md),
> [CANDIDATE_EVIDENCE_PLAN.md](CANDIDATE_EVIDENCE_PLAN.md), và
> [BUFFERED_PROPAGATION_PLAN.md](BUFFERED_PROPAGATION_PLAN.md). Pilot temporal
> mới: [AMAZON_BOOKS_2014_TEMPORAL_PLAN.md](AMAZON_BOOKS_2014_TEMPORAL_PLAN.md).
> Tổng kết hướng SFT/RL đã đóng: [RL_WORK_SUMMARY.md](RL_WORK_SUMMARY.md).
>
> Quy tắc: ghi entry ngay sau mỗi milestone/run; không diễn giải kết quả chưa có
> artifact. Mọi command, commit, hash và cost phải đủ để tái lập run.

> MH2 và CE1 là hard-stop cho hai giả thuyết read-side/ranking-time trên
> InstructRec. Buffered propagation trên InstructRec dừng ở P0/P1. Amazon Books
> 2014 temporal P2-v1 bị dừng vì output contract; P2-v2 smoke-first đã hoàn tất
> 100 event nhưng không qua oracle headroom gate.

## Trạng thái hiện tại

| Milestone | Trạng thái | Quyết định / đầu ra |
|---|---|---|
| MH0 — Freeze protocol | ✅ complete (offline) | Topology và 2,327 one-hop controls đã materialize, có manifest/hash và guard candidate-blind. Chưa có API call. |
| MH1 — 2-hop pool + bundles | ✅ complete (offline) | Bounded C2 pools train/val/test và 12 oracle bundle/validation user đã materialize; MH2 dùng cohort 141 user eligible. |
| MH2 — Oracle headroom (val) | ✅ complete — hard stop | Oracle 2-hop không vượt 1-hop: ΔNDCG@5 = −0.0094, 95% CI [−0.0362, +0.0176]. |
| MH3 — Selective selector | ⛔ not admitted | MH2 không qua headroom gate; không tune selector sau khi đã thấy kết quả. |
| MH4 — Locked test | ⛔ not admitted | Không có config MH3 được admission trên validation. |
| MH5 — Selective propagation | ⛔ not admitted | Read-side 2-hop không có headroom thực dụng dưới budget cố định. |
| CE0 — 1-hop evidence materialization | ✅ complete (offline) | 100 deterministic row, 384-token cap, artifact hash/materialization checks pass; 0 API call. |
| CE1 — Evidence reranking pilot | ✅ complete — hard stop | Candidate evidence không vượt baseline/request; 100/100 user, report độc lập. |
| CE2 — Full validation | ⛔ not admitted | CE1 không qua gate đã khóa; không tune selector/budget rồi rerun. |
| P0 — Temporal propagation audit | ✅ complete — dynamic hard stop | Timestamp chỉ có thứ tự theo user: 207,012/207,759 event cross-user ambiguous; không causal replay Stage-W. |
| P1 — Source-only item route coverage | ✅ complete — static hard stop | Với 8 anchor, 16 peer, 16 endpoint: gold support>=2 chỉ 9/149 (6.04%), không admission P2. |
| P2–P4 — Buffered item propagation | ⛔ not admitted | Không gọi LLM, không ghi memory, không tuning path cap/score sau P1. Đổi dataset/protocol có global event time nếu tiếp tục. |
| AB14 P0 — Temporal data audit | ✅ complete — pass batched replay | 2,438,194 usable events; global Unix day timeline. Rows userless bị loại; same-day events dùng strict-past batch semantics. |
| AB14 P1 — Packet/route feasibility | ✅ complete — P2 admitted | 560 source packet, 100 fixed val targets; coverage gold support>=1/2 = 24%/17%; 0 LLM call. |
| AB14 P2-v1 — 1k-request item-only oracle | ⛔ invalid execution — sealed | Journal 768 attempts: rerank schema bị truncate 90 response. Không metric/gate; không resume hay trộn partial output. |
| AB14 P2-v2 — smoke-first oracle rerun | ✅ complete — hard stop | 100 event/300 final rerank; oracle ΔNDCG@5 = +0.0324, 95% CI [+0.0073, +0.0592], dưới gate +0.05. |
| AB14 P3 — bounded 3-hop self-host oracle | ✅ complete — hard stop | Coverage support>=2 tăng 17%→24%, nhưng 3-hop chỉ +0.0155 vs local và +0.0014 vs 2-hop; cả hai CI cắt 0. Không tăng tiếp n-hop bằng cùng item-overlay. |
| AB14 P4 — filtered 3-hop | ✅ complete — hard stop | Filter giữ support>=2 ở 19%; NDCG@5 0.5939, chỉ +0.0186 vs local và +0.0031 vs raw 3-hop, cả hai CI cắt 0. |
| AB14 P5/P5b — candidate graph 3/5-layer | ✅ complete — hard stop | Full graph tăng gold coverage 15%→30% nhưng negative evidence 2.33%→13.78%; residual 5-layer +0.0077 vs local và −0.0019 vs 3-layer. |
| AB14 P6 — learned adaptive multi-hop | ✅ complete — hard stop | Validation +0.0265 không transfer: fresh test adaptive 0.5672 vs local 0.5897 (−0.0225, CI [−0.0559,+0.0113]). Oracle best-of còn +0.0305 nhưng gate sai 22 event. |
| AB14 P7 — temporal transition PPR | 🟡 running — fresh test admitted | Directed next-item graph; calibration PPR residual 0.7088 vs local 0.5897 (+0.1191), alpha 0.8 locked. Fresh labels chưa mở. |

## AB14 — Amazon Books 2014 temporal P0/P1 — 2026-08-26

- New raw input is `data/Books_rating.csv` and `data/books_data.csv`, not
  InstructRec. P0 streams and hashes both file inputs; derived artifacts remain
  gitignored in `data/temporal_amazon_books_2014/`.
- P0 finds a valid global **daily** timeline, 1996-08-17 to 2013-03-04. It
  drops exactly 561,787 rows missing `User_id` and 19 invalid-time rows, then
  locks absolute 80/10/10 cutoffs: train 1,950,481, val 242,896, test 244,817.
  Tied timestamps never receive an invented file-order tiebreak.
- Exact review-title to metadata-title matching covers 99.992% of usable rows;
  the 195 titleless rows cannot consume metadata. P2 must preserve this guard.
- P1 uses a deterministic `blake2b % 48 == 0` user sample. It selects 560
  source users (>=5 train events), creates a strict-past source-to-item ledger
  with 4,357 endpoints, then separately measures 100 fixed future targets.
- Final coverage is support>=1 24/100 and support>=2 17/100, both above frozen
  20%/10% feasibility gates. P1 manifest SHA256 is
  `c02426ec5f07f01cef8652f93a610bfb37afc7668fcf3fa5b019f52e3de804ec`.
  API requests and memory writes: **0**.
- P2 preparation freezes 100 ten-item candidate lists and their strict-past
  histories (`p2_prepared.json` SHA256
  `e138e9ba9df56a0ae1b97974a35008442c669e71729dca22722b9813cf745653`).
  It passes duplicate/gold/history-time checks. The direct-anchor 1-hop overlay
  changes 13 candidate slots in 12 events (3 gold); the support>=2 pool reaches
  17 gold endpoints. The latter remains explicitly target-aware oracle-only.

## AB14 — P2 execution incident — 2026-08-26

- Lệnh chạy `python -m src.temporal_books.p2_oracle --config
  configs/temporal_amazon_books_2014/pilot.yaml --request` ghi 768 attempt vào
  `p2_attempts.jsonl` trước khi được dừng an toàn: 560 semantic packet và 100
  Stage-R đều thành công; rerank có 10 JSON hợp lệ, 90 JSON bị cắt và 8 attempt
  đã được journal nhưng chưa có response khi tiến trình bị dừng.
- Lỗi là `JSONDecodeError: Unterminated string`: schema rerank cũ bắt buộc cả
  ranking lẫn rationale tự do, vượt output cap 180 token. 10 ranking lẻ không
  được tính metric hoặc dùng chọn cohort.
- Runner được sửa thành schema chỉ có `ranking` và submit theo từng worker
  batch; nếu lỗi lặp lại thì dừng ngay sau batch + identical retry thay vì queue
  cả phase. Vì output contract đã đổi, journal P2 cũ bị đóng, không resume.
- Cần người dùng cấp **budget mới/run id mới** nếu muốn chạy clean rerun; không
  được vượt allocation 1,000 request của P2 cũ.

## AB14 — P2-v2 smoke-first contract — 2026-08-26

- Config riêng `configs/temporal_amazon_books_2014/p2_v2_smoke.yaml` dùng run
  ID/journal/artifact mới; không thể chạm journal P2 v1 bị sealed.
- `--request` bị block cho đến khi `--smoke-only` thành công. Smoke được khóa
  theo stable hash với 20 event; phải materialize thêm mọi source packet mà
  rerank prompt của 20 event này thực sự đọc, thành 46 packet + 20 Stage-R + 60
  rerank = 126 request. Các key đó được reuse ở full run nên quota P2-v2 vẫn là
  960 primary + 40 retry, không cộng thêm smoke quota.
- Reranker v2 trả strict JSON `ranking` duy nhất (không rationale tự do); runner
  submit theo worker batch và fail-fast sau identical retry nếu format lỗi.
- Smoke đã pass **126/126** request (46 packet, 20 Stage-R, 60 rerank), zero
  retry. Manifest khóa config/prepared hash và cache key; full P2-v2 chỉ còn
  514 packet + 80 Stage-R + 240 rerank = 834 primary request.

## AB14 — P2-v2 result — 2026-08-26

- Full run hoàn tất 960 primary + 1 identical retry = **961/1,000** attempts.
  Một rerank trả duplicate/missing label bị validator bắt, retry cùng key/prompt
  thành công; 300 final rerank đều hợp lệ. Artifact: `p2_v2_manifest.json`
  SHA256 `9c51700095c26d99d9adb244c8a40d78682e1f2af4766451ee74155289a977a4`.
- Local NDCG@5 = 0.6474. Direct 1-hop = 0.6353, Δ = −0.0121, paired bootstrap
  95% CI [−0.0432, +0.0179]. Oracle 2-hop = 0.6799, Δ = **+0.0324**, CI
  **[+0.0073, +0.0592]**, H@5 0.87 vs 0.84 local.
- Bootstrap deterministic 10,000 paired resample (seed `20260826`), analysis
  SHA256 `2808d5e491105637d005810f73576f1917a3c932680059e99601af1450991d30`.
  Oracle improves 19, worsens 6, and leaves 75/100 event unchanged; it only
  touches the 17 support>=2 gold endpoints and is target-aware/non-deployable.
- **Gate decision: stop.** Although oracle CI lower > 0, Δ=+0.0324 < frozen
  +0.05 threshold. Không admission candidate-blind router, buffer write,
  selector tuning, hoặc locked test cho cơ chế này.
- A first 1/128 sample was below the already-stated 10K–30K pilot cohort target
  (7,915 users; 476 sources; 43 targets), so it was discarded before admission
  and replaced with 1/48. This sizing correction did not alter route caps,
  coverage thresholds or spend any LLM call.

## AB14 — P3 bounded 3-hop self-host result — 2026-09-21

- Structural ledger candidate-blind mở route
  `source→anchor→peer1→bridge→peer2→endpoint`, mọi edge strict `< packet_time`.
  Với cùng 560 source packet, support>=2 gold tăng từ 17/100 ở 2-hop lên
  24/100 ở 3-hop; ledger được serialize trước khi mở validation label.
- Self-host model là `Qwen/Qwen3-4B-Instruct-2507`, exact revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, BF16/greedy và JSON-schema
  constrained. Smoke v4 pass 116/116; full reuse cache và kết thúc 960/960
  primary success, 0 retry, 0 repair.
- Local / oracle 2-hop / oracle 3-hop NDCG@5 lần lượt là 0,5752 / 0,5893 /
  0,5907. 3-hop vs local = **+0,0155**, CI [−0,0117; +0,0465]; 3-hop vs
  2-hop = **+0,0014**, CI [−0,0188; +0,0208]. Gate +0,05 và +0,02 đều fail.
- Bảy gold chỉ được 3-hop chạm có mean delta −0,0609; coverage thêm chủ yếu
  đưa noise vào ranker. Ngay cả post-hoc best-of-local/2/3 diagnostic cũng chỉ
  +0,0279, dưới gate +0,05; số này không phải preregistered result.
- Một H100 được dùng, peak 7,85 GiB; model process thoát ngay sau task và GPU 0
  về 1 MiB. Completion manifest SHA256:
  `dee6ecc05cf7f97d2f2989f799911e0df5d96fac11aeb9353048f7d7b78e4edd`.
- **Quyết định:** hard stop cho việc chỉ tăng 4-hop/n-hop trên cùng
  source-packet/item-overlay. Hướng multi-hop tiếp theo, nếu có, phải là protocol
  mới thay đổi representation/scoring thay vì tăng depth.

## AB14 — P4 filtered 3-hop result — 2026-09-21

- Candidate-blind filter chỉ giữ rating>=4 edge; xếp path bằng temporal decay
  half-life 730 ngày, anchor/bridge hub penalty và tối đa ba distinct bridge
  path/origin. Structural smoke 24 source pass; full support>=2 = 19/100.
- P4 pin và reuse exact P3 packets/Stage-R/local/raw rankings. Nó chỉ sinh 100
  filtered rerank: smoke 20 được reuse trong full, 100 primary success, 0 retry,
  0 repair.
- Filtered NDCG@5 = 0,5939, H@5 = 0,80. Delta vs local = +0,0186, CI
  [−0,0021; +0,0425]; delta vs raw 3-hop = +0,0031, CI
  [−0,0160; +0,0238]. Hai gate đều fail.
- Trên 19 event có filtered support, gain vs raw là +0,0165; nhưng 5 event cải
  thiện và 4 event tệ đi. Best-of-three post-hoc cũng chỉ +0,0370 vs local.
- Peak 7,85 GiB trên một H100; process thoát và GPU về 1 MiB. Manifest SHA256:
  `2321936b677ea46e729fa4b38336a8a359692b0d49d9868a37c8e0ed0993631d`.
- **Quyết định:** hard stop cho path filter + packet overlay này. Không tune
  rating/half-life/hub/diversity sau result; hướng tiếp theo phải đổi sang
  candidate-level graph scoring/retrieval nếu tiếp tục.

## AB14 — P5/P5b candidate-directed graph scoring — 2026-09-21

- P5 tìm path riêng cho mọi candidate, strict-past và rating>=4; so max 3 vs 5
  item-layers bằng cùng beam/cap/recency/hub/length score. Graph score fuse vào
  frozen local rank với alpha 0,25; không LLM/GPU.
- Trên 1/48 graph, layer-5 chỉ cover 5/100 gold và residual gain +0,0013. P5b
  giữ nguyên scorer nhưng dùng full graph: gold evidence tăng 15%→30%, trong khi
  negative-slot evidence tăng 2,33%→13,78%.
- Full-graph residual layer-3/5 NDCG@5 = 0,5848/0,5829. Layer-5 vs local =
  +0,0077, CI [−0,0100; +0,0266]; layer-5 vs layer-3 = −0,0019, CI
  [−0,0137; +0,0086]. Gate +0,02/+0,01 đều fail.
- Post-hoc best-of local/3/5 chỉ +0,0188; không đủ headroom để justify learned
  scorer trên cùng cohort. P5b metrics SHA256:
  `928c12300868ac662355373fe2d83bfcbdf426f2416cabf0293689661da84168`.
- **Quyết định:** hard stop cho fixed 5-layer scorer. Density khắc phục recall,
  nhưng deeper expansion làm noise tăng nhanh hơn signal. Chỉ xét protocol mới
  với independent training cohort + learned depth-adaptive gate.

## AB14 — P6 learned adaptive multi-hop — 2026-09-21

- Khóa fresh-test protocol vì cohort P3–P5 đã bị quan sát: train pairwise scorer
  trên 1.200 event quá khứ, dùng 100 event cũ chỉ calibration và dành 100 event
  sau validation cutoff làm primary test.
- Graph smoke 24 train + 20 test pass. Temporal audit trước fit bắt được và sửa
  lỗi same-timestamp: các review cùng ngày không còn được tính là history của
  nhau. Toàn bộ test prompt có 5–6 review strict-past.
- Validation chọn alpha 0,6, margin gate 0,1: 0,6018 vs local 0,5752 (+0,0265),
  active 44/100. Model/config được hash trước khi test labels được mở.
- Fresh test: local/adaptive NDCG@5 = 0,5897/0,5672; delta −0,0225, CI
  [−0,0559; +0,0113]. Adaptive cải thiện 8, làm tệ 22, giữ nguyên 70; active
  57/100. Oracle best-of post-hoc +0,0305 nhưng không deployable.
- Self-host smoke 20 pass; full 200/200 success, zero retry/error/repair, peak
  7,84 GiB trên một H100. Process thoát và GPU về 1 MiB. Metrics SHA256:
  `91389ca081e24a6fcb981d74f1add16f630801f81de37e22be3799169a305816`.
- **Quyết định:** hard stop cho depth-weight + scalar margin gate. Không retune
  trên fresh test; continuation cần learned intervention-risk gate và một test
  cohort mới.

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

## CE0/CE1 — Candidate-conditioned 1-hop evidence — active

Chi tiết protocol và gate ở [CANDIDATE_EVIDENCE_PLAN.md](CANDIDATE_EVIDENCE_PLAN.md).
CE không mở C2/n-hop và không đảo kết luận MH2; nó chỉ kiểm tra liệu 1-hop đã
frozen có thể được căn chỉnh tốt hơn với request/candidate ở ranking-time hay
không.

| Field | CE0 / CE1 |
|---|---|
| Cohort | 100 user đầu tiên theo user_id từ locked cohort 141 user MH2 |
| Source | MH0 1-hop snippets và exact one-hop Stage-R cached ở MH2 |
| Arms | baseline, request-evidence, candidate-evidence |
| Budget | Tối đa 384 estimated evidence tokens; 10 total slot hoặc 1/candidate |
| Selector input | Request; với candidate arm thêm title/memory của đúng candidate; không gold/outcome |
| Rerank/report | 2 rerank độc lập/arm/user; không oracle/selection pass |
| CE0 artifact/hash | 100 row; SHA256 `b7b296…a266`; rerun deterministic; request estimate 210–359, candidate 226–383 token |
| CE1 result/gate | 600 canonical success record, 100/100 user. Candidate vs baseline: −0.0080 [−0.0434, +0.0281]; candidate vs request: −0.0110 [−0.0464, +0.0245]. **Hard stop; CE2 not admitted.** |

**CE1 execution/audit — 2026-08-25**

- Run `ce1_val_books_gpt54mini_pilot100` dùng exact one-hop Stage-R cache từ
  MH2; 100 user × 3 arm × 2 repeat = 600 canonical independent rerank record.
  Raw cache SHA256: `632fb362f758a2063ecfba955f707c43d3885d66eabaf487863280eea3ccda72`.
- Results (mean của hai repeat/user): baseline NDCG@5 0.7847; request evidence
  0.7877 (Δ +0.0030, 95% CI [−0.0329, +0.0370]); candidate evidence 0.7767
  (Δ baseline −0.0080, [−0.0434, +0.0281]; Δ request −0.0110,
  [−0.0464, +0.0245]).
- Two request-evidence responses cũng chấm source node IDs như candidate ID.
  Chúng được archive vào `retry_errors.jsonl`, rồi retry đúng hai key bằng cùng
  prompt/protocol; canonical cache cuối có 600 unique success, error = 0 và
  100/100 user analysed. Đây có thể đã phát sinh 2 API request bổ sung ngoài
  600 canonical record; Azure price không được cung cấp.
- Variability (mean absolute difference hai repeat NDCG@5): baseline 0.0830,
  request 0.1125, candidate 0.0931. Token stats trong `metrics.json` chỉ của
  invocation retry (2 request), không được dùng làm total-cost claim.
- **Gate:** fail. Candidate không vượt request +0.03 với CI lower > 0, cũng
  không vượt baseline +0.02. Không thực hiện CE2 hoặc tune lexical mechanism.

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
