# Thesis roadmap — Temporal Transition Graph Augmentation

**Cập nhật:** 2026-09-24

**Trạng thái:** active, canonical cho các bước nghiên cứu tiếp theo

**Ràng buộc:** không tune thủ công; mọi full run phải qua smoke 20–30 mẫu

Tài liệu này tổng hợp evidence đã có, khóa lại framing của luận văn và quy định
thứ tự các thí nghiệm còn lại. Các protocol/result đã sealed vẫn là nguồn số
liệu gốc; roadmap này không thay đổi outcome hay decision của các run cũ.

## 1. Quyết định pivot

Tên hướng nghiên cứu nên dùng từ đây:

> **Temporal Transition Graph Augmentation for LLM-based Sequential
> Recommendation**

Đóng góp trung tâm là **semantics của cạnh temporal transition**, không phải
việc tăng số hop và cũng không phải PPR tự thân. Kết quả hiện tại ủng hộ cách
diễn giải sau:

- naïve multi-hop trên graph đồng sở thích tăng coverage nhưng dễ khuếch đại
  hub/noise;
- graph chuyển tiếp có hướng bám trực tiếp objective dự đoán item kế tiếp;
- one-step transition là structural baseline mạnh và là method chính;
- PPR là biến thể propagation để tăng coverage trong graph thưa, không được
  claim là luôn tốt hơn one-step;
- LLM local ranker cung cấp semantic preference signal, còn transition graph
  cung cấp structural sequential signal bổ sung.

Liên hệ với MemRec vì vậy là **complementary graph augmentation for an LLM
memory ranker**, không phải “multi-hop semantic-memory propagation”.

## 2. Research questions đã khóa

### RQ1 — Transition signal

> Temporal transition graph có cung cấp ranking signal bổ sung cho một
> LLM-based semantic recommender hay không?

Amazon discovery/replication và MovieLens frozen transfer đã cung cấp evidence
ban đầu; graph-hard end-to-end là phép kiểm định còn thiếu.

### RQ2 — Propagation depth

> Multi-hop propagation có tăng chất lượng so với direct transition hay không,
> và trong điều kiện nào?

Kết luận hiện tại: **không phổ quát**. PPR tăng reachability rõ trên Amazon thưa,
nhưng không vượt one-step ổn định trên MovieLens graph-hard; learned router cũng
không pass.

### RQ3 — Robustness under structural hardness

> Transition augmentation còn hữu ích khi mọi negative đều có graph evidence
> hợp lệ hay không?

Đây là primary question của thí nghiệm tiếp theo. Nó tách complementary
preference signal khỏi shortcut “gold reachable, uniform negative không
reachable”.

Strong classical baselines là threat-to-validity check cho ba RQ trên, không
phải một vòng tìm method mới sau khi thấy test outcome.

## 3. Evidence ledger hiện tại

| Study | Contract | Kết quả NDCG@5 | Decision |
|---|---|---:|---|
| Amazon discovery, 100 event | Local vs Local + PPR | 0,6533 → 0,7285; Δ +0,0752; CI95% [+0,0179; +0,1375] | Pass |
| Amazon replication, 200 event | User-disjoint, frozen method | 0,624675 → 0,761807; Δ +0,137133; CI95% [+0,090385; +0,183835] | Pass; số final Amazon |
| MovieLens frozen transfer, 500 event | 1 positive + 9 uniform negatives | Local 0,471944; exact residual 0,838235; Δ +0,366290; CI95% [+0,330511; +0,400667] | Pass, nhưng candidate dễ |
| MovieLens frozen transfer, secondary | Session-300s residual | 0,849310; Δ +0,377365; CI95% [+0,343037; +0,411524] | Pass, nhưng candidate dễ |
| MovieLens graph-hard headroom, 200 event | PPR graph-only vs one-step graph-only | Exact +0,019396, CI95% [-0,015394; +0,053643]; session -0,035299, CI95% [-0,067050; -0,004538] | Fail; stop global PPR promotion |
| MovieLens depth router | Fixed 12 features, OLS, 5-fold OOF | Exact -0,000376; session -0,008650 vs one-step | Fail; stop router/depth tuning |

Các số và hashes đầy đủ nằm trong:

- [TRANSITION_PPR_METHOD.md](TRANSITION_PPR_METHOD.md);
- [MOVIELENS32M_PROTOCOL.md](MOVIELENS32M_PROTOCOL.md);
- [MOVIELENS_GRAPH_HARD_HEADROOM.md](MOVIELENS_GRAPH_HARD_HEADROOM.md);
- [MOVIELENS_DEPTH_ROUTER_PROTOCOL.md](MOVIELENS_DEPTH_ROUTER_PROTOCOL.md).

## 4. Interpretation boundary và headroom còn lại

MovieLens uniform-negative không chứng minh PPR/deeper hop là nguyên nhân của
gain lớn. Trong chính cohort đó, exact/session one-step graph-only đạt
0,861449/0,869054, cao hơn graph-only PPR tương ứng. Graph-hard M5 còn cho thấy
PPR không có gain ổn định khi tất cả negative đã reachable.

Ngược lại, one-step vẫn đạt 0,593659 (exact) và 0,643045 (session) NDCG@5 trên
M5 graph-hard. Điều này xác nhận transition graph còn có signal khi shortcut
zero-vs-nonzero đã bị loại. Tuy nhiên M5 không chạy local LLM, nên **incremental
headroom của Local + one-step so với Local hiện chưa được đo**. Không được lấy
chênh lệch giữa local của cohort dễ và graph-only của cohort khó để ước lượng
headroom vì đó là hai candidate distribution khác nhau.

Tính khả thi của thí nghiệm tiếp theo là cao ở tầng dữ liệu: 494/500 user trong
fixed M5 scan tạo được graph-hard candidate set. Compute cũng đã được kiểm chứng:
Qwen3-4B dùng khoảng 7,7 GiB peak VRAM, và một run 500 event tương ứng 1.000
physical LLM requests. Rủi ro khoa học chính không phải khả năng chạy, mà là hai
signal local và one-step có thể trùng nhau sau khi candidate đã graph-hard.

## 5. Thứ tự công việc tiếp theo

### P0 — Claim/protocol freeze — hoàn tất bằng roadmap này

- Dừng hoàn toàn việc thử thêm hop count, rule, threshold, feature hoặc router
  trên M5/M6 labels.
- Giữ one-step làm method chính; PPR chỉ là secondary arm.
- Giữ exact-timestamp view làm primary để nhất quán với frozen-transfer
  protocol; session-300s là secondary cố định, không được dùng thay primary nếu
  primary fail.
- Goodreads hiện chỉ là dataset deferred có điều kiện; xem
  [GOODREADS_READINESS.md](GOODREADS_READINESS.md).

### P1 — Baseline infrastructure, trước khi mở fresh outcome

Các baseline bắt buộc trên cùng temporal split, cohort và candidate set:

| Baseline | Vai trò | Trạng thái code |
|---|---|---|
| Global MostPopular | non-personalized sanity baseline | đã implement; chờ full score |
| First-order Markov / one-step transition | structural sequential baseline; chính là graph-only one-step | đã có scorer |
| BPR-MF | collaborative non-sequential baseline | model/trainer/early stopping đã test offline; chờ GPU smoke |
| SASRec | strong sequential baseline | model/trainer/early stopping đã test offline; chờ GPU smoke |

LightGCN là optional. Chỉ được đưa vào protocol nếu implementation và smoke đã
hoàn tất **trước** score lock; không được thêm sau outcome để cứu kết quả.

Mỗi learned baseline dùng đúng một config preregistered, không grid/random
search. Model selection chỉ được dùng deterministic early stopping trên
validation độc lập. Khóa trước source/revision, preprocessing, seed list,
maximum epochs, patience, metric và tie-breaking. Báo cáo mọi seed đã khóa,
không chọn seed tốt nhất.

### P2 — M7: fresh MovieLens graph-hard end-to-end — ưu tiên cao nhất

Tạo `MOVIELENS_GRAPH_HARD_END2END_PROTOCOL.md` và khóa nó trước khi materialize
test scores. Contract chi tiết nằm ở §6.

Luồng thực thi:

1. unit test candidate, fusion, baseline scoring và leakage guard;
2. prepare/hash cohort và candidate order nhưng chưa compute metric;
3. CPU graph smoke trên 20–30 event, rồi full graph scoring;
4. trainer/inference smoke trên 20–30 event với đúng config/model của full run;
5. train/score BPR-MF và SASRec tuần tự trên một GPU; release GPU sau từng task;
6. chạy local LLM smoke, reuse kết quả smoke trong full 500-event run;
7. hash toàn bộ arm scores và manifests;
8. mở labels đúng một lần để tính metric, paired bootstrap và gate.

### P3 — Mechanism analysis, không dùng để chọn method

Sau khi M7 sealed, tạo một analysis-only candidate difficulty curve với tỷ lệ
graph-reachable negatives cố định ở 0/25/50/75/100%. Ưu tiên CPU graph-only;
không đổi method và không dùng plot này để tune alpha/depth. Báo cáo:

- gold/negative reachability;
- one-step và PPR NDCG@5;
- delta theo structural hardness;
- coverage, graph degree và history-length buckets đã định trước.

Mục tiêu là giải thích mechanism của effect, không tạo thêm claim confirmatory.

### P4 — Goodreads chỉ khi readiness gate pass

Snapshot hiện tại không có interaction timestamp và rating table chỉ định danh
item bằng `Name`. Vì vậy không thể tạo strict-past next-item protocol đáng tin
cậy. Không chạy full parse, model hay GPU trên Goodreads hiện tại.

Chỉ promote Goodreads nếu có nguồn bổ sung chứa ít nhất:

```text
user_id, stable_book_id, interaction_timestamp, rating_or_event
```

và stable book ID join được metadata. Nếu gate pass, thứ tự là bounded audit →
20–30 event smoke → CPU-only graph coverage/headroom → chỉ chạy LLM khi
pre-registered gate pass. Nếu không có timestamp, ghi dataset là excluded vì
construct mismatch; không ép row order thành chronology.

### P5 — Thesis packaging

Sau M7, dừng experimental expansion bất kể pass/fail và hoàn thiện:

- method definition và computational complexity;
- baseline table trên identical candidates;
- confidence intervals và per-event paired analysis;
- candidate hardness mechanism plot;
- qualitative cases được chọn bằng deterministic rule, không cherry-pick;
- limitations về 10-candidate reranking, dataset age, title join và graph
  construction cost;
- reproducibility appendix gồm config/hash/request/GPU audit.

## 6. M7 contract cần preregister

### 6.1 Cohort và candidates

- 500 fresh MovieLens test users, một target mỗi user.
- Zero user overlap với mọi M1 development/primary cohort và toàn bộ fixed scan
  set của M5/M6.
- Deterministically scan đúng 1.000 eligible users theo salt mới; giữ 500 event
  khả thi đầu tiên, không mở rộng scan nếu thiếu.
- Giữ rating threshold, minimum history, sáu recent unique seeds và interface
  1 positive + 9 negatives.
- Mỗi negative phải có positive one-step score trong **cả exact và session
  graph**, đồng thời không nằm trong strict-past user history.
- Negative được chọn uniform bằng hash trong reachable intersection; không dựa
  score magnitude, genre, popularity, semantic similarity hay target score.
- Gold reachability không phải eligibility condition.
- Graph snapshot chỉ dùng interaction trước global validation cutoff; target và
  candidate outcome không được đọc trong scorer.

### 6.2 Ranking arms

Primary exact-timestamp view:

1. Local LLM;
2. one-step graph-only / first-order Markov;
3. **Local + one-step residual — primary treatment**;
4. PPR graph-only;
5. Local + PPR residual — secondary.

Session-300s lặp lại arms 2–5 như một fixed secondary robustness view. Local là
cùng output, không gọi LLM lần hai theo graph view. MostPopular, BPR-MF và SASRec
được score trên cùng candidate order để kiểm threat “weak LLM baseline”.

### 6.3 Frozen scoring

- Local model/prompt/parser/revision/decoding giữ nguyên M1.
- Fusion giữ nguyên reciprocal-log local score, within-candidate min–max graph
  score và `alpha = 0,80` cho cả one-step/PPR; không calibrate lại.
- PPR giữ sáu seed, restart 0,15, 50.000 walk, max depth 64.
- Candidate order và tie-break được hash trước outcome.
- Không best-of-arm, target-aware routing hay oracle selection trong deployable
  result.

### 6.4 Primary gate

Primary comparison duy nhất:

```text
Local + exact one-step residual  vs  Local
```

Pass khi đồng thời:

- mean ΔNDCG@5 `>= +0,03`;
- paired-bootstrap 95% CI lower bound `> 0`.

Hit@5, session view, PPR, classical baselines và bucket analyses là secondary.
Không promote secondary view nếu primary fail. Không chọn arm tốt nhất sau khi
thấy outcome.

### 6.5 Compute/request contract

- Offline/unit test trước mọi compute nặng.
- Smoke đúng 20–30 event; cùng checkpoint, prompt, parser, dtype và candidate
  protocol với full run.
- 500 event local ranker dự kiến 1.000 physical requests; smoke outputs phải
  được reuse, không gọi lại.
- Mỗi thời điểm chỉ dùng một H100; không đồng thời train baseline và serve LLM.
- Tuân thủ `internal_docs/H100_RESOURCE_RULES.md`: Slurm preflight, dynamic GPU
  selection, đúng một visible GPU, lưu before/after snapshot và unload ngay khi
  task kết thúc.

## 7. No-manual-tuning governance

Những thao tác sau bị cấm sau protocol lock hoặc sau khi thấy bất kỳ M7 label:

- đổi alpha, prompt, model revision, seed count, hop/depth/restart;
- đổi graph view primary, session gap, negative filter hay candidate hardness;
- thêm feature/router/baseline hoặc chọn checkpoint/seed theo test metric;
- mở rộng cohort/scan vì outcome không thuận lợi;
- báo cáo best arm như confirmatory method.

Mọi lỗi schema/runtime được sửa bằng run ID mới và smoke lại. Sửa lỗi không
được thay đổi semantic contract. Development, validation và test artifacts phải
user-disjoint, immutable và có SHA256; ranking code không nhận gold index.

## 8. Decision table sau M7

| Outcome | Kết luận và hành động |
|---|---|
| Local + one-step pass | Claim complementary transition signal dưới graph-hard candidates; dừng tìm method mới, chuyển sang baseline/analysis/writing |
| Delta dương nhưng CI/gate fail | Evidence chưa đủ; báo effect size/uncertainty, không tune rescue |
| Local + one-step không hơn Local | Gain trước đây phụ thuộc candidate reachability; thu hẹp claim và dừng method expansion |
| SASRec vượt method | Không claim SOTA; định vị contribution là graph signal/LLM augmentation và phân tích complementarity |
| Method vượt SASRec/BPR-MF | Báo identical-candidate comparison, nhưng vẫn giới hạn claim ở 10-candidate reranking |
| Goodreads không có timestamp | Exclude khỏi temporal core và nêu construct mismatch |

Pass hay fail đều tạo kết luận khoa học rõ. Không outcome nào cho phép quay lại
tune M5/M6 hoặc tăng thêm hop để tối ưu test.

## 9. Progress checklist

- [x] Amazon discovery và internal replication sealed.
- [x] MovieLens frozen cross-domain transfer sealed.
- [x] MovieLens graph-hard PPR-vs-one-step headroom sealed; gate fail.
- [x] No-tuning depth router sealed; gate fail.
- [x] Thesis framing/RQs và next-study priority được khóa.
- [x] Bounded Goodreads readiness audit; deferred do thiếu timestamp/item ID.
- [x] Viết M7 graph-hard end-to-end protocol; hash cùng config khi prepare.
- [x] Hoàn thiện và unit-test MostPopular, BPR-MF, SASRec cùng M7 score-lock/evaluator.
- [x] Prepare fresh 500-user M7 cohort và seal candidates.
- [ ] Smoke 20–30 event cho từng workload mới (candidate/graph/local dry pass;
  baseline và LLM GPU smoke còn lại).
- [ ] Score/hash tất cả arms trước khi mở outcome (graph 500/500 complete;
  baseline và local LLM còn lại).
- [ ] One-time M7 evaluation và result documentation.
- [ ] Candidate hardness mechanism analysis.
- [ ] Thesis tables, plots, limitations và reproducibility appendix.

GPU preflight ngày 2026-09-24 xác nhận allocation Slurm `15288` đã hết hạn.
Theo resource runbook, baseline/LLM smoke và full run đang dừng ở compute gate;
không tự tạo allocation mới và chưa có outcome nào được mở.
