# MULTIHOP_PLAN.md — Selective Multi-Hop Collaborative Memory

> **Trạng thái:** hoàn tất ở MH2 — hard stop. Bounded oracle trên validation
> không có headroom thực dụng; xem [RESULTS.md](RESULTS.md) và
> [PROGRESS.md](PROGRESS.md).
>
> **Tài liệu nền:** [RL_WORK_SUMMARY.md](RL_WORK_SUMMARY.md) ghi lại nghiên cứu
> SFT/RL đã đóng. Kế hoạch này không tiếp tục GRPO.
>
> **Hướng đã thử sau MH2:** [CANDIDATE_EVIDENCE_PLAN.md](CANDIDATE_EVIDENCE_PLAN.md)
> đã dừng ở CE1 vì lexical candidate-conditioned evidence không qua gate; đây
> không phải continuation của multi-hop expansion.
>
> **Pivot write-side đã audit:** [BUFFERED_PROPAGATION_PLAN.md](BUFFERED_PROPAGATION_PLAN.md)
> tách rõ Stage-W propagation khỏi read-side expansion. P0 cho thấy dataset
> không có global cross-user clock và P1 static source-only không đủ endpoint
> coverage; không có write-side LLM experiment được admission.

## 1. Mục tiêu và câu hỏi nghiên cứu

MemRec hiện khai thác collaborative context từ immediate neighborhood. Câu hỏi
trung tâm là:

> Với **cùng budget node và token context**, thông tin ở xa hơn trên graph có
> tăng chất lượng ranking không; và nếu có, có thể chọn/routing thông tin đó bằng
> semantic relevance, path strength và redundancy thay vì mở toàn bộ neighborhood không?

Ba câu hỏi đo được:

- **RQ1 — Headroom:** 2-hop tốt nhất trong cùng budget có vượt 1-hop trên
  validation một cách thực chất không?
- **RQ2 — Selection:** selector candidate-blind có lấy được một phần đáng kể của
  upper bound đó, và vượt naive 2-hop không?
- **RQ3 — Routing:** nếu read-side selection có ích, selective propagation của
  update tới endpoint xa có giữ gain trong đánh giá tuần tự không?

Kết quả negative vẫn hợp lệ: không có headroom đáng kể dưới budget cố định, hoặc
oracle tốt nhưng selector không chuyển được thành gain.

## 2. Phạm vi và nguyên tắc cố định

### Trong phạm vi

- Dataset chính: instructrec-books; memory snapshot/split/candidate hiện có trong
  data/rl và topology sẽ được đóng băng từ interaction history trong MH0.
- So sánh 1-hop MemRec, naive 2-hop và selective 2-hop.
- Oracle bounded trên validation trước; selector/routing chỉ làm sau khi oracle qua gate.
- LM_Rec/reranker đóng băng; không SFT, RL hay joint training.
- Mọi arm có cùng candidate set, node budget và token context budget.

### Ngoài phạm vi hiện tại

- Full GRPO/SFT cho LM_Mem hoặc LM_Rec.
- Full-catalog retrieval, đổi candidate sampling, hoặc tăng context window để đổi lấy gain.
- Mở toàn bộ 2-hop graph, propagation không giới hạn, hay test-time tuning.
- Chạy test khi selector/config chưa được khóa từ validation.

### Hợp đồng đánh giá (không được nới lỏng)

| Thành phần | Quy tắc |
|---|---|
| User split | Train 1,185 / val 149 / test 993, user-disjoint; chỉ tune trên train/val. |
| Graph state | Frozen memory snapshot **và** topology hash từ interaction history trước held-out target. Không ghi memory từ test feedback trong read-side experiment. |
| Candidates | Cùng 10 candidate tất định/user, cùng thứ tự cho mọi arm của user. |
| Selector input | Chỉ M_u, frozen memory/metadata graph và path; **không** instruction, candidates, gold hay ranking outcome. |
| Context budget | Cùng K_u node đã packed của control và cùng cap T_u token/user; record số thực tế. |
| Model/call | Cùng Stage-R prompt/schema, reranker, temperature, max tokens và retry policy. |
| So sánh | Paired theo user; delta metric và bootstrap CI trên cùng user set. |

## 3. Định nghĩa graph và budget

### 3.1 Control 1-hop

N1(u) là đúng context của frozen protocol hiện có: UserItemGraph,
LLMRulePruner và SnippetPacker tạo state candidate-blind, với candidate-block
reserve giữ nguyên. LLM_Rec vẫn nhận cùng candidate item memories/instruction
như mọi arm khi chấm ranking.

N1(u) gồm:

- item trong history của u (cạnh u -> item);
- peer user cùng tương tác item history của u (u -> item -> peer);
- sau đó pruner chọn và pack vào prompt.

Graph là bipartite, vì vậy peer user có distance cấu trúc hai cạnh. Trong tài liệu
này, 1-hop nghĩa là **immediate collaborative neighborhood hiện có của MemRec**,
không phải số cạnh thuần túy. Đây là control bắt buộc.

### 3.2 Extended pool 2-hop

Từ peer user v trong N1(u), mở rộng một collaborative round nữa:

- remote item j trong H(v) nhưng không thuộc H(u), theo path u -> i -> v -> j;
- remote user w nối với remote item j, nếu cần quota user, theo path
  u -> i -> v -> j -> w.

Loại target u, node đã ở N1(u), duplicate, node không có memory/snippet hợp lệ,
và bất kỳ edge nào chứa held-out target. Pool này gọi là C2(u).

GraphSnapshot chỉ lưu memory và 1-hop render, không lưu adjacency đầy đủ. Vì
vậy MH0 phải dựng một **immutable topology artifact** từ cùng pre-target history,
hash nó, rồi MH1 chỉ được mở rộng trên artifact đó. Remote text vẫn phải được
render bằng cùng rule SnippetPacker; remote user không có stored memory dùng
static snippet, không được gọi thêm LLM để làm giàu context.

Mọi candidate trong C2(u) được tạo từ frozen graph trước khi ranker thấy candidate
list. Độ sâu, path, source node và loại node phải được lưu để audit.

Vì construction không được đọc gold/candidate để xóa node có điều kiện, guard
thực hiện theo hai pha: (1) build toàn bộ C2 candidate-blind; (2) screen C2 và
mọi bundle đã materialize. Nếu phát hiện candidate/gold/title, **loại user khỏi
mọi arm** của comparison; tuyệt đối không target-aware filter một node rồi giữ
user đó.

**MH1 structural cap đã khóa.** Để pool không biến thành full 2-hop expansion,
MH1 chỉ giữ tối đa 128 remote item và 64 remote user cho mỗi target user; mỗi remote item
đóng góp tối đa 8 remote user. Cap dùng duy nhất topology + N1 candidate-blind,
trước khi đọc candidate/instruction/gold, và lớn hơn rất nhiều quota oracle
{2, 4, 6}. Giá trị exact nằm trong `configs/multihop/mh0_books.yaml` và manifest
MH1. Pool thiếu node hoặc không đủ token luôn fill one-hop và ghi shortfall.

### 3.3 Budget controller

SnippetPacker hiện có k=16, tau_tokens=1800 nhưng số snippet/token thực tế khác
nhau theo user. Để không confound multi-hop với lượng context:

1. Materialize control 1-hop trước, thu K_u = n_neighbors và T_u = serialized
   neighbor-token count cho từng user.
2. Mỗi arm 2-hop được cấp đúng K_u slot và không vượt T_u, sau cùng một
   truncation rule. Nếu thiếu remote node hợp lệ, fill bằng node 1-hop và ghi
   remote_shortfall.
3. Giữ type mix của control khi khả thi; report số item/user/remote thực tế.
4. User/arm không đạt format hoặc budget assertion bị loại khỏi **mọi** arm của
   comparison đó; không được chỉ loại arm bất lợi.

## 4. Arm thí nghiệm

| Arm | Context được chọn | Mục đích |
|---|---|---|
| one_hop | Control hiện tại N1(u) | Baseline chính. |
| naive_two_hop | Thay q slot bằng node C2(u) theo traversal/path score cố định, không semantic/redundancy | Đo thêm signal xa khi chưa có selector thông minh. |
| oracle_two_hop | Best trong một tập hữu hạn bundle 2-hop candidate-blind; **gold chỉ dùng sau đó để chọn** | Upper bound bounded cho cùng budget, không phải phương pháp deploy. |
| selective_two_hop | Chọn node C2(u) bằng score semantic + path − redundancy; hyperparameter khóa từ validation | Phương pháp triển khai. |

Trong headroom pilot, dùng quota remote q trong {2, 4, 6}. Mỗi quota giữ đúng budget;
nếu count của một loại node không đủ, fill 1-hop và báo cáo.

## 5. Milestone và Definition of Done

### MH0 — Freeze protocol và kiểm tra artifact

**Mục đích:** biến hợp đồng ở §2 thành check tự động trước bất kỳ API run nào.

- Verify hash/record counts snapshot và ba jsonl; dựng/hash topology từ history
  tương ứng; kiểm tra split disjoint, gold-in-candidates, candidate-order hash
  và leakage guard hiện có.
- Re-materialize one_hop từ snapshot bằng pipeline hiện tại, không dùng bảng M0
  RNG theo thread.
- Viết manifest gồm git SHA, snapshot hash, config, model/version, prompt hash,
  seed, thời gian và cost.

**DoD:** mỗi record có user_id, immutable candidate/order hash, K_u, T_u và source
node list; test fail ngay nếu selector nhận instruction/candidate/gold.

### MH1 — Build C2(u) và bundle generator

**Mục đích:** xây extended pool offline, chưa claim quality.

- Implement deterministic graph expansion theo §3.2, tự loại duplicate/leakage.
- Sinh naive_two_hop bundle theo seed cố định; sinh B=12 bundle ứng viên oracle,
  phân tầng theo quota q (4 bundle/quota).
- Persist pool, bundle IDs, source paths, budget audit và text đã pack; Stage-R
  input phải reproducible byte-for-byte khi dùng cùng artifact.

**DoD:**

- C2(u) không chứa node 1-hop trừ fallback, không chứa held-out target và không
  dùng candidate/instruction/gold để sinh pool.
- Tất cả arm pass exact K_u/T_u assertions hoặc có reason code chung.
- Có unit tests cho path validity, deduplication, determinism, budget equality và leakage.

### MH2 — Oracle/headroom validation (gate bắt buộc)

**Mục đích:** chỉ trả lời RQ1 trước khi tối ưu selector/routing.

Protocol cho mỗi user validation:

1. Generate/cache Stage-R output cho one_hop, naive_two_hop và 12 oracle
   candidate bundles. Context construction vẫn hoàn toàn candidate-blind.
2. **Selection pass:** rank candidate bundle; offline analytics được phép dùng
   gold/NDCG@5 để chọn bundle tốt nhất của user. Đây là bounded oracle, không
   phải output selector.
3. **Report pass độc lập:** rerank lại one_hop, naive_two_hop và bundle oracle đã
   chọn ít nhất hai lần độc lập. Report mean paired metric của report pass, không
   report điểm selection pass. Điều này chặn winner's curse và nhiễu API.
4. Tính Delta_oracle và Delta_naive NDCG@5 paired theo user, với 10,000 bootstrap
   resamples; lưu per-user deltas và raw completions.

Metric chính là NDCG@5. H@1/H@3/H@5, NDCG@3, token/call latency, remote ratio và
coverage là metric phụ. Phân tích thêm theo history length, graph degree, remote
quota và path type, nhưng không dùng chúng để sửa test protocol.

**Gate sau MH2:**

| Kết quả validation | Quyết định |
|---|---|
| Delta_oracle <= +0.02, hoặc 95% CI upper <= +0.03 | **Dừng hướng multi-hop**: không đủ headroom thực dụng. |
| CI chưa nằm trên 0, hoặc +0.02 < Delta_oracle < +0.05 | Báo cáo borderline; chỉ làm audit/error analysis, không triển khai routing tốn kém. |
| Delta_oracle >= +0.05 và CI lower > +0.01 | Có headroom để tiếp tục MH3. Đây chưa phải claim final. |

Mốc +0.05 là admission gate thực dụng: selector không thể kỳ vọng đạt 100% oracle,
còn test claim chỉ được quyết ở MH4.

### MH3 — Selective 2-hop selector

**Không được admission trong run hiện tại:** MH2 cho Δ_oracle NDCG@5 = −0.0094
(95% CI [−0.0362, +0.0176]), kích hoạt hard-stop. Phần dưới là design record,
không phải công việc được phép tiếp tục sau run này.

Với remote node z, dùng frozen encoder và score đã chuẩn hóa trong từng user:

~~~
semantic(z)   = cosine(E(query_u), E(memory_or_snippet_z))
path(z)       = max_path product(edge_strength) * degree_attenuation
redundancy(z) = max_x_in_selected_1hop cosine(E(memory_or_snippet_z), E(memory_or_snippet_x))
score(z)      = alpha * semantic(z) + beta * path(z) - gamma * redundancy(z)
~~~

- query_u là M_u trong read-side experiment; trong routing experiment là update
  Delta M do Stage-W tạo. E là encoder đóng băng.
- path(z) lưu witness path. Edge strength dùng recency/overlap sẵn có; degree
  attenuation tránh hub node thống trị chỉ vì phổ biến.
- Tune alpha, beta, gamma, quota q và tie-break rule trên train/validation; chốt
  một config trước test. Không tune theo candidate/gold.
- So sánh selective với **cùng quota** naive và one-hop. Báo cáo oracle capture
  Delta_selective / Delta_oracle khi mẫu số dương và CI oracle không chứa 0.

**DoD:** selective vượt naive trên validation theo paired NDCG@5, không phá
budget/leakage contract, và qualitative audit 30 user cho thấy remote node có
path/relevance hợp lý thay vì hub/duplicate spam.

### MH4 — Locked test và phân tích

**Không được admission trong run hiện tại.** Chỉ bắt đầu khi MH3 config đã khóa. Chạy đúng một materialized test run với
one_hop, naive_two_hop và selective_two_hop; reranker/retry policy bằng nhau và
không chỉnh parameter sau khi thấy số.

**Claim policy:**

- Claim positive khi selective có paired Delta NDCG@5 >= +0.05, bootstrap 95% CI
  lower > 0, budget/latency không vượt control theo tolerance đăng ký, và gain
  không chỉ tập trung vào một failure bucket.
- Nếu oracle thắng nhưng selective không thắng naive/one-hop: báo cáo *selection
  gap*, không chuyển sang training/RL để che gap.
- Nếu naive thắng còn selective không: phân tích score/routing, không gọi đó là
  bằng chứng semantic selection có ích.

### MH5 — Selective propagation/routing (nhánh sau cùng)

**Không được admission trong run hiện tại.** Chỉ làm nếu MH3–MH4 chứng minh read-side signal xa có ích. Mục tiêu là đưa
update Stage-W Delta M tới tối đa q_prop endpoint trong C2(u) theo score §MH3,
thay cho fanout mở toàn bộ.

- So sánh one_hop_write, naive_two_hop_write, selective_two_hop_write với cùng
  endpoint cap, số write calls và temporal order.
- Mỗi update chỉ dùng interaction trước điểm eval; giữ separate snapshot/episode
  log để không ghi test target vào memory.
- Report coverage, unique endpoint, stale/hub rate, duplicate update rate,
  latency/token và ranking. Tách kết quả này khỏi read-side MH4.

Nếu MH2 fail, MH5 không được thực hiện.

## 6. Phân tích bắt buộc và rủi ro

| Rủi ro | Guard/measurement |
|---|---|
| Gain chỉ do nhiều context | Per-user K_u + T_u equality assertion và budget audit. |
| Gold/candidate leakage | Candidate-blind pool/selector; automated text/field checks; oracle tách biệt. |
| Oracle winner's curse | Selection pass và report pass độc lập; report only second pass. |
| Hub/noise amplification | Degree attenuation, redundancy penalty, report degree/path distributions. |
| Reranker API variability | Cache prompts/completions, rerun report pass, paired bootstrap. |
| Tuning test | Val-only config lock + manifest hash trước test. |
| 2-hop pool rỗng/nhỏ | Report coverage/shortfall; fill 1-hop consistently, không drop chọn lọc. |
| Ambiguous “hop” | Log structural path và dùng control N1(u) định nghĩa ở §3.1. |

## 7. Artifact và cập nhật tài liệu

Không ghi đè artifact RL cũ. Dùng namespace mới:

~~~
data/multihop/
  mh0_manifest.json
  mh0_topology_books.json
  mh1_pools_{train,val,test}.jsonl
  mh1_bundles_val.jsonl
  mh1_manifest.json
  mh2_headroom_val.json
  mh3_selector_config.json
  mh4_locked_test.json
results/multihop/
  <run_id>/raw_completions.jsonl
  <run_id>/budget_audit.json
  <run_id>/metrics.json
~~~

- [PROGRESS.md](PROGRESS.md): ghi milestone, commit, command, compute/API cost và
  gate decision ngay sau mỗi run.
- [RESULTS.md](RESULTS.md): chỉ ghi số đã qua protocol; bảng có ô trống là chủ ý
  cho tới khi artifact tương ứng tồn tại.
- [RL_WORK_SUMMARY.md](RL_WORK_SUMMARY.md): không cập nhật bằng kết quả multi-hop;
  đây là record đã đóng của hướng SFT/RL.

## 8. Quyết định sau gate

MH0–MH2 đã hoàn tất. MH2 có Δ_oracle NDCG@5 = −0.0094 với bootstrap 95% CI
[−0.0362, +0.0176], do đó thỏa hard-stop (`Δ_oracle ≤ +0.02`). Không tiếp tục
MH3/MH4/MH5 trong protocol này và không mở locked test để tìm một kết quả thuận
lợi hơn. Artifact, command và cost caveat của run được lưu ở PROGRESS/RESULTS.
