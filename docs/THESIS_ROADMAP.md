# Thesis roadmap — Cải tiến **full MemRec**

**Cập nhật:** 2026-09-25

**Trạng thái:** reset framing và protocol; **chưa có kết quả chứng minh cải tiến full MemRec**

**Nguồn benchmark:** [MemRec paper, ACL 2026](https://aclanthology.org/2026.acl-long.2061/)

## 1. Mục tiêu và ranh giới tuyên bố

Mục tiêu chính là cải tiến hệ thống MemRec **đầy đủ**: collaborative user/item
memory, graph, Stage-R synthesis, Stage-ReRank và Stage-W propagation. Phương
pháp mới phải vượt **full MemRec không sửa đổi** và **SASRec** trên **cùng dữ
liệu, split, candidate list, metric và điều kiện đánh giá**. Chưa đạt hai điều
kiện này thì chưa được viết rằng thesis đã thành công hoặc đạt SOTA.

Paper gốc đánh giá bài toán **ranking trên candidate set được cho trước**
(`N=10` ở bảng chính), không đo retrieval toàn bộ catalog. Do đó “full MemRec”
ở đây nghĩa là **full kiến trúc MemRec**, không phải tự ý đổi task thành
full-catalog retrieval. Nếu sau này mở thêm end-to-end retrieval, báo cáo thành
một protocol riêng và không so số trực tiếp với Table 2/3 của paper. Với 1
positive + 9 negative, Hit@10 luôn bằng 1 nên metric chính là NDCG@5 và Hit@1;
NDCG@3/Hit@3/Hit@5 là secondary.

## 2. Sửa sai của giai đoạn trước

Các thí nghiệm Temporal Transition/PPR/MovieLens M7 đã cải thiện **local LLM
ranker** bằng graph residual, **không** tích hợp vào full MemRec. M7 MovieLens
graph-hard có NDCG@5 `0,699773`, thấp hơn SASRec `0,827329` trên cùng 500
case. Đây là nghiên cứu thăm dò về tín hiệu graph; không phải thesis result
hay bằng chứng cải tiến MemRec. Amazon transition replication cũng dùng local
ranker khác full MemRec. Giữ artifacts để tham khảo giả thuyết, không xóa hay
viết lại kết quả; ngừng dùng [THESIS_DRAFT.md](THESIS_DRAFT.md) làm bản đồ án.

Số `0,6601` (MemRec) và `0,2824` (SASRec) NDCG@5 ở Books Table 2 của paper
khác hẳn bài toán MovieLens M7. Không so chéo dataset/protocol/LLM. Paper
dùng GPT-4o-mini và toàn bộ test set; API cũ hiện không dùng được. Một baseline
self-host phải được xây lại cùng model với method mới. Số paper là mốc tham
chiếu có ghi khác biệt model, **không** thay cho so sánh paired thực nghiệm.
Mục tiêu định lượng thêm là vượt `0,6601` trên Books paper-style; chỉ được gọi
là vượt **published MemRec** nếu đối chiếu được candidate/split/feedback và
LLM tương đương. Nếu không, ghi rõ “vượt mốc số học, khác điều kiện LLM”.

## 3. P0 — Dựng lại benchmark công bằng (ưu tiên tuyệt đối)

Dataset chính là InstructRec Books gốc đang có tại
`data/processed/instructrec-books/`; 7.377 user. PKL gốc có đúng 10 candidate
cho mỗi user và target nằm trong list. Repo hiện có full MemRec code nhưng
evaluator trước đây bỏ qua các list này và lấy negative ngẫu nhiên. Chi tiết
audit và các rủi ro ở [FULL_MEMREC_BASELINE_AUDIT.md](FULL_MEMREC_BASELINE_AUDIT.md).

Các gate theo thứ tự:

1. **Baseline provenance:** pin commit [upstream MemRec chính thức](https://github.com/rutgerswiselab/memrec),
   diff các module `src/{memory,models,train,data}` với fork hiện tại và
   tách mọi fix protocol (candidate/leakage) khỏi thay đổi thuật toán. Đã pin
   `58d9031` và audit khác biệt ở [baseline audit](FULL_MEMREC_BASELINE_AUDIT.md);
   self-host baseline là upstream-aligned, không phải exact GPT-4o-mini run.
2. **Candidate contract:** dùng `ranked_lists` gốc cho Books test; không
   reshuffle, không lấy negative ngẫu nhiên. Xác thực 30 mẫu trước, sau đó
   7.377/7.377 mẫu: mỗi list 10 item duy nhất, target đúng một lần, không
   trùng history, ID có metadata. Hash manifest candidate và user order.
3. **Split và leakage:** graph chỉ từ train history. Quy định rõ validation
   item được biết đến lúc test hay chưa; freeze snapshot tương ứng cho tất cả
   arms. Không cho nhãn test của user trước đi vào shared memory khi đánh giá
   user sau. Nếu cần mô phỏng online feedback thì phải có trật tự thời gian
   toàn cục đáng tin cậy; Books chỉ có sequence position nội bộ, nên primary
   test chạy **không có test-label Stage-W**. Stage-W vẫn được warm-up/cập nhật
   bằng train/validation đã cho phép, không bị bỏ khỏi kiến trúc.
4. **Validation và test:** hash split *user-disjoint trên original test rows*;
   không tạo validation candidates khác phân phối. 1.085 user có outcome từ
   historical runs thuộc dev; thêm 915 user theo hash để dev=2.000. Còn lại
   5.377 user chưa từng được chấm là held-out primary. Chỉ **evaluation
   labels** disjoint; train history của mọi user vẫn được dùng như benchmark.
   Held-out hash `b7751e0a…4af8be`; xem audit cho full digest. Không mở
   held-out outcome để chọn method. Báo cáo toàn bộ 7.377 user kiểu paper chỉ
   sau khi khóa method, và ghi rõ phần dev overlap.
5. **Baseline parity:** cùng self-host LLM, version/checkpoint, prompt,
   temperature, context budget, candidate order, warm-up memory và scoring
   retries cho full MemRec và method mới. Smoke 30 user warm-up/ranking trước;
   full run phải dùng `warmup_user_scope=all` (7.377 user warm-up từ pre-test
   interactions), rồi score chỉ cohort cần đo. Chạy full MemRec nguyên bản trước,
   đo Stage-R/RR/W thực sự được gọi và memory/graph có nội dung. SASRec được
   train trên đúng train split và score **cùng candidate lists**; hyperparameter
   selection tự động chỉ qua validation, không tune thủ công/test.
6. **Reproducibility:** lưu per-user predictions, metric denominator, failure
   count, candidate/snapshot/checkpoint hashes, GPU/model time và token cost.
   Failure giữ trong mẫu số; không âm thầm bỏ case lỗi. Cố định seed/retry và
   sử dụng paired bootstrap theo user cho delta.

P0 hoàn thành chỉ khi ít nhất 20–30 case end-to-end smoke (gồm Stage-R/RR/W
warm-up), rồi một run baseline held-out ổn định; chưa vội báo final method.

## 4. P1 — Chẩn đoán headroom của **full MemRec**

Trên development cohort, phân rã lỗi theo history length, item popularity,
candidate semantic similarity và graph coverage. Chạy ablation có cùng
candidate/LLM: full; tắt collaborative read; tắt Stage-W; giữ full nhưng thay
đổi duy nhất nguồn evidence được đưa vào Stage-R. Mục đích là tìm bottleneck
có *incremental headroom* so với full, không recycle kết quả local ranker.

Thực hiện oracle hợp lệ chỉ trên development labels để ước lượng ceiling của
evidence selection / fusion. Không dùng oracle test làm score thật. Nếu graph
transition không mang tín hiệu bổ sung khi full memory đã hoạt động, bỏ giả
thuyết đó và chuyển sang cơ chế khác; không tiếp tục tăng hop hay chỉnh rule
thủ công để cứu kết quả.

## 5. P2 — Thiết kế method gắn vào MemRec đầy đủ

Hai hypothesis có lý do từ exploration nhưng **chưa được xác nhận**:

- **Evidence-aware collaborative memory:** dùng path/transition signal để
  chọn evidence cho Stage-R trong cùng `k=16`, `τ=1800` và Stage-W fanout.
  Output vẫn là memory/facets có provenance, Stage-ReRank vẫn là MemRec LLM.
  Không cộng residual vào final score rồi gọi là full MemRec improvement.
- **Learned sequential–collaborative fusion:** đưa sequential model (SASRec)
  thành một nguồn evidence/prior cho MemRec, học cách phối hợp trên dev thay
  vì trọng số hand-tuned. Giữ full memory graph và Stage-W. Arm SASRec đơn lẻ
  là bắt buộc để xác nhận fusion có tăng thật, không chỉ sao chép SASRec.

Chọn một architecture bằng development protocol; báo cáo ablation sao cho
phân biệt gain từ memory integration, sequence model và chi phí context/compute.
Nếu method dùng supervised training còn full MemRec zero-shot, phải báo riêng
trade-off dữ liệu và thêm baseline SASRec/learned fusion tương ứng; không tuyên
bố hơn do dùng tài nguyên không cân xứng mà không phân tích.

## 6. P3 — Đánh giá xác nhận và tiêu chí dừng

Khóa code/config/checkpoints/candidate manifest **trước** held-out test.
Primary gate: Δ NDCG@5 (method − full MemRec) > 0 và Δ NDCG@5
(method − SASRec) > 0 trên **cùng held-out cohort**, 95% paired bootstrap CI
cho cả hai delta có cận dưới > 0. Báo cả Hit@1, NDCG@3/5, lỗi runtime,
latency, token/GPU budget và không chọn seed tốt nhất. Nên lặp trên ít nhất
hai seed hoặc độc lập replicate nếu budget cho phép; pre-register trước khi
mở test. Full Books paper-style là secondary và kiểm tra mốc `0,6601` nêu ở
§2; dataset gốc thứ hai khi có dữ
liệu đúng protocol là kiểm tra generalization. MovieLens32M là secondary
transfer **chỉ sau khi full MemRec được port sang đó**, không so M7 với Books.

Nếu primary fail, kết luận trung thực là chưa đạt aim; quay về dev với test
cohort mới hoặc thay cơ chế, không tái tune trên held-out labels đã mở. Nếu
full MemRec self-host kém paper, chẩn đoán model/prompt/protocol trước khi
đánh giá phương pháp. Không hứa trước kết quả vượt paper hoặc SASRec.

## 7. Tài nguyên và quy tắc vận hành

Trước mọi full run: smoke 20–30 case, đúng config; không tune thủ công bất kỳ
rule/weight nào sau khi thấy test. GPU chỉ dùng trong Slurm allocation đang
RUNNING và sau preflight ở `internal_docs/H100_RESOURCE_RULES.md`; tối đa một
H100 80GB cho task mới, nhả card ngay khi xong. Quyền 2 GPU của job trước
không tự động áp dụng cho task này. Không chạy GPU từ turn lập kế hoạch này.

## 8. PROGRESS & RESULTS (điền tiếp; chưa phải số final)

| Mốc | Trạng thái | Evidence/Artifact | Kết quả |
|---|---|---|---|
| Upstream provenance/diff | Xong 2026-09-25 | `58d9031`; [diff audit](FULL_MEMREC_BASELINE_AUDIT.md) | Core memory unchanged; prompt corner case guarded |
| Audit Books original candidates | Xong 2026-09-24 | [Baseline audit](FULL_MEMREC_BASELINE_AUDIT.md); SHA-256 `a13f7435…2f87cb6` | 7.377/7.377 valid; code path đã được nối |
| Candidate/cohort + feedback guards | Xong 2026-09-25 | held-out SHA-256 `b7751e0a…4af8be`; tests | 2.000 dev, 5.377 held-out; no test writes |
| CPU full-agent wiring smoke | Xong 2026-09-25 | `scripts/smoke_full_memrec_cpu.py`; 30 user | 30/30 predictions, 30 warm-up writes, 0 failures; fake LLM, không phải ranking result |
| CPU all-user warm-up wiring smoke | Xong 2026-09-25 | 7.377 fake warm-up + 30 fake ranking | 7.377 Stage-W warm-up, 0 test writes; không phải ranking result |
| Self-host LLM smoke | Chưa chạy — allocation cũ hết hạn | 20–30 user + GPU manifest | — |
| Full MemRec self-host smoke 20–30 | Chưa chạy | run ID, Stage calls, failures | — |
| Full MemRec + SASRec paired baseline | Chưa chạy | same candidate/split | — |
| Headroom/ablation dev | Chưa chạy | preregistered dev report | — |
| Method smoke/full held-out | Chưa chạy | hashes, per-user predictions | — |
| Independent replication | Chưa chạy | second seed/domain | — |

Các số M7/transition giữ trong historical protocol documents, **không chuyển
vào bảng kết quả full MemRec**.
