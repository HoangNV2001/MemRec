# Audit: full MemRec baseline trước khi phát triển method mới

**Ngày:** 2026-09-25
**Scope:** audit và chuẩn bị benchmark; chưa chạy LLM/GPU hay mở outcome mới.

## Kết luận ngắn

Repo **có** full MemRec implementation và dữ liệu InstructRec Books gốc. Tuy
nhiên, run full MemRec cũ **chưa phải** baseline đối chứng đủ sạch: test
candidate của pkl bị bỏ qua, evaluator tạo negative ngẫu nhiên; config cho phép
ghi ground-truth feedback trong test vào shared memory; chỉ đánh giá 1.000
user. Kết quả local graph/M7 càng không thay thế full MemRec. Vì vậy chưa có
kết quả nào chứng minh phương pháp thesis tốt hơn full MemRec và SASRec.
Đã pin [upstream MemRec](https://github.com/rutgerswiselab/memrec) commit
`58d9031ed91c623b8034d2bf04f39aa937424c33` (cũng là merge-base với
fork). Độ trung thực *protocol/paper results* vẫn cần kiểm chứng bằng run mới.

## Provenance so với upstream

`git diff 58d9031 -- src/memory src/models src/train src/data` cho thấy:

| Nhóm | Khác biệt ở fork | Ảnh hưởng tới full baseline |
|---|---|---|
| `src/memory/*` | Chỉ `__init__.py` chuyển import sang lazy | Không đổi graph/pruner/packer/manager/storage |
| `src/models/memrec_agent.py` | Thêm `vanilla_mode` cho ablation và compatibility switch | Full config vẫn `vanilla_mode=False`; core R/RR/W giữ nguyên |
| `src/models/reranker_llm.py` | Optional evidence và thay wording khi facets rỗng | Evidence không được cấp cho baseline; full config bật `upstream_empty_facets_prompt` để phục hồi wording gốc |
| `src/models/llm_client.py` | Provider/model handling, retry parameter | Cần cho self-host; không cùng backend GPT-4o-mini của paper |
| `src/train/trainer_memrec.py` | Sửa provider Stage-ReRank, candidate/feedback/cohort guards và logging | Protocol wrapper chung cho baseline và method; không claim bitwise reproduction |

Config benchmark kế thừa tham số upstream Books (`k=16`, `τ=1800`,
`N_f=7`, LLM rerank, Stage-W warm-up) và ghi đè provider sang self-host.
Vì vậy tên đúng là **full MemRec, upstream-aligned architecture, shared
self-host protocol**; chưa được gọi là “exact paper replication”.

## Benchmark paper vs những gì repo thực sự chạy

| Khía cạnh | Paper chính | Repo hiện trạng trước sửa |
|---|---|---|
| Task | Rank candidate set `N=10` với instruction | Rank 10 candidate |
| Architecture | Collaborative memory + Stage-R + LLM re-rank + Stage-W | Full agent có các stage; M7 thì **không** |
| Books | 7.377 user; full test | 1.000 user trong run lịch sử; protocol mới có dev/held-out |
| LLM | GPT-4o-mini cho LM_Mem/LLM_Rec | Run lịch sử: GPT-4o-mini; hiện key hết hạn |
| Candidates | Fixed ranking task | `ranked_lists` có trên đĩa nhưng evaluator lấy random negatives |
| NDCG@5 | MemRec `0,6601`; SASRec `0,2824` | Run cũ full MemRec 1k: `0,665572`, **không so trực tiếp** |

Nguồn paper: [MemRec ACL 2026, §3/Table 2](https://aclanthology.org/2026.acl-long.2061/).
Run lịch sử `results/instructrec-books_memrec_agent_seed42_20260805_144341.json`
có `n_stage_r_calls=1000`, `n_stage_rr_calls=1000`, `n_stage_w_calls=1000`;
đúng là đã chạy các stage nhưng candidate/protocol khác và `eval_feedback=gt`.
File lịch sử có config credentials; không đưa vào report/public artifacts.
`scripts/run_train.py` đã được sửa để redaction credentials ở run mới, nhưng
không thay đổi file lịch sử/secret của người dùng.

## Audit dữ liệu Books

Nguồn `data/iagent/booksAll_recagent.pkl`, bản copy
`data/processed/instructrec-books/booksAll_recagent.pkl`, và
`data/processed/instructrec-books/instructrec-books.inter`.

| Kiểm tra | 30-user smoke | Toàn bộ |
|---|---:|---:|
| Sequence `asin` khớp `.inter` theo user/position | 30/30 | 7.377/7.377 |
| Candidate list dài 10, 10 item unique | 30/30 | 7.377/7.377 |
| Test target nằm trong list | 30/30 | 7.377/7.377 |
| List không chứa train-history/validation item | chưa tách | 7.377/7.377 |

Target position trong original list gần đều từ 1 đến 10 (full counts:
`760,755,742,732,711,759,728,724,753,713`), nên không có shortcut
“target luôn đầu tiên”. `.inter` ghi `timestamp=position+1` trong từng user,
**không** có global timestamp; không được dựng cross-user online order từ đó.
Metadata có 190.756 item ID; `.inter` chỉ thấy 120.925 interacted item. Random
negative trước đây có thể chọn nhiều item không xuất hiện trong interaction
graph, làm độ khó khác với fixed list. Chưa chứng minh fixed list trùng chính
xác candidate construction nội bộ của paper; cần đối chiếu public artifact
trước khi claim exact replication.

Lệnh tái lập audit: `python scripts/audit_instructrec_books_candidates.py`
(30-user smoke) rồi `python scripts/audit_instructrec_books_candidates.py --full`.
Ordered candidate/target manifest SHA-256 cho toàn bộ 7.377 user là
`a13f7435b5f788503bb1f66608fd072bbc9b48d252bbdd0284126d79c2f87cb6`.

## Khóa development/held-out

Không dùng `RecDataset.valid_data` với candidate tự sinh khác phân phối test.
Thay vào đó chia **user-disjoint trên 7.377 original test rows**; train
histories của mọi user vẫn có thể dùng cho graph/model như benchmark gốc.
Các user có outcome full MemRec từng được xem ở run 1k, run 100 và run 3
được đưa hết vào dev. `src/data/books_protocol.py` khóa seed
`full-memrec-books-v1-20260925`, cho ra:

| Cohort | Số user | SHA-256 của sorted ID list |
|---|---:|---|
| Historically exposed | 1.085 | `d94c17508805b64c709ec625f832e43139b714de59c2aa76fce797316aa84171` |
| Development | 2.000 | `ee83fe12d61df57c5678337458c8a2b556d5bb01c2d39542a52832fb4118f2b0` |
| Held-out primary | 5.377 | `b7751e0ae8f4f23709736ed9b19c457e570aa3edf2a6e0db19c6da938c4af8be` |

Held-out có **0** user overlap với dev/previously exposed. Không mở outcome
held-out để chọn method. Full 7.377 paper-style chỉ báo secondary sau lock.

## Audit code và thay đổi đã thực hiện

- `src/models/memrec_agent.py`, `src/memory/*`, `src/train/trainer_memrec.py`
  chứa full pipeline. `src/temporal_movielens/local_ranker.py` và transition
  residual là pipeline **khác**.
- `src/data/dataset_base.py:load_ranked_lists()` vốn đã đọc pkl theo row/user
  nhưng không được gọi trong `MemRecTrainer`. CLI
  `--use_pregenerated_candidates` từng chỉ set config, không ảnh hưởng tới
  ranking. Giờ trainer thực sự load list và dùng đúng order khi bật flag;
  validate length, uniqueness, target, item range; cấm reuse cho validation.
  Legacy random sampler còn giữ cho exploratory compatibility, **không** dùng
  cho confirmatory Books benchmark.
- `configs/memrec_instructrec-books_full_benchmark.yaml` cấu hình self-host,
  bật fixed lists và `eval_feedback: none`; một run phải truyền endpoint/model
  qua environment trong Slurm step. Bản này vẫn cần smoke GPU trước full run.
- `eval_feedback: gt` trong Books configs có thể làm nhãn test user trước ảnh
  hưởng memory của user sau trong cùng run, nhất là shared item/neighbor memory.
  Đây là **nguy cơ leakage thực tế từ code path**, cần kiểm tra causal trace;
  chưa kết luận được mức tác động. Primary benchmark sẽ tắt test-label write,
  nhưng vẫn thực thi Stage-W trong warm-up từ dữ liệu hợp lệ.
- Trong parallel evaluation, thứ tự memory write/read không cố định; không dùng
  như confirmatory run trước khi có snapshot isolation và kiểm thử. Fixed-list
  benchmark hiện fail-fast nếu bật parallel, hoặc nếu số ranking không bằng số
  user; metric ghi rõ số failed rankings. Benchmark cũng fail-fast nếu
  `eval_feedback != none`. Serial với fresh state là mặc định.
- Fixed-list run ghi một JSONL prediction mỗi user (candidates, order, gold
  position, lỗi) và tính malformed/fallback output là miss. Như vậy có thể
  tính paired delta với SASRec trên đúng user/candidate, không chỉ so metric
  tổng không kiểm toán được.
- Upstream trainer trước đây warm-up **chỉ các user được đánh giá**. Với split
  dev/held-out, như vậy sẽ làm thay đổi collaborative memory coverage giữa
  hai cohort. Đã thêm `warmup_user_scope`: smoke dùng `eval` (30 user), còn
  **full baseline/method bắt buộc `all`** để warm-up cả 7.377 user bằng
  interaction trước test; ranking chỉ chấm cohort được chọn. Metric ghi số
  user warm-up thực tế. Đây là khác biệt protocol cần báo rõ so với run cũ.
- `RecDataset._precompute_user_negatives()` vốn lưu negative pool cực lớn cho
  từng user dù fixed-list test không cần. Fixed-list run giờ bỏ bước đó; warm-up
  sample on-demand bằng RNG riêng cố định theo user/target/seed. Cần đo RAM
  thực tế ở full dry-run trước khi chạy LLM.

## Việc chưa làm / blocking gates

CPU **wiring smoke 30 user** đã chạy bằng
`python scripts/smoke_full_memrec_cpu.py --users 30`: dùng Books thật,
candidate thật, train graph thật và đủ Stage-R/ReRank/W path; `30/30` warm-up
Stage-W, `30/30` test Stage-R và Stage-ReRank, `30/30` per-user prediction,
`0` malformed/failure, `0` test-time Stage-W. Lượt đầu phát hiện lỗi local
`Path` shadowing ở nhánh cohort; đã sửa và chạy lại pass. LLM trong smoke là
**fake schema-shaped client**, do đó mọi NDCG in ra là vô nghĩa khoa học và
không được đưa vào thesis result. Chưa chạy smoke LLM self-host thật.

Đã kiểm thêm `--warmup-user-scope all` bằng fake client: `7.377/7.377`
Stage-R, Stage-ReRank, Stage-W warm-up; sau đó `30/30` prediction hợp lệ,
`0` test write. Đây vẫn chỉ là CPU plumbing. Full dev dự kiến tối đa base
`26.131` LLM calls (`3×7.377 + 2×2.000`), held-out base `32.885`
(`3×7.377 + 2×5.377`) trước retry. Config khóa reserve `10%`, tương ứng
hard cap `28.745` và `36.174` physical requests; smoke 30/eval-scope base
`150`, hard cap `165`. Cả LM_Mem và LLM_Rec dùng chung budget. OpenAI SDK
retry nội bộ được tắt (`sdk_max_retries=0`) để đếm từng HTTP attempt; parser
kiểm schema tối thiểu và mọi API/schema error fail-fast. Các con số này là
**lập kế hoạch tài nguyên**, không phải request đã gửi/GPU đã dùng.
Config còn yêu cầu `MEMREC_SELFHOST_REVISION` không rỗng và Stage-R/W cùng
checkpoint với Stage-ReRank; model/revision được ghi vào metrics. Full run
không được dùng partial warm-up (`eval` scope chỉ cho smoke 20–30 user).

### Baseline SASRec đối chứng cùng protocol

Đã tạo `configs/books_sasrec_baseline.yaml`, `src/baselines/books_sasrec.py`
và `scripts/train_books_sasrec.py`. Đây là **baseline SASRec tự huấn luyện**
theo benchmark mới, không tự nhận là đúng checkpoint/hyperparameter của
SASRec trong paper. Model dùng toàn bộ chuỗi *trước test* của 7.377 user:
train history + item áp chót/validation; không dùng target cuối. Item áp chót
cũng là interaction được full MemRec dùng ở Stage-W warm-up, nên hai arm có
cùng mốc thông tin. Chấm đúng original list 10 candidate/user, cùng dev user,
và assert cùng SHA-256 candidate `a13f7435…2f87cb6`.

Trước khi xem kết quả Books SASRec, đã khóa seed `20260925`, Adam LR `0,001`,
batch `512`, tối đa `50` epoch, early-stop patience `5`, 2 Transformer block,
2 attention head, dropout `0,2` và grid 4 cấu hình: dimension `64/128` ×
sequence length `50/100`. Epoch và kiến trúc được chọn **tự động** bằng
NDCG@5 của 2.000 dev user; không thay rule/trọng số bằng tay sau outcome.
Held-out 5.377 user **không được chấm trong pha chọn model**. Sau train,
runner reload checkpoint thắng, xác minh metric dev và ghi per-user predictions,
hash checkpoint, candidate và predictions. Khi đánh giá held-out sau method
lock, phải dùng chính checkpoint đã chọn, không train lại theo held-out.

GPU workflow SASRec được code hóa thành `--gpu-smoke` rồi `--train-dev` trong
cùng run dir `*-hnv` dưới `/mnt/data/users/anhnct/memrec-hnv/runs/`. Smoke
dùng đúng 30 dev user đầu, chạy một training batch và 30 ranking cho **cả 4**
architecture, lưu peak VRAM và manifest. Train-dev tự từ chối nếu thiếu manifest
hoặc commit/config/cohort/candidate hash không khớp. Đây là gate code; ngoài
ra vẫn phải làm preflight Slurm/GPU, snapshot cả 4 card và kiểm nhả VRAM theo
runbook. Chưa có GPU smoke thật hay checkpoint đã train.

Unit tests hiện tại `82/82` pass; SASRec CPU smoke trên **Books thật** có
`30/30` ranking hợp lệ và training loss hữu hạn với một batch. Đây không phải
model đã train đầy đủ và **không phải** SASRec result để đưa vào thesis.
Ngày 2026-09-25 người dùng cung cấp cách resolve allocation `train_TTS`;
kiểm tra ra duy nhất job `17272` RUNNING. Runbook nội bộ đã đổi sang resolve
động và vẫn giới hạn một H100. Snapshot cả 4 card lúc preflight: 0% utilization,
1 MiB/81.559 MiB mỗi card. Kiểm tra CUDA visibility trên đúng một GPU pass,
process thoát và VRAM về 1 MiB. Repo trên node đã pull đúng `db56215`.
Bốn file Books bắt buộc (`.pkl`, `.inter`, `.instruction`, `.meta`) đã đồng bộ
vào root MemRec riêng; SHA-256 local/server khớp từng file. SASRec CPU smoke
trên node pass 30/30, 5/5 adapter tests pass. Chưa chạy GPU smoke/training.
Launcher `scripts/run_books_sasrec_gpu.sh` ghi snapshot/log, chọn GPU theo
utilization rồi VRAM và từ chối nếu card được chọn đang bận/giữ nhiều VRAM.

1. Kiểm thử sâu causal trace trên **LLM thật** 20–30 case, bao gồm schema
   100%, token/call counts, memory provenance và GPU release. Candidate/cohort
   hash và test-feedback guard đã khóa, CPU plumbing đã pass.
2. Kết nối full MemRec với self-host LLM qua compatible endpoint, đảm bảo cả
   LM_Mem và LLM_Rec dùng cùng checkpoint. Smoke 20–30 case đúng config trên
   allocation hợp lệ; không chạy LLM trên login node.
3. Chạy full MemRec và SASRec trên cùng candidates, rồi mới làm headroom/method.

Xem [THESIS_ROADMAP.md](THESIS_ROADMAP.md) để biết acceptance gate và plan.
