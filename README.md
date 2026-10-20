# MemRec — nghiên cứu cải tiến toàn hệ thống

Mục tiêu luận văn là cải tiến **full MemRec** (collaborative memory, Stage-R,
LLM ReRank và Stage-W), rồi kiểm chứng bằng đối chứng full MemRec và SASRec
trên cùng tập user/candidate. **Chưa có kết quả xác nhận đạt mục tiêu này.**
Temporal Transition/PPR cho local ranker là kết quả thăm dò, không phải method
thesis đã được xác nhận.

Hướng hiện tại là **CM-IRank**: giữ Stage-R/Stage-W, thay Stage-ReRank bằng
chính sách loại dần candidate, fine-tune **Qwen/Qwen3.5-4B bằng PPO** với MPSS.
**Cập nhật đã duyệt15/10:** dùng **LoRA cho actor và critic (+value head)**,
chỉ GPU1; không triển khai đề xuất2 GPU. Full-tuning smoke cũ được giữ riêng,
chưa có kết quả train/ranking LoRA. CPU20 scope/roundtrip phải pass trước GPU.
[Protocol hiện hành §0.54](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#054-approved-lora-amendment-one-designated-gpu-only--2026-10-15).

## Trạng thái hiện hành — 2026-10-20

**Mới nhất: smoke PPO hai role đã PASS và review.** Actor4 +critic4 native
updates, giữ cả20 episode/124 actions (10 valid/10 malformed), MPSS/GAE và
FP32 gradient/Adam/cleanup audits đạt. Khoảng8 phút GPU, peak52,51GiB; model
đã nhả và keeper phục hồi. Đây là **8 updates synthetic bỏ đi, chưa train
Books/chưa có gain ranking**. Primary-memory smoke: CPU20 đã pass, nhưng GPU
dừng ở user8 vì model trích `Item-5922` trong khi input có `User-5922`.
Đã review23 requests/7 writes; model nhả và keeper1 phục hồi. Không có pseudo
cache/PPO Books từ lượt v8. Researcher đã duyệt v9: giữ raw Stage-R, ghi
citation nguồn ngoài context thành warning cho cả hai method; không sửa output,
bỏ user hoặc đổi decoding. Các gate candidates/provenance/Stage-W giữ nguyên.
V9 đã code và qua764 tests; CPU→GPU→handback→CPU review là chuỗi tiếp theo.
Allocation đã được researcher đổi; tiếp tục chỉGPU1, luôn bật lại keeper sau
task. Các đoạn launch/gate phía dưới là lịch sử.
[Kết quả và bước tiếp §0.65](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#065-both-role-ppo-smoke-reviewed-pass-primary-train-only-memory-next--2026-10-20).
[Primary failure và quyết định cần chốt §0.66](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#066-primary-train20-failure-reviewed-citation-policy-decision--2026-10-20).
[Protocol v9 đã duyệt §0.67](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#067-approved-raw-primary-source-citation-warnings-v9--2026-10-20).

**Mới nhất20/10:** LoRA rollout/reference v2 đã PASS và được đối chiếu độc lập:
20/20 responses hợp lệ/60 calls, native cold-restore actor+critic/Adam/RNG
khớp; reference bằng `disable_adapter()` khớp20 frozen controls (delta0).
ESS0,99687, weights0,87462–1,08242, không vượt cap2; 282,02s, peak52,51GiB.
Model đã nhả1MiB và keeper1 phục hồi. **0 updates, chưa train PPO Books**.
Gate kế tiếp đã code/test:20 synthetic episodes ×9 lượt, native MPSS/GAE
xuyên episode rồi5 PPO updates/role (bỏ đi), không dùng Books outcomes.
CPU smoke phải pass trước GPU1; không đổi LoRA rank/LR, budget hoặc primary.
CPU v1 đã bắt lỗi padding biến độ dài (10 vs9), dừng **trước GPU/updates**.
Đã sửa adapter theo native dynamic microbatch padding; giữ nguyên method,
seed/LR/budget và log FAIL. V2 phải rerun CPU trước khi nối GPU1.
V2 CPU đã **PASS360 probes**, nhưng GPU chỉ10/20 episode hợp lệ (125 calls,
chọn lại nhãn đã loại), nên0 updates; model đã nhả/keeper1 phục hồi.
Đang bổ sung đúng §7.4: giữ mọi episode lỗi, terminate/phạt−1 và học từ toàn bộ
batch, không repair/filter/tune prompt. Đây là gate cơ chế, không đổi FAIL
chất lượng cũ thành PASS ([§0.63](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#063-cpu-v2-reviewed-pass-gpu-malformed-action-failure-implement-74--2026-10-20)).
[Kết quả và protocol gate mới §0.61](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#061-lora-rolloutreference-reviewed-pass-complete-episode-ppo-smoke--2026-10-20).
V3 source`eda05eb`/723 tests pass đã deploy, đang chạy CPU→GPU1 có xử lý
episode lỗi đúng protocol. **V3 terminal:** native GAE đã chạy,4 critic PPO
updates (synthetic/bỏ đi), actor0; harness kiểm tra gradient sau lúc native
context đã xóa nó nên gate vẫn FAIL. GPU đã nhả, keeper1 phục hồi.
Đang sửa đúng lifecycle kiểm tra, không tắt cleanup/nới guard hay đổi recipe
([§0.64](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#064-native-gae-and-four-critic-updates-reached-train-context-audit-fix--2026-10-20)).
V4 source`fd6f741`/732 tests pass đã kết thúc; kết quả actor4+critic4 updates
đã được review và thay thế trạng thái launch cũ ở đầu trang/§0.65.

| Phần việc | Bằng chứng / giới hạn |
|---|---|
| Benchmark đối chứng | Full MemRec NDCG@5 **0,747918**, SASRec **0,343320**, cùng 200 dev users/candidates; MemRec warm-up 700 users, không phải full-data/paper replication |
| Data/protocol CM-IRank | Khóa 1.497 train / 300 validation; snapshot target-blind và 3.594 candidate sets v2 đã audit CPU, không có cờ trong 7 probe cố định; chưa chứng minh không còn shortcut |
| Policy infrastructure | Native MPSS/GAE/PPO smoke đã pass4 actor+4 critic updates, all20 episodes kể cả lỗi; chỉ synthetic/bỏ đi, **chưa train PPO Books** |
| Real memory | V6 fail đã review; **v7 secondary control + independent review pass**, 20/20 user, 100 calls, 20 warm-up writes / 0 pseudo writes; không chứng minh semantic grounding hay thay primary |
| Format / PPO runtime | Format-only SFT 256 examples /64 updates và IS-corrected native PPO smoke đã pass; **chưa PPO Books**. Các strict HF–vLLM parity FAIL lịch sử vẫn giữ, không chọn lượt tốt nhất để promote |
| Method result | **Chưa có checkpoint/score CM-IRank**; held-out 5.377 users vẫn niêm phong |

**LoRA joint GPU PASS đã review19/10:** hai native4B actor/critic cùng nằm
trên **một H100**,20 mẫu/role, một update bỏ đi/role; worker137,41s,
toàn gate155,98s. Peak allocated**37,67GiB**, reserved**39,39GiB**;
actor train64,93M adapter parameters, critic thêm2561 value-head parameters.
20 artifacts CPU/GPU đã đối chiếu SHA/source,40 vector repeat delta0;
model nhả về1MiB rồi keeper1 được phục hồi. **Chưa có PPO/ranking result**:
rollout/reference chưa co-host, checkpoint/GAE chưa được gate này chứng minh.
Tiếp theo checkpoint native đầy đủ adapter/head/Adam/scheduler/RNG trên20
mẫu, rồi rollout/ref sync và multi-turn. Không tăng cap RAM/GPU, không đổi
LoRA rank/LR hay dùng GPU0.
[Kết quả và gate kế tiếp §0.58](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#058-joint-lora-gpu-reviewed-pass-native-checkpoint-gate--2026-10-19).
**Checkpoint cũng đã PASS và review:** actor496 /critic498 tensor trainable,
Adam/scheduler/RNG và20 readouts/role khôi phục đúng; cold-process restart
chưa nằm trong gate đó. Native265,18s/coordinator283,34s, VRAM peak53,04GiB;
GPU nhả1MiB rồi keeper1 phục hồi. Tiếp theo CPU export adapter → vLLM20 mẫu
mới → native cold-restore/reference, chạy tuần tự trên GPU1, không co-host.
[Checkpoint và rollout gate §0.59](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#059-native-checkpoint-reviewed-pass-sequential-lora-rolloutreference-gate--2026-10-19).
Source`844284d` đã deploy,605 tests pass. Profile v2 sửa hạch toán clean file
LRU, giữ nguyên cap64GiB/margin20GiB và budget working-set đã đo. CPU20
export đã **PASS58,16s**,496 tensor/64,93M parameters khớp byte, CUDA chưa khởi
tạo. Launch này đã kết thúc thành công và được review20/10 ở §0.61;
không dùng trạng thái launch19/10 làm tiến độ hiện hành.

**Gate full-tuning lịch sử:** corrected-rollout v2 **PASS**: 20/20 responses sinh thật đúng
format, 0 truncation; native FSDP2 chấm 40 lượt, repeat delta0. CPU review
sequence IS native đạt ESS0,99363, weights0,86660–1,15159, không mẫu vượt cap2.
GPU task3,94 phút; model đã nhả về1MiB và keeper1 phục hồi. Artifacts được
đối chiếu SHA và tính lại độc lập; đây vẫn là **0 optimizer updates / không
có kết quả ranking**, không đổi các strict parity FAIL cũ.
[Kết quả và giới hạn §0.48](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#048-generated-rollout-v2-and-native-correction-review-pass--2026-10-14).

**Các gate full-tuning đã hoàn thành trước quyết định LoRA (lịch sử):** CPU20 packing/loss/memory → actor native FSDP2/AdamW riêng,
FP32 master/BF16 forward, một update synthetic bỏ đi; sau đó critic, checkpoint
và multi-turn PPO/GAE trước research training. Không suy từ smoke BF16 inference
rằng full PPO vừa một H100.
[Protocol optimizer §0.49](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#049-native-actor-optimizer-and-memory-smoke-preregistered--2026-10-14).
**Actor optimizer v2 PASS toàn pipeline:** 20 mẫu/một update bỏ đi,
119,71s; peak allocated68,72GiB/reserved69,01GiB, đủ FP32 gradients/Adam states
cho toàn text backbone. 21 artifacts/SHA/source được review độc lập; model
nhả1MiB rồi keeper phục hồi. V1 lỗi integration vẫn giữ FAIL.
[Chi tiết §0.50](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#050-native-actor-update-passed-v1-coordinator-release-integration-failed--2026-10-14).
**Critic cũng PASS toàn pipeline:** 20 samples, một native full-text+scalar-head
update bỏ đi,116,72s, peak allocated68,72GiB/reserved68,96GiB. 21 artifacts
đã review; actual GPU release1MiB/keeper phục hồi. Chưa có fullPPO/ranking gain.
[Kết quả actor và plan critic §0.51](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#051-actor-v2-reviewed-pass-native-critic-optimizer-preregistered--2026-10-14).
**Resource audit full-tuning (lịch sử, trước quyết định LoRA):** xác nhận3 layout native đã kiểm
tra không an toàn dưới1 H100+64GiB RAM. Riêng hai model+Adam states đã96,49GiB;
chưa tính reference/rollout/workspace. Không thử OOM/đổi precision/LoRA/critic
để lách budget; keeper1 không bị dừng, MemRec không giữ model GPU.
Chọn thêm host RAM (ít nhất128GiB, vẫn cần đo fit) hoặc cấp riêng2 H100 rồi
smoke lại topology. Không hủy job reserved/động workload khác để lấy tài nguyên.
[Review critic và resource gate §0.52](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#052-critic-reviewed-pass-combined-resource-audit--2026-10-15).
**Đề xuất cũ, không được cấp quyền 2 GPU, đã thay bằng LoRA§0.54:** dùng trước2 H100 tối đa2 giờ
(4 GPU-giờ) để qualify topology/save-resume và đo throughput. Cache memory đầy
đủ còn thiếu; mốc5 epochs có thể cần50–100+ giờ/nhánh/seed, chưa phải ETA đã đo.
[Căn cứ đo, giới hạn và kế hoạch §0.53](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#053-proposed-two-h100-time-resource-plan--2026-10-15).
Tổng quan phương pháp vẫn ở [design §0.14](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#014-overall-review-and-next-gates--2026-10-07).
Các đoạn dưới là lịch sử gate; không dùng trạng thái launch cũ làm tiến độ hiện tại.
CPU review v6 đã đối chiếu 35 request/schema và replay đúng 11 lần Stage-W;
không sửa ID, bỏ user, tune prompt hay nới hard gate để thông qua smoke.
Researcher đã duyệt **v7 constrained-decoding secondary control**: schema chỉ cho
phép ID từ input, không thay upstream-aligned primary; cần compile schema CPU
rồi smoke thật 20 user/100 calls và review riêng ([§0.15](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#015-approved-secondary-input-id-decoding-control--2026-10-07)). Sau memory gate mới tới real N−1/direct smoke,
one-card PPO compatibility và tiny PPO; không tự nối sang full training.
**V7 terminal:** 16,55 phút / 240.750 tokens / 0 cache hits, peak observed VRAM
48 GiB; còn 67 candidate-context citations phải giữ cảnh báo, không chứng minh
semantic grounding. 26/26 artifacts hash-match; GPU về 1 MiB trước handback.
`omni-gen-1` đã chạy lại; từ nay GPU task xong phải unload rồi bật lại riêng
keeper theo runbook, kể cả khi còn CPU review. GPU0/job reserved được giữ nguyên.
Independent review đã xác nhận exact schemas/domains, 20 raw writes và 20 pseudo
inputs read-only, không có unknown ID. Bước hiện tại: **N−1/direct functional
smoke của Qwen3.5-4B**, 20 frozen inputs secondary / tối đa 200 generations,
CPU token audit đã pass **40 prompts / 709–1.044 tokens**, 319 tests pass;
**V2 GPU0 terminal:** 86,51 giây /177 generations, N−1 hợp lệ **14/20**, direct
**0/20** vì thiếu answer tags (không truncation/repair). GPU đã unload rồi
keeper0 chạy lại; GPU1/job được bảo vệ. 20/20 artifacts hash-match và CPU replay
khớp. Lỗi format dẫn tới đề xuất **format-only SFT trước PPO**
([§0.18](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#018-real-qwen35-n1direct-smoke-outcome-and-format-decision--2026-10-09)).
Researcher đã cho tiếp tục: **format-only SFT**256 synthetic examples,64 updates
đã khóa; trước đó smoke20 examples/5 updates và save/reload. Giữ no-SFT controls,
không nhãn Books/tune ranking;340 tests pass. CPU data/mask/token/hash gate đã pass
256 examples +20 holdout /80 prompt audit. Đã có xác nhận keeper0; v1 GPU preflight
dừng trước reclaim vì GPU1 đổi sang workload TTS. V2 sửa guard để GPU1 luôn
read-only, giữ nguyên SFT recipe; tiến độ và receipts ở design §0.19–0.20.
**SFT smoke PASS:**5 updates/134,44s; synthetic N−1/direct20/20; save/reload20
logit probes delta0. GPU đã unload và keeper bật lại. Sau review và xác nhận mới,
**Full SFT đã xong:**64 updates/11,31 phút; N−1/direct đều20/20 trên synthetic
và cùng20 Books inputs; save/reload delta0,24 artifacts hash-match. GPU đã unload,
keeper chạy lại. Đây chỉ là **format initializer**, chưa có gain NDCG/Hit.
Bước tiếp: PPO actor/critic/reference compatibility trong env riêng. Researcher
đã đồng ý chọn/re-pin CUDA12-compatible (design §0.23); profile CPU đầu tiên là
VeRL0.9 /Torch2.11+cu129 /vLLM0.20+cu129 /Transformers5.10.1 /TRL0.25.1.
**CPU gate đã pass**:237 dependencies/pip-check/20 miniature probes,11,11 phút,
12 artifacts hash-match. Đây vẫn là **candidate runtime**, chưa phải PPO4B đã
xác nhận. **FA2 build/import đã pass** sau67,82 phút CPU resume,6 artifacts
khớp SHA. CPU prep20 mẫu đã pass22,73s. **Smoke full4B v1 đã dừng sau53,75s**:
kernel và actor/reference20/20 pass, nhưng guard critic của harness giả định
sai cấu trúc native VeRL. GPU đã nhả, keeper0 bật lại;14 artifacts hash-match.
V2 đã sửa harness theo native scalar `score`/logits và tied embeddings.
**CPU v2 pass20/20 trong20,10s**, backbone transfer/head gradients/roundtrip
đạt,3 artifacts SHA-match;386 tests pass. **Full GPU v2 pass** kernel/actor/
critic20/20,68,09s, checkpoint delta0; allocated27,78GiB/reserved36,41GiB.
GPU đã nhả, keeper0 phục hồi,17 artifacts SHA-match. Đã tự triển gate CPU
và mã smoke tiếp theo: vLLM/HF response-logprob parity trên cùng20 synthetic
inputs. **Chưa PPO/quality gain**, native worker/optimizer memory vẫn cần gate
(§0.24–0.29). **Parity CPU v1 pass20 mẫu/17,39s; GPU v1 lỗi khởi động
vLLM do thiếu image processor trong checkpoint text-only, 0 completion**, chưa
đo được numerical parity. GPU đã nhả, keeper phục hồi. V2 dùng native text-only
mode và kiểm nhánh registry trên CPU trước retry; giữ nguyên weights/tokens/ngưỡng
([§0.30](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#030-parity-v1-startup-fail-native-text-only-retry--2026-10-12)).
**V2 cũng dừng trước completion** do optional FlashInfer/DeepGEMM startup,
không phải kết quả numerical/ranking fail. Model đã nhả và keeper phục hồi.
V3 dùng native Triton/skip optional warmup, cùng20 mẫu và ngưỡng; user chuyển
quyền single-card sang GPU1, tự handoff keeper1 với guards/handback đầy đủ.
Chi tiết và giới hạn ở [§0.32](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#032-parity-v2-terminal-fail-gpu1native-kernel-v3-amendment--2026-10-12).
**V3 đã chạy20/20 requests nhưng parity chỉ17/20 pass**: ba token nhãn vượt
ngưỡng sai lệch0,05; không nới ngưỡng/bỏ mẫu. GPU1 đã nhả và keeper1 phục hồi.
Đang triển chẩn đoán attention/readout/precision cố định trên cả20 mẫu,0LLM
calls/updates, không chọn profile hay promote PPO ([§0.34](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md#034-parity-v3-numerical-fail-not-startup-fail-fixed-precision-diagnosis--2026-10-12)).
Phase pass và đủ điều kiện sẽ tự triển phase tiếp; chỉ
end-turn để chờ xử lý dài hoặc cần researcher quyết định.
PPO worker/rollout/optimizer memory và canonical primary memory vẫn còn gate.
Không đổi driver/env baseline, không tự dùng GRPO/LoRA/2GPU.
Chưa có score/checkpoint PPO CM-IRank hoặc quyết định thay primary.

Migration và cleanup nguồn cũ đã hoàn tất, CPU smoke sau cleanup pass. Mọi compute
mới theo root/account/allocation hiện hành trong runbook nội bộ; SSH dùng chung
ControlMaster, không hủy job reserved và không giữ model idle. Chi tiết lịch sử
lỗi hạ tầng được giữ ở design §0.5–0.13, không nhầm với tiến độ nghiên cứu.

## Bản đồ tài liệu — chỉ 5 tài liệu dự án

| Tài liệu | Nguồn chuẩn cho |
|---|---|
| README này | Điểm vào, cấu trúc repo và setup |
| [CM-IRank design](docs/CM_IRANK_FULL_IMPLEMENTATION_DESIGN.md) | Kế hoạch nghiên cứu/triển khai mới, chuẩn học thuật, G0–G3 và ledger tiến độ |
| [BOOKS_BENCHMARK](docs/BOOKS_BENCHMARK.md) | Dữ liệu/protocol InstructRec Books, full-agent baseline, SASRec, Stage-R trial, hashes và tái lập |
| [EXPLORATORY_HISTORY](docs/EXPLORATORY_HISTORY.md) | Phương pháp/kết quả Amazon–MovieLens, các hướng dừng, Laya, Goodreads và audit trail |
| [H100_RESOURCE_RULES](internal_docs/H100_RESOURCE_RULES.md) | Quy tắc nội bộ về Slurm/GPU; không đưa nội dung hạ tầng này vào tài liệu công khai |

Roadmap cũ đã được hợp nhất vào CM-IRank design; các hướng đã đóng nằm trong
lịch sử thăm dò. Artifacts, per-user predictions, configs và source code vẫn là bằng
chứng gốc; tài liệu chỉ là bản diễn giải. Tệp metadata đi kèm dataset, manifest
dependency và cache sinh tự động không phải tài liệu nghiên cứu của repo.

## Exploratory experiment surface (không phải thesis benchmark)

```text
configs/temporal_amazon_books_2014/
├── dataset_audit.yaml
├── p7v2_transition_ppr.yaml
└── transition_ppr_replication_200.yaml

src/temporal_books/
├── common.py
├── current_support.py
├── p0_audit.py
├── p7_selfhost_local.py
└── p7_transition_ppr.py
```

Amazon Books Kaggle nằm ở `data/amazon_books/` (`raw/` và `artifacts/`),
InstructRec Books ở `data/processed/instructrec-books/`, MovieLens 32M ở
`data/ml-32m/`, và Goodreads raw ở `data/goodreads/`. Không gộp metric giữa
các nguồn này: task, candidate construction và mức độ tin cậy thời gian khác
nhau. Raw/derived data được quản lý theo `.gitignore` hiện hành.

## Setup và kiểm tra

```bash
conda create -n memrec python=3.10
conda activate memrec
pip install -r requirements.txt

python -m pytest -q tests/temporal_books
python -m pytest -q tests/test_cmirank_g0.py
python scripts/cmirank/02_preview_policy_split.py --seed cmirank-books-v1-draft-20260930 --min-prefix-length 5 --verify-locked configs/cmirank/policy_split_manifest.json
python scripts/cmirank/03_audit_prefix_snapshot.py --query-users 20
python scripts/cmirank/04_smoke_pseudo_memory_cpu.py --users 20
python -m src.temporal_books.p0_audit
python -m src.temporal_books.p7_transition_ppr --prepare
python -m src.temporal_books.p7_transition_ppr --graph-smoke
```

Self-host LLM phải chạy smoke-first qua Slurm theo
[runbook nội bộ](internal_docs/H100_RESOURCE_RULES.md); không chạy model trên
login node và phải nhả GPU ngay khi task kết thúc. Lệnh ở trên chỉ là smoke
cho pipeline graph thăm dò, **không** tái lập kết quả full MemRec. Lệnh Books,
config, model revision và các cổng kiểm tra nằm trong
[Books benchmark](docs/BOOKS_BENCHMARK.md).

## Upstream MemRec

Core memory/model/training modules trong `src/{memory,models,train,data}` và các
config `configs/memrec_*.yaml` là điểm khởi đầu để tái lập full baseline. Dữ
liệu Books có original fixed test candidates trong
`data/processed/instructrec-books/booksAll_recagent.pkl`; bật
`--use_pregenerated_candidates` khi đánh giá Books. Hiện chỉ có baseline
full-architecture **700 warm-up/200 dev user**; chưa có full 7.377-user hay
held-out result, và chưa có method đạt improvement gate. Paper gốc:

> Chen et al., “MemRec: Collaborative Memory-Augmented Agentic Recommender
> System,” ACL 2026. https://aclanthology.org/2026.acl-long.2061.pdf
