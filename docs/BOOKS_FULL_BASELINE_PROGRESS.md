# InstructRec Books — tiến độ benchmark full MemRec và SASRec

**Cập nhật:** 2026-09-25. Đây là kết quả **development**, không phải held-out
hay kết quả cuối của phương pháp thesis. Protocol và lý do reset baseline ở
[FULL_MEMREC_BASELINE_AUDIT.md](FULL_MEMREC_BASELINE_AUDIT.md) và
[THESIS_ROADMAP.md](THESIS_ROADMAP.md).

## Dữ liệu và so sánh được phép

- 7.377 user, original candidate list 10 item/user; ordered candidate SHA-256
  `a13f7435b5f788503bb1f66608fd072bbc9b48d252bbdd0284126d79c2f87cb6`.
- Development 2.000 user, SHA-256
  `ee83fe12d61df57c5678337458c8a2b556d5bb01c2d39542a52832fb4118f2b0`.
  Held-out 5.377 user, SHA-256
  `b7751e0ae8f4f23709736ed9b19c457e570aa3edf2a6e0db19c6da938c4af8be`.
  Held-out outcome chưa được mở cho bất kỳ lựa chọn model/method nào.
- Cùng original candidate order, target và mốc history trước test. SASRec train
  trên train history + item áp chót của mọi user; không train trên test target.
  Full MemRec sẽ warm-up Stage-W từ interaction trước test của mọi user và
  không ghi nhãn test vào shared memory trong evaluation.
- Bốn file cần trên node (`.pkl`, `.inter`, `.instruction`, `.meta`) đã được
  đối chiếu SHA-256 local/server lần lượt:
  `5bdc05f44db5f7c4fa2ca05db70bdaf9adf31c68fa2d7425695e83f63cbb7dbc`,
  `891d4007af3e63eef40493520218f644478250ec35d4443faa1d2b96384efc24`,
  `44a3ae1b3ab49a4328eebd50ed9e5c1e7bbe2819e454a44c6d50662477916150`,
  `69b46870c62e207fca11b6d695d203efef5ee1925f33428f43c1e6eb6ccec88c`.

## SASRec development baseline — run v3 được giữ

Grid và seed đã khóa **trước** outcome Books trong
`configs/books_sasrec_baseline.yaml`: dimension 64/128 × max sequence 50/100,
2 block, 2 head, dropout 0,2, Adam LR 0,001, batch 512, tối đa 50 epoch,
patience 5, seed 20260925. Chọn architecture và epoch tự động bằng NDCG@5
trên dev. Không tune tay sau khi xem score.

| Architecture | Epoch tốt nhất | Dev NDCG@5 |
|---|---:|---:|
| d64_s50 | 1 | 0,303372 |
| d64_s100 | 1 | 0,295173 |
| **d128_s50 (được chọn)** | **1** | **0,321127** |
| d128_s100 | 1 | 0,302800 |

Mọi architecture được early-stop sau epoch 6. Loss huấn luyện tiếp tục giảm
trong khi dev NDCG giảm; đây là quan sát trên development, **không** phải lý
do để đổi grid sau kết quả. Checkpoint thắng được reload rồi chấm lại 2.000
dev user; mọi candidate, target, user ID, ranking và target position được
kiểm độc lập với dữ liệu gốc. Kết quả:

| Metric | Dev, 2.000 user |
|---|---:|
| Hit@1 | 0,125500 |
| NDCG@3 | 0,239462 |
| Hit@3 | 0,327500 |
| **NDCG@5** | **0,321127** |
| Hit@5 | 0,527500 |
| NDCG@10 | 0,471692 |
| Hit@10 | 1,000000 — tất yếu với 10 candidate |

Run ID `sasrec-books-dev-v3-hnv`, code SHA
`8758a22d557c08d7043ae72b25c53b26c6f18a91`, config SHA-256
`0758e127cae1be3d66b95456d6b885b0ec3bd9bc54bbabe862b1461103f80ba0`.
Checkpoint thắng SHA-256
`0c59d0c0563816e729729574d3675ff83d04e9fffd594a35b67e0152522eb8fd`;
2.000-row prediction JSONL SHA-256
`f4827100061c3bccb9a27d91813cc4fd365b8cbdff47b2dcf330e7fb87a20c69`.
Artifacts và GPU logs đã kéo về thư mục local
`results/full_memrec_books_baselines/sasrec-books-dev-v3-hnv/` (git-ignored),
đồng thời còn trong `/mnt/data/users/anhnct/memrec-hnv/runs/` trên node.
Chưa chấm held-out; checkpoint này phải được giữ cố định cho test về sau.

Smoke GPU v3 trên đúng một H100: 30 user × cả 4 architecture, 30/30 ranking
hợp lệ và loss hữu hạn từng arm; tối đa 569.180.672 byte peak torch allocation,
17 giây. Full train dev: peak torch allocation 1.750.859.264 byte trên card
84.929.347.584 byte, 32 giây. GPU 3 được chọn từ snapshot 4 card theo
utilization rồi memory; trước/sau train đều 4 MiB, không có process MemRec
giữ VRAM sau khi thoát. Allocation `train_TTS` job `17272` vẫn RUNNING sau run.
Peak nêu ở đây là `torch.cuda.max_memory_allocated`, không thay thế snapshot
toàn card; cả hai đã được lưu. Run v1 fail trước training do reset peak stats
trước `torch.cuda.init()`, không có checkpoint. Run v2 tái lập metric nhưng thiếu
peak full-train nên giữ để audit, **không** dùng làm run báo cáo chính.

## Full MemRec và giới hạn kết luận hiện tại

Full MemRec đã pass 30-user real-LLM smoke bằng Qwen3-30B-A3B-Instruct FP8,
revision `5a5a7763…90db`, đủ Stage-R/RR/W warm-up, 150 physical requests,
0 failure. Smoke dùng warm-up chỉ 30 user nên NDCG@5 `0,689081` **không phải**
baseline 2.000-user. Do đó **chưa thể so
SASRec với full MemRec, chưa có paired delta, chưa có method gain**. Số SASRec
ở đây cũng không so trực tiếp với SASRec/ MemRec trong paper: khác cohort,
checkpoint/hyperparameter và điều kiện LLM. Mốc tiếp theo là pin self-host
model/revision/backend đã khóa và chạy full MemRec dev với all-user warm-up.
Giữ sealed held-out cho sau khi method/config đã khóa.

Dry-run CPU trên node ở commit `eb9e0f4`: 30 user ranking sau all-user warm-up
7.377/7.377, `0` failure và `0` test-label write. Fake client ghi 14.784
Stage-R/W requests và 7.407 Stage-ReRank requests = 22.191 lời gọi giả lập,
đúng base budget `3×7.377 + 2×30`; physical LLM request thật bằng **0**.
Elapsed 25,51 giây; peak RSS 1.342,1 MiB. Đây chỉ là footprint/đường code,
**không** dự báo latency hay chất lượng của LLM thật. Data/candidate/cohort
contract đã được validate lại ở node trước dry-run.

Để chịu được run LLM kéo dài, `books-memrec-llm-dev-v1-hnv` dùng journal
transactional từng user và request ledger SQLite. Dry-run CPU mới 7.377 user
warm-up + 30 eval, 22.191 fake calls, 0 failure; chạy lại replay cho đúng
prediction SHA-256 `cb365a1a…ba987a688ef`, bằng non-journal CPU run. Đây là
kiểm tra tính đúng của resume, không phải metric nghiên cứu. Full-dev gate
chỉ được báo sau 2.000 prediction, 7.377 warm-up journal row và GPU trả VRAM.

Run full-dev đã khởi chạy nhưng **dừng theo yêu cầu người dùng do chi phí** ở
129/7.377 warm-up user, 0/2.000 eval user; request ledger ghi 299/28.745
physical attempts được reserve. Đây là dữ liệu dở dang, **không có metric
full MemRec**. Session và các process MemRec/vLLM đã thoát; GPU 3 giảm từ
1 MiB trước run về 4 MiB sau dừng. Journal và cache được giữ nguyên để audit,
không tự động resume; đợi quyết định protocol/nguồn lực mới.
