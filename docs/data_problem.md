> **Cập nhật triển khai, 2026-08-26:** Amazon Reviews’23 vẫn là lựa chọn scale-up
> chính. Tuy nhiên, pilot đã materialize trước trên Kaggle Amazon Books Reviews
> (Amazon 2014 repackage) vì file đã có local và nhỏ hơn. Kết quả/gate/code của
> pilot nằm ở [AMAZON_BOOKS_2014_TEMPORAL_PLAN.md](AMAZON_BOOKS_2014_TEMPORAL_PLAN.md):
> P0/P1 pass theo strict-past daily batches, 0 LLM calls; P2 1,000-request chưa
> chạy. Nội dung dưới đây là proposal cho Amazon Reviews’23, không phải mô tả
> dataset đang được chạy ngay.

Mình chọn **Amazon Reviews’23 – Books** làm dataset chính cho hướng `multi-hop memory evolution`. Nó khớp nhất với bài toán hiện tại của bạn và giải quyết đúng thiếu hụt lớn của InstructRec-Books: **mỗi interaction có timestamp Unix thực**, nên có thể sort toàn bộ `(user, item, event)` theo một timeline chung và replay Stage-W đúng nhân quả. Dataset cũng có review text và item metadata, nên vẫn phù hợp với semantic memory của MemRec. ([amazon-reviews-2023.github.io][1])

### Vì sao Amazon Reviews’23 Books là lựa chọn tốt nhất

| Dataset                     | Global event time                  | Text/metadata            | Gần MemRec hiện tại    | Hợp multi-hop evolution |
| --------------------------- | ---------------------------------- | ------------------------ | ---------------------- | ----------------------- |
| **Amazon Reviews’23 Books** | **Rất tốt – Unix timestamp/event** | **Rất giàu**             | **Rất gần**            | **★★★★★**               |
| MovieLens 32M               | Rất tốt – Unix timestamp/event     | Trung bình: genre + tags | Khá gần recommendation | ★★★★                    |
| Yelp Open                   | Có date/review event               | **Rất giàu**             | MemRec đã dùng Yelp    | ★★★★                    |
| MIND News                   | Có impression time                 | Article text tốt         | Khác bài toán hơn      | ★★★                     |

Amazon Reviews’23 thậm chí cung cấp sẵn các split theo **absolute timestamp**, tức train/validation/test được cắt tại các mốc thời gian chung, thay vì leave-last-out riêng từng user. Với Books 5-core, trang chính thức báo khoảng **776K users, 495K items, 9.5M interactions**; absolute-time split tương ứng khoảng **8.7M train / 426K validation / 328K test interactions**. ([amazon-reviews-2023.github.io][2])

Điểm này rất phù hợp cho experiment của mình:

```text
t0                  t_train                t_val              t_test
│────────────────────│──────────────────────│───────────────────│
     replay events          validation            test

(u1,i4,t1)
(u7,i2,t2)
(u2,i4,t3)
...
      ↓
Stage-W update
      ↓
1-hop / 2-hop propagation
      ↓
memory graph evolves causally
```

Tại event ở thời điểm `t`, mọi routing, buffer, memory và graph structure **chỉ được phép dùng event có timestamp `< t`**. Đây là thứ dataset hiện tại của bạn không cung cấp đủ để test nghiêm túc.

### Nhưng mình không khuyên chạy full Books ngay

9.5M interaction là quá lớn nếu mỗi interaction lại gọi LLM để update memory. Mình sẽ tách thành hai bước.

**P0/P1 pilot:** vẫn dùng Amazon Reviews’23 Books nhưng materialize một temporal subgraph khoảng **10K–30K users**, được chọn hoàn toàn từ statistics của training period. Sau đó replay vài trăm nghìn event bằng algorithmic routing/embedding; LLM chỉ dùng ở những chỗ thực sự cần tạo semantic update.

Nếu oracle có headroom rõ mới scale cohort lên. Như vậy không cần đổi domain rồi sau đó lại phải chứng minh transfer.

Một detail rất quan trọng: mình sẽ ưu tiên **raw/0-core hoặc tự tạo k-core chỉ từ pre-cutoff training period**. Amazon có sẵn 5-core rất tiện, nhưng 5-core được tính trên toàn bộ dataset; với một thesis nhấn mạnh causal temporal replay, dùng tương tác tương lai để quyết định node nào tồn tại trong cohort tạo một dạng survivorship bias nhẹ. Amazon Reviews’23 cũng cung cấp 0-core/pure IDs nên mình có thể tự xử lý. ([amazon-reviews-2023.github.io][3])

Nếu muốn nhanh hơn trong giai đoạn engineering, có thể dùng official 5-core trước, rồi final experiment chuyển sang temporal filtering nghiêm ngặt.

## Lựa chọn thứ hai: MovieLens 32M

Nếu mục tiêu chỉ là **chứng minh mechanism multi-hop propagation có hoạt động hay không** với temporal protocol cực sạch, MovieLens 32M còn dễ hơn. Nó có 32M ratings từ hơn 200K users và 87K movies; mỗi rating có trường

`userId, movieId, rating, timestamp`

với timestamp là Unix seconds UTC. Free-text tag cũng có timestamp riêng. ([GroupLens Files][4])

Ưu điểm lớn là causal replay rất sạch và không có chuyện timestamp mơ hồ.

Nhược điểm lại quan trọng với đề tài của bạn: item information chủ yếu là title, genre và user tags, không có rich review/item description tương đương Amazon. ([GroupLens Files][4]) Do đó nếu multi-hop semantic memory không cải thiện, rất khó biết do phương pháp hay do semantic evidence nghèo.

Mình xem MovieLens là **sanity-check dataset tốt**, không phải dataset chính.

## Yelp thì sao?

Yelp cũng rất hấp dẫn vì MemRec đã benchmark trên Yelp và dataset hiện có gần **7 triệu reviews / 150K businesses**, cùng business attributes và review text. ([Yelp Business][5]) Review record có `user_id`, `business_id`, stars, text và date. ([DOI][6])

Nhưng nếu mục tiêu cụ thể là **strict global event replay**, Amazon vẫn sạch hơn vì timestamp Unix cấp event được công bố rõ ràng. Yelp mình sẽ giữ làm **cross-domain validation** sau khi phương pháp đã chạy trên Amazon.

## Mình sẽ định nghĩa dataset/protocol mới thế này

Không còn:

```text
per-user history
→ frozen graph
→ inject 2-hop into prompt
→ rerank
```

mà là:

```text
Amazon Reviews'23 Books
        ↓
sort ALL events globally by timestamp
        ↓
training-time chronological replay
        ↓
for each event (u, i, t):
    update local memory
    propagate 1-hop
    optionally route hop-2 packet
    update/buffer only using state at t-
        ↓
freeze/evaluate at temporal checkpoints
```

Sau đó P1 chỉ cần ba arms:

1. **No propagation / local memory**
2. **MemRec-style 1-hop propagation**
3. **Oracle item-only 2-hop propagation**

Chưa cần buffer, router hay RecNet-like algorithm. Trước tiên hỏi đúng một câu:

> Nếu tại mỗi thời điểm ta biết *oracle* packet 2-hop nào nên propagate, thì memory evolution có giúp dự đoán **future interactions** tốt hơn 1-hop hay không?

Nếu `oracle 2-hop - 1-hop < +0.05 NDCG@5`, mình lại dừng sớm. Nếu oracle mạnh, lúc đó mới đáng thiết kế `hop-aware routing + buffer + receiver gate`.

Điều này cũng giải quyết vấn đề của MH2 hiện tại: MH2 đã chứng minh khá chắc rằng **raw read-side 2-hop replacement** không có headroom dưới equal budget, với oracle `−0.0094` NDCG@5.  Dataset mới không nhằm “rerun MH2 trên dataset khác”, mà để test **một hypothesis khác đúng với limitation của paper: temporal multi-hop propagation**.

**Chốt:** mình sẽ chọn **Amazon Reviews’23 Books**, sử dụng **global chronological/absolute-time protocol**, không dùng InstructRec split. MovieLens 32M chỉ nên dùng như dataset phụ/sanity check.

Nếu muốn, bước tiếp theo mình có thể thiết kế luôn **P0 → P3 plan cụ thể cho Amazon Reviews’23 Books**, bao gồm cách tạo temporal subgraph, oracle 2-hop mà không leakage, candidate sampling và metric/gate để tránh lại mất thời gian như hai hướng trước.

[1]: https://amazon-reviews-2023.github.io/main.html "https://amazon-reviews-2023.github.io/main.html"
[2]: https://amazon-reviews-2023.github.io/data_processing/5core.html "https://amazon-reviews-2023.github.io/data_processing/5core.html"
[3]: https://amazon-reviews-2023.github.io/data_processing/index.html "https://amazon-reviews-2023.github.io/data_processing/index.html"
[4]: https://files.grouplens.org/datasets/movielens/ml-32m-README.html "https://files.grouplens.org/datasets/movielens/ml-32m-README.html"
[5]: https://business.yelp.com/data/resources/open-dataset/ "https://business.yelp.com/data/resources/open-dataset/"
[6]: https://doi.org/10.1108/prr-02-2017-0016 "https://doi.org/10.1108/prr-02-2017-0016"
