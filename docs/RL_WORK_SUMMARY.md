# RL_WORK_SUMMARY.md — Tổng kết hướng SFT/RL cho Stage-R

> **Trạng thái:** hoàn tất điều tra, không tiếp tục huấn luyện SFT/GRPO.
>
> Tài liệu này lưu phần bằng chứng và artifact còn giá trị. Hướng active là
> [MULTIHOP_PLAN.md](MULTIHOP_PLAN.md).

## Kết luận quyết định

Mục tiêu ban đầu là warm-start một LM_Mem nhỏ bằng SFT rồi dùng GRPO để tối ưu
M_collab theo chất lượng ranking của LM_Rec đóng băng. Hướng này được dừng
**trước full GRPO**, không phải vì collaborative memory vô ích mà vì phần có thể
cải thiện chỉ bằng cách tổng hợp tốt hơn trên **cùng một 1-hop neighborhood** quá
nhỏ so với mục tiêu accuracy.

- Oracle của Stage-R synthesis: khoảng **+0.06 đến +0.08 NDCG@5**.
- Với độ chính xác reward within-user đo được, gain thực tế ước **+0.02 đến
  +0.035 NDCG@5**.
- Mục tiêu cần một gain rõ ràng so với MemRec gốc (mốc đặt ra là **>= +0.05
  NDCG@5**). Vì vậy đầu tư thêm cho GRPO là rủi ro cao, không có tỷ lệ
  chi phí/lợi ích hợp lý.

Điều này chỉ đóng hướng **tối ưu synthesis trên cùng neighborhood**. Nó không
bác bỏ giả thuyết collaborative memory, cũng không đo trần của multi-hop.

## Công việc đã hoàn thành

### 1. Baseline và môi trường đánh giá

- Đã reproduce các baseline trên InstructRec-Books và sửa lỗi prompt/eval làm
  sai ablation vanilla và no-collaborative-read.
- Đã phát hiện đường eval gốc sample negative theo thread nên không hoàn toàn
  reproducible. Không dùng bảng M0 cũ để so sánh trực tiếp với thí nghiệm mới;
  multi-hop phải đo lại trên candidate set cố định.
- Đã dựng snapshot memory graph sạch, cố định, candidate-blind, với split user
  disjoint: **train 1,185 / validation 149 / test 993**. Có 23 user bị loại vì
  title/id gold lộ qua neighbor table; prompt sau lọc không chứa gold, candidate
  hay instruction.
- Candidate set có 10 item/user, tất định theo (seed, user_id, salt); warm-up và
  eval dùng salt khác nhau để tránh negative eval ảnh hưởng memory warm-up.

### 2. Kiểm chứng collaborative memory và reward

- M_collab thật cải thiện LM_Rec thật so với empty memory: **+0.1112 NDCG@5**,
  95% CI **[+0.0640, +0.1585]**, n=149.
- Kết quả không biến mất với ranker mạnh hơn: headroom thô đo được là +0.1724
  với gpt-5.6-luna và +0.1371 với Qwen3.5-4B.
- Đã thử nhiều proxy/ranker và cách chấm: Qwen 1.5B, 3B, pointwise scoring,
  frontier API, Qwen3.5-4B và candidate reward rộng hơn. Từ đó tách được lỗi
  do proxy khỏi giới hạn thật của action space Stage-R.
- Proxy Qwen3.5-4B cuối cùng đạt validation aggregate (Spearman **0.7726**) và
  tín hiệu within-user (**62.9%**, CI [57.1%, 68.4%]) khi dùng NDCG@5 cộng
  0.02 nhân margin_logit.

### 3. Kiểm tra tính khả thi của GRPO

- Đã đo đúng tín hiệu mà một group GRPO nhìn thấy: cùng user, nhiều M_collab
  khác nhau. Đây quan trọng hơn tương quan gộp giữa các user.
- Reward chỉ phụ thuộc gold rank có rất nhiều tie; bổ sung đại lượng liên tục có
  thể tạo cảm giác hết tie nhưng dễ chỉ bơm nhiễu. Mọi kết luận về tie-breaker
  đều được đối chiếu với một pass nhiễu của **cùng memory**.
- Đã giảm group suy biến từ 71.1% xuống 0% với margin_logit, nhưng độ chính xác
  reward trên so sánh synthesis tinh vẫn không đủ để biến headroom +0.06–0.08
  thành gain an toàn >= +0.05.
- Đã tạo teacher data và chạy SFT LoRA thử nghiệm (116 step, train loss 0.3902),
  nhưng **không dùng checkpoint này làm kết quả** và không chạy full GRPO.

## Những bài học cần mang sang multi-hop

1. **Đo oracle/headroom trước khi xây cơ chế hoặc thuê GPU.** Với action space
   có trần thấp, reward tốt không làm xuất hiện gain không tồn tại.
2. **Dùng so sánh paired theo user và candidate set cố định.** Không trộn bảng
   M0 cũ (RNG theo thread) với protocol mới.
3. **Tách candidate construction khỏi đánh giá.** Graph/context/selector không
   được thấy instruction, candidate hay gold; gold chỉ tồn tại trong phép đo
   offline và oracle được gắn nhãn rõ là upper bound.
4. **Oracle phải chống winner's curse.** Nếu chọn context tốt nhất từ nhiều
   biến thể bằng một lần gọi LLM, cần chấm lại độc lập context được chọn trước
   khi báo cáo headroom.
5. **Cùng budget thật sự:** giữ đồng thời số node, token context, model,
   candidate list, thứ tự candidate, generation settings và số lần gọi ranker.

## Artifact còn dùng được

| Artifact | Vai trò trong hướng mới |
|---|---|
| data/rl/graph_snapshot_books.json | Frozen user/item memory và 1-hop state đã materialize; không phải full adjacency graph. |
| data/processed/instructrec-books/ | Interaction history để dựng topology 2-hop tất định trong MH0, không cần warm-up lại API. |
| data/rl/stager_books_{train,val,test}.jsonl | Split, candidate cố định, user memory, candidate memory và metadata chống leakage. |
| data/rl/m2_val_reference_books.json | Reference để kiểm tra lại ranker và chi phí API. |
| src/rl/splits.py, src/rl/leakage.py, src/rl/env.py | Sampling tất định, guard leakage và đọc snapshot. |
| src/memory/graph.py, pruner.py, packer.py | Định nghĩa 1-hop hiện tại; multi-hop phải lấy đây làm control. |
| tests/rl/ | Regression tests cho split, leakage, parsing và reward utility. |

Các module SFT/GRPO và reward được **giữ nguyên trong code** như nghiên cứu đã
hoàn thành, nhưng không thuộc critical path của multi-hop. Không xoá chúng vì
snapshot, kiểm tra leakage và harness có thể vẫn tái sử dụng; chỉ bỏ các kế hoạch
triển khai RL/LM_Rec không còn active.

## Những gì không tiếp tục

- Full GRPO cho LM_Mem/Stage-R trên 1-hop cố định.
- SFT như contribution độc lập hoặc dùng checkpoint SFT chưa eval làm baseline.
- Extension SFT/GRPO cho LM_Rec và mọi joint training.
- Tìm thêm reward proxy chỉ để tối ưu cùng action space synthesis.

## Hướng kế tiếp

Kiểm tra liệu mở rộng thông tin collaborative theo **multi-hop có chọn lọc** có
tạo headroom thực tế khi giữ nguyên budget. Chỉ sau khi oracle validation cho
thấy headroom đủ lớn mới xây selector/routing; nếu không, dừng sớm.
