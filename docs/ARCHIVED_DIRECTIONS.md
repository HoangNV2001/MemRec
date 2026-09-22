# Archived research directions

File này chỉ giữ decision trail ngắn cho các hướng đã đóng. Implementation,
config, notebook và artifact dư của chúng đã được loại khỏi current surface;
lịch sử đầy đủ vẫn khôi phục được từ Git trước cleanup.

| Hướng | Kết quả chính | Quyết định |
|---|---|---|
| SFT/GRPO cho LM_Mem | Oracle synthesis headroom chỉ khoảng +0,06–0,08 NDCG@5; expected practical gain +0,02–0,035 | Dừng trước full GRPO; reward/ranking signal không đủ justify cost |
| InstructRec selective 2-hop | Oracle ΔNDCG@5 −0,0094, CI [−0,0362,+0,0176] | Hard stop read-side 2-hop |
| Candidate-conditioned evidence | Candidate vs baseline −0,0080, CI cắt 0 | Hard stop evidence reranking |
| Buffered propagation | Không có global cross-user clock; source-only gold coverage support>=2 chỉ 6,04% | Stop trước LLM/write |
| Amazon P2 item-overlay oracle | +0,0324, dưới gate +0,05 | Không admit selector |
| Raw 3-hop packet overlay | +0,0155 vs local; CI cắt 0 | Không tăng n-hop cùng representation |
| Filtered 3-hop | +0,0186 vs local; filter không tạo significant gain | Stop rule/filter tuning |
| Candidate graph 3/5-layer | 5-layer +0,0077 vs local và −0,0019 vs 3-layer | Deeper expansion tăng noise nhanh hơn signal |
| Learned adaptive multi-hop P6 | Calibration +0,0265 nhưng fresh test −0,0225, CI [−0,0559,+0,0113] | Hard stop learned depth/margin gate |
| P7-v1 output contract | Duplicate label làm identical retry fail | Execution invalid, sealed; không metric |

Kết luận xuyên suốt: thêm hop trên co-preference graph không đủ. Hướng hiện tại
khả quan vì đổi **edge semantics** sang temporal next-item transition, không vì
thêm rule hoặc target-aware selection. Chi tiết current method ở
[TRANSITION_PPR_METHOD.md](TRANSITION_PPR_METHOD.md).
