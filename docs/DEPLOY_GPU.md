# DEPLOY_GPU.md — chuyển project lên server GPU

> Áp dụng từ **M2 Phần B** trở đi. M0/M1/M2-A đã xong trên CPU và **không cần chạy lại**.
> Nguyên tắc §11.6: máy tính tiền chỉ làm đúng phần cần GPU.

---

## 0. TL;DR

Cần chuyển **~28 MB** dữ liệu, không phải 1.1 GB.

Bốn file jsonl/json dưới đây là tất cả những gì đường GPU đọc. Đã kiểm chứng bằng cách
dựng một cây thư mục chỉ có code + 4 file này rồi chạy `src.rl.validate_reward`: kết quả
**trùng từng chữ số** với cây đầy đủ, và `pytest tests/rl/` cho 139 pass / 1 skip
(cái skip là test duy nhất thật sự cần `.inter`).

---

## 1. Chuyển gì lên server

### 1.1 Code — qua git

```bash
# trên server
git clone <remote> MemRec && cd MemRec
git checkout rl/m2-reward
```

Nếu chưa có remote, đóng gói thẳng từ máy hiện tại:

```bash
# trên máy local
git bundle create memrec.bundle --all
# scp memrec.bundle server:~/  → trên server:  git clone memrec.bundle MemRec
```

### 1.2 Dữ liệu — 28 MB, **bắt buộc**

| File | Kích thước | Vì sao cần |
|---|---:|---|
| `data/rl/stager_books_train.jsonl` | 14 MB | M3 SFT, M4 GRPO |
| `data/rl/stager_books_test.jsonl` | 12 MB | eval cuối |
| `data/rl/stager_books_val.jsonl` | 1.7 MB | M2-B, eval trong lúc train |
| `data/rl/m2_val_reference_books.json` | 440 KB | **nửa gpt-4o-mini của Validation A/B đã cache** — thiếu file này là phải tiêu lại $0.5 và 6 phút API |

`data/rl/user_splits_books.json` đã nằm trong git, không cần chép.

**Hai tarball đều lưu đường dẫn tương đối so với gốc repo, nên chúng tự vào đúng chỗ —
không phải move gì cả:**

```bash
cd /path/to/MemRec
tar xzf memrec_rl_data.tgz          # -> data/rl/*.jsonl, m2_val_reference_books.json
tar xzf memrec_rl_extras.tgz        # -> data/rl/graph_snapshot..., results/m1_warmup_2350/...
python -m src.rl.verify_transfer    # BẮT BUỘC: xác nhận đã sang đủ và không hỏng
```

Bên trong mỗi tarball có sẵn `data/rl/TRANSFER_MANIFEST.txt` mô tả nội dung, để mô tả
không bao giờ bị tách khỏi dữ liệu.

`verify_transfer` kiểm tra **nội dung**, không chỉ sự tồn tại — một lần `scp` đứt để lại
file vẫn mở được nhưng sai. Nó check: sha256, số dòng chính xác (1185/149/993), đủ mọi
cột mà reward đọc, gold nằm trong candidate, split vẫn disjoint, và reference cache vẫn
cho thấy memory thật thắng memory rỗng. Artifact tuỳ chọn thiếu thì chỉ cảnh báo.

> Đã kiểm chứng: giải nén hai tarball vào một checkout sạch (chỉ có code, không có
> `data/processed`) → `verify_transfer` all pass, `pytest tests/rl/` 139 pass + 1 skip,
> và `src.rl.validate_reward` chạy hết 745 cặp.

### 1.3 `.env` — bắt buộc, **không** nằm trong git

```bash
scp .env server:~/MemRec/.env
```

Cần cho eval bằng gpt-4o-mini ở M3/M7a. M2-B thì không gọi API.

### 1.4 Nên chép để phòng hờ (không bắt buộc để chạy)

| File | Kích thước | Vì sao |
|---|---:|---|
| `data/rl/graph_snapshot_books.json` | 17 MB | Dựng lại jsonl nếu prompt template đổi, khỏi cần `data/processed` |
| `results/m1_warmup_2350/memory_warmup_only.json` | 52 MB | Artifact **đắt** duy nhất của M1 (~$3.15 API). Mất là phải trả lại tiền. §11.4② yêu cầu để trên storage bền — nên có bản backup ở đâu đó, không nhất thiết trên GPU box |

### 1.5 **KHÔNG** cần chuyển

| Thư mục | Kích thước | Lý do |
|---|---:|---|
| `data/processed/` | **1.1 GB** | `.meta` 567 MB + `.pkl` 328 MB + `.text` 132 MB. Đường GPU không đọc file nào trong đây — title và item memory của candidate đã được nhúng sẵn vào jsonl đúng để tránh việc này |
| `data/iagent/` | 895 MB | Chỉ dùng lúc convert dataset ban đầu |
| `results/m0_*`, `results/m1_*` | ~470 MB | Log và conversation dump của M0/M1. Chỉ cần khi viết luận văn, giữ ở máy local |

> Chỉ chuyển `data/processed/` lên nếu muốn **dựng lại dataset** trên server
> (`src.rl.build_dataset` cần `RecDataset`). Bình thường không cần: jsonl đã dựng xong.

---

## 1.6 Chọn máy trên vast.ai — **đừng thuê H100 ngay**

Việc đầu tiên (M2 Phần B) là **745 forward pass prefill-only của model 1.5B**. Chạy vài phút
trên bất kỳ GPU nào ≥ 8 GB. Thuê H100 $3/h cho việc đó là đốt tiền — đúng loại lỗi §2.5.1 cảnh báo.

| Giai đoạn | Máy nên thuê | Vì sao |
|---|---|---|
| **Bây giờ** — M2-B, dựng môi trường, M3-B chấm điểm 9600 mẫu, M4-A dry-run 0.5B | **RTX 4090 / A10 / L4, 24 GB, ~$0.25–0.50/h** | Toàn bộ đều là inference model nhỏ hoặc câu hỏi "code có chạy không". Vài giờ ở đây = vài chục nghìn đồng |
| **Sau đó** — M3 SFT thật, M4 GRPO thật | **H100 80 GB, ~$2.5–3.5/h** | §4.3: policy 4B + vLLM + ranker colocate ≈ 66 GB |

- **Disk:** chọn ≥ **80 GB**. Model ~12 GB, data ~100 MB, còn lại cho checkpoint và HF cache.
- **M3 Phần A (sinh teacher bằng gpt-4o-mini, ~$10–14) là CPU + API** — chạy trên máy local
  hoặc trên chính con GPU rẻ, nhưng **đừng** để H100 chạy nó.
- Trên vast.ai, **destroy instance là mất sạch**. Đẩy artifact ra ngoài trước khi destroy (§5).

---

## 2. Môi trường

Với image `vastai/base-image` (Ubuntu + CUDA). Image này **đã có sẵn** Python và driver,
nhưng **chưa chắc** có torch — kiểm tra trước thay vì đoán:

```bash
nvidia-smi                       # driver + CUDA version của host
python3 -V
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" \
    || echo "chưa có torch"
```

Image của vast.ai thường kích hoạt sẵn một venv. Nếu có thì dùng luôn; nếu không, tạo mới:

```bash
python3 -m venv /workspace/venv && source /workspace/venv/bin/activate
```

Rồi:

```bash
cd /workspace/MemRec

# 1) torch khớp CUDA của host TRƯỚC — bỏ qua nếu image đã có sẵn torch chạy được GPU
pip install torch --index-url https://download.pytorch.org/whl/cu124   # đổi cu124 cho khớp nvidia-smi

# 2) phần còn lại
pip install -r requirements-gpu.txt

# 3) vllm cài SAU CÙNG, riêng, và CHỈ khi tới M4
pip install vllm
```

`requirements.txt` cũ pin `torch==2.9.0` bản **CPU** — cài nó trên server sẽ ra môi trường
chạy được test nhưng chết ở forward pass đầu tiên. **Dùng `requirements-gpu.txt`.**

> Nếu image đã có torch bản GPU, đừng cài đè: chạy thẳng bước 2. `requirements-gpu.txt`
> cố tình **không** pin torch chính vì lý do này.

### Kiểm tra nhanh — chạy đủ 3 lệnh này TRƯỚC khi làm gì khác

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m src.rl.verify_transfer     # data sang đủ và không hỏng chưa
pytest tests/rl/ -q                  # 140 pass (hoặc vài skip nếu thiếu data/processed)
```

Cả ba chạy trên CPU, không gọi API, mất chưa tới 1 phút — nhưng bắt được gần như mọi lỗi
môi trường và lỗi truyền file, thay vì phát hiện giữa một phiên đang tính tiền.

---

## 3. Model weights — tải sẵn vào persistent volume (§11.6)

Trên vast.ai đặt `HF_HOME` vào thư mục volume đã thuê (thường `/workspace`), đừng để mặc
định `~/.cache`. Lưu ý: **destroy instance vẫn mất**, đây chỉ là tránh tải lại giữa các lần reboot.

```bash
export HF_HOME=/workspace/hf-cache
echo 'export HF_HOME=/workspace/hf-cache' >> ~/.bashrc

# chỉ cần cho M2-B — đừng tải hết 12 GB ngay
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct     # ~3.1 GB  reward ranker (M2-B)
huggingface-cli download BAAI/bge-small-en-v1.5         # ~130 MB  grounding (§5.2), chạy CPU

# tới M3/M4 mới cần
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507    # ~8 GB    policy LM_Mem
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct     # ~1 GB    dry-run tầng T1 (§M4-A)
```

Cả bốn tên đã được verify còn tồn tại trên HF ngày 2026-08-06 (§6.1 yêu cầu).
Fallback nếu OOM ở M4: `Qwen/Qwen2.5-3B-Instruct`.

---

## 4. Việc đầu tiên trên GPU: M2 Phần B

```bash
bash scripts/rl/02_validate_reward.sh hf
```

Lệnh này **không gọi API** — nó chấm lại 745 cặp `(user, arm)` đã cache bằng ranker 1.5B
rồi tính tương quan. Chạy cả hai chế độ `include_instruction` và ghi ra
`data/rl/m2_validation_report{,_no_instruction}.json`.

DoD (§7 M2): Spearman ρ ≥ 0.6 · `r(thật) ≥ max(arm hỏng) + 0.02` · ≥ 20 reward/s @ batch 64.

Trước khi bật máy, đọc lại `docs/RESULTS.md` mục "M2 Reward Validation" — Phần A đã đo
sẵn phía gpt-4o-mini và đã phát hiện vấn đề tỉ lệ trùng reward 74%, có ảnh hưởng tới
quyết định `soft_weight` ở Phần B.

> **Ước lượng:** M2-B chỉ là 745 forward pass prefill-only của một model 1.5B — vài phút.
> Đừng thuê H100 riêng cho nó; gộp cùng phiên với M3-B đúng như §11.5.

---

## 5. Sau mỗi phiên — đẩy về storage bền (§11.6)

```bash
tar czf m2_artifacts.tgz data/rl/m2_validation_report*.json
# checkpoint SFT/GRPO thì đẩy checkpoints/rl/ lên object storage TRƯỚC KHI terminate
```

Artifact không được để trên ephemeral disk: mất checkpoint SFT nghĩa là mất ~4 GPU-hour.
