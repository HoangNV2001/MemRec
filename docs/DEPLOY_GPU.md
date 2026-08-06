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

## 2. Môi trường

```bash
conda create -n memrec python=3.10 -y && conda activate memrec

# 1) torch khớp CUDA của máy TRƯỚC
pip install torch --index-url https://download.pytorch.org/whl/cu124   # đổi cu124 cho khớp

# 2) phần còn lại
pip install -r requirements-gpu.txt

# 3) vllm cài SAU CÙNG, riêng, chỉ cần từ M4
pip install vllm
```

`requirements.txt` cũ pin `torch==2.9.0` bản CPU — **đừng dùng nó trên server**.

### Kiểm tra nhanh

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest tests/rl/ -q          # phải 140 pass (hoặc 139 pass + 1 skip nếu thiếu data/processed)
```

Bộ test này chạy hoàn toàn trên CPU, không gọi API. Chạy nó **trước** khi làm gì khác —
nó bắt mọi lỗi môi trường trong 30 giây thay vì giữa một phiên thuê máy.

---

## 3. Model weights — tải sẵn vào persistent volume (§11.6)

Đặt `HF_HOME` trỏ vào volume bền, **không** phải ephemeral disk, để không phải tải lại mỗi phiên.

```bash
export HF_HOME=/persistent/hf-cache
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct     # ~3.1 GB  reward ranker (M2-B)
huggingface-cli download BAAI/bge-small-en-v1.5         # ~130 MB  grounding (§5.2), chạy CPU
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507    # ~8 GB    policy LM_Mem (M3, M4)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct     # ~1 GB    dry-run tầng T1 (§M4-A)
```

Cả bốn tên đã được verify còn tồn tại trên HF ngày 2026-08-06 (§6.1 yêu cầu).
Fallback nếu OOM: `Qwen/Qwen2.5-3B-Instruct`.

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
