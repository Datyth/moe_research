# Hướng dẫn chạy Phase A: Small-CNN mask reconstruction

Phase A huấn luyện autoencoder tái tạo mask tổn thương ISIC2018: mask `256×256` → encoder CNN → latent 256 chiều → decoder → logits mask `256×256`. Loss là BCE + Dice với trọng số `0.5/0.5`. Model tốt nhất được chọn theo **validation loss thấp nhất**; sau training, runner tự đánh giá `best.pt` trên test và lưu loss, Dice.

Mọi lệnh bên dưới dùng **Bash**, chạy từ thư mục gốc repository. Thay đường dẫn nếu checkout nằm ở máy khác:

```bash
cd /home/teama/projects/project_01/dat/moe_research
```

## 1. Các file và script cần dùng

| File | Vai trò |
| --- | --- |
| [`scripts/run_shape_pretraining.py`](../scripts/run_shape_pretraining.py) | CLI chạy mới hoặc resume Phase A |
| [`configs/phase_a/isic2018_shape_common.yaml`](../configs/phase_a/isic2018_shape_common.yaml) | Dataset và hyperparameter chung; chưa đủ để chạy độc lập |
| [`configs/phase_a/isic2018_s0_small_cnn.yaml`](../configs/phase_a/isic2018_s0_small_cnn.yaml) | Baseline đầy đủ, 100 epoch |
| [`configs/phase_a/isic2018_s0_small_cnn_smoke.yaml`](../configs/phase_a/isic2018_s0_small_cnn_smoke.yaml) | Kiểm tra pipeline trong 1 epoch |
| [`scripts/02_prepare_isic2018_dataset.sh`](../scripts/02_prepare_isic2018_dataset.sh) | Tải, giải nén và chuẩn bị ISIC2018 |
| [`scripts/data/prepare_isic2018.py`](../scripts/data/prepare_isic2018.py) | Chuyển đổi dữ liệu và tạo manifest train/validation/test |

CLI Phase A nhận **một trong hai** tham số: `--config FILE.yaml` hoặc `--resume RUN_DIR`. Các thiết lập như device, batch size, epochs nằm trong YAML; CLI không có `--device`, `--epochs`, `--batch-size` hay `--checkpoint`.

## 2. Chuẩn bị môi trường

Project khai báo Python `>=3.10,<3.11`. Nếu chưa có môi trường:

```bash
conda activate moe-research
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Kiểm tra Python, PyTorch và GPU:

```bash
python --version
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

python scripts/run_shape_pretraining.py --help
```

Config đầy đủ dùng CUDA và AMP `bfloat16`. Nếu GPU không hỗ trợ BF16, đặt `training.amp: false` trong config local để chạy float32. Cấu hình CPU có ở mục 4.

## 3. Chuẩn bị dữ liệu và kiểm tra đường dẫn

### Khi chưa có dữ liệu

```bash
bash scripts/02_prepare_isic2018_dataset.sh
```

Script cần mạng, `unzip`, và `wget` hoặc `curl`. Hiện script đặt đường dẫn cố định:

```text
Raw data:  /home/teama/projects/project_01/dataset/raw/isic2018
Data root: /home/teama/projects/project_01/dataset/isic2018_task1
Manifest:  manifests/isic2018_task1_v1.json
```

Nếu chạy trên máy khác, sửa `RAW_DIR` và `OUT_DIR` trong script trước khi chạy. Dataset được chuyển thành ảnh và sparse mask `.npz`; hãy dùng đường dẫn data root đã chuyển đổi trong config, không trỏ thẳng vào thư mục ZIP/raw mask.

### Khi dữ liệu đã được chuẩn bị

Dùng manifest đã có trong repository. Chỉ khi cần tạo lại manifest từ dữ liệu đã chuyển đổi, chạy:

```bash
python scripts/data/prepare_isic2018.py \
  --data-root /home/teama/projects/project_01/dataset/isic2018_task1 \
  --manifest-output manifests/isic2018_task1_v1.json \
  --val-ratio 0.2 \
  --seed 42
```

Lệnh này ghi lại manifest. Giữ nguyên split/version khi so sánh các thực nghiệm; seed training trong YAML không thay đổi split dữ liệu.

**Chú ý đường dẫn:** config Phase A mặc định có `dataset.root: dataset/isic2018_task1`, được resolve từ gốc repository. Đường dẫn này khác data root của script chuẩn bị dữ liệu ở trên. Mục 4 tạo config local để trỏ đúng nơi lưu dữ liệu mà không cần sửa config baseline.

Phase A chỉ đưa mask vào model, nhưng DataLoader hiện vẫn đọc cả ảnh RGB và mask. Vì vậy cần đủ các file `image` và `label` mà manifest tham chiếu.

Kiểm tra toàn bộ đường dẫn trong manifest; sửa `PHASE_A_DATA_ROOT` nếu cần:

```bash
export PHASE_A_DATA_ROOT=/home/teama/projects/project_01/dataset/isic2018_task1
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PHASE_A_DATA_ROOT"])
manifest = json.loads(Path("manifests/isic2018_task1_v1.json").read_text())
missing = []
for split in ("training", "validation", "test"):
    records = manifest[split]
    print(f"{split}: {len(records)} samples")
    if not records:
        raise SystemExit(f"Split rỗng: {split}")
    for record in records:
        for key in ("image", "label"):
            path = root / record[key]
            if not path.is_file():
                missing.append(str(path))
if missing:
    print("\n".join(missing[:10]))
    raise SystemExit(f"Thiếu {len(missing)} đường dẫn ảnh/mask")
print("OK: đủ đường dẫn ảnh và mask cho cả ba split")
PY
```

## 4. Chạy thử 1 epoch bằng terminal

### GPU: config local với đường dẫn dữ liệu của máy hiện tại

Tạo file sau một lần; chỉnh `dataset.root` theo nơi dữ liệu thực sự nằm:

```bash
cat > configs/phase_a/isic2018_s0_small_cnn_smoke_local.yaml <<'YAML'
extends: isic2018_s0_small_cnn_smoke.yaml

dataset:
  root: /home/teama/projects/project_01/dataset/isic2018_task1
YAML

python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_smoke_local.yaml
```

Smoke dùng 1 epoch, batch size 4, workers 0, AMP tắt, scheduler `none`. **Device vẫn là `cuda`** do kế thừa config chung. Smoke chạy toàn bộ split train, validation và test; không giới hạn số sample.

Nếu dữ liệu đã nằm đúng đường dẫn mặc định của baseline, có thể chạy trực tiếp:

```bash
python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_smoke.yaml
```

Khi thành công, terminal in `Run directory`, tiến trình epoch, rồi `Test Loss` và `Test Dice`. Kết quả nằm trong `runs/phase_a_s0_small_cnn_smoke/<run-id>/`.

### CPU: khi không có CUDA

```bash
cat > configs/phase_a/isic2018_s0_small_cnn_smoke_cpu.yaml <<'YAML'
extends: isic2018_s0_small_cnn_smoke_local.yaml

experiment:
  name: phase_a_s0_small_cnn_smoke_cpu

training:
  device: cpu
  amp: false
  batch_size: 2
  num_workers: 0
YAML

python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_smoke_cpu.yaml
```

Ví dụ CPU kế thừa file `smoke_local.yaml` vừa tạo. Chạy toàn bộ dữ liệu trên CPU có thể mất nhiều thời gian; mục 9 có test dùng dữ liệu giả để kiểm tra pipeline.

## 5. Huấn luyện đầy đủ trên GPU

Tạo config local cho run 100 epoch:

```bash
cat > configs/phase_a/isic2018_s0_small_cnn_local.yaml <<'YAML'
extends: isic2018_s0_small_cnn.yaml

dataset:
  root: /home/teama/projects/project_01/dataset/isic2018_task1
YAML

python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_local.yaml
```

Nếu dữ liệu đã đúng đường dẫn mặc định, dùng trực tiếp `--config configs/phase_a/isic2018_s0_small_cnn.yaml`.

| Thiết lập baseline | Giá trị |
| --- | --- |
| Epochs / batch size / workers | `100` / `32` / `8` |
| Device / AMP | `cuda` / bật, `bfloat16` |
| Optimizer | AdamW, learning rate `0.0003`, weight decay `0.0001` |
| Scheduler | Cosine, `eta_min: 0.0` |
| Loss | BCE + Dice, trọng số `0.5/0.5` |
| Gradient clipping | Norm `1.0` |
| Chọn best checkpoint | `monitor: loss`, `monitor_mode: min` |
| Threshold tính Dice | `0.5` |

Để chọn một GPU cụ thể, đặt biến môi trường trước lệnh, ví dụ GPU số 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_local.yaml
```

Để đổi hyperparameter, thêm block tương ứng vào file local trước khi chạy, ví dụ:

```yaml
training:
  batch_size: 8
  num_workers: 4
  amp: false
```

`extends` được resolve theo thư mục chứa file YAML và merge đệ quy. Các trường không khai lại giữ giá trị từ file cha. `dataset.root`, `dataset.manifest` và `experiment.output_root` tương đối được resolve từ gốc repository.

Giữ `dataset.image_size: [256, 256]` và các tham số kiến trúc baseline: implementation hiện cố định latent 256 chiều và kích thước reconstruction 256×256. Khi thiếu VRAM, giảm batch size hoặc workers thay vì giảm kích thước ảnh. `dataset.in_channels: 3` mô tả loader RGB; model shape vẫn nhận mask một kênh.

## 6. Lưu log và chạy bằng shell script

### Vừa xem terminal vừa ghi log

Chạy trong Bash đã kích hoạt môi trường Python:

```bash
mkdir -p logs
PHASE_A_LOG="logs/phase_a_$(date -u +%Y%m%dT%H%M%SZ).log"
set -o pipefail
python -u scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_local.yaml \
  2>&1 | tee "$PHASE_A_LOG"
```

`-u` giúp log Python xuất ngay. `pipefail` giữ mã lỗi của training khi dùng `tee`. Đường dẫn run chính xác xuất hiện ở dòng `Run directory` trong log.

### Tạo script tiện dụng để chạy lại

Đây là script tùy chọn do bạn tạo từ terminal. Nó chuyển về đúng repository root, dùng Python của môi trường đang kích hoạt, rồi chuyển nguyên tham số sang CLI Phase A:

```bash
cat > scripts/run_phase_a_local.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
PHASE_A_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PHASE_A_PROJECT_ROOT"
exec python -u scripts/run_shape_pretraining.py "$@"
BASH

bash scripts/run_phase_a_local.sh \
  --config configs/phase_a/isic2018_s0_small_cnn_local.yaml
```

Chạy nền và lưu log, sau khi đã tạo script trên:

```bash
mkdir -p logs
PHASE_A_LOG="logs/phase_a_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash scripts/run_phase_a_local.sh \
  --config configs/phase_a/isic2018_s0_small_cnn_local.yaml \
  > "$PHASE_A_LOG" 2>&1 &
PHASE_A_PID=$!
echo "PID: $PHASE_A_PID"
echo "Log: $PHASE_A_LOG"
tail -f "$PHASE_A_LOG"
```

`Ctrl+C` ở `tail -f` chỉ dừng xem log; tiến trình nền tiếp tục chạy. Mỗi lần gọi `--config` tạo một run mới, nên chọn một cách khởi chạy cho mỗi thực nghiệm.

## 7. Resume khi training bị gián đoạn

Lấy đường dẫn chính xác từ dòng `Run directory` của lần chạy trước. Ví dụ dưới đây dùng run-id minh họa; thay bằng thư mục thực tế:

```bash
PHASE_A_RUN_DIR="runs/phase_a_s0_small_cnn/20260905T120000Z_seed-42"
python scripts/run_shape_pretraining.py --resume "$PHASE_A_RUN_DIR"
```

Hoặc dùng shell script đã tạo:

```bash
bash scripts/run_phase_a_local.sh --resume "$PHASE_A_RUN_DIR"
```

Runner đọc `config.yaml` trong run, khôi phục `last.pt` cùng `history.json`, model, optimizer, scheduler, AMP scaler và giá trị monitor tốt nhất. Nó tiếp tục từ epoch hoàn tất gần nhất + 1; phần epoch đang chạy lúc bị ngắt phải chạy lại.

- Truyền **thư mục run**, không truyền `last.pt` hoặc `best.pt`.
- Run phải có `config.yaml`, `metadata.json`, `last.pt` và `history.json` khớp nhau; giữ cả `best.pt` để dùng cho final test.
- `training.epochs` là tổng epoch mục tiêu. Nếu đã hoàn tất đủ epoch, resume bỏ qua training và đánh giá lại `best.pt` trên test.
- Thay đổi file config gốc trong `configs/` không ảnh hưởng config đã lưu của run đang resume.
- Giữ monitor `loss/min` và cấu hình model, dataset, loss, optimizer, scheduler của run. Muốn đổi thiết lập thực nghiệm, tạo config và run mới.
- Smoke 1 epoch đã hoàn tất không tự biến thành run 100 epoch khi resume. Dùng config đầy đủ để bắt đầu run huấn luyện chính.
- Nếu bị lỗi trước khi có checkpoint epoch đầu tiên, cần chạy mới bằng `--config`.

## 8. Đọc kết quả

Một run hoàn tất có cấu trúc:

```text
runs/phase_a_s0_small_cnn/<run-id>/
├── config.yaml
├── metadata.json
├── history.json
├── best.pt
├── last.pt
└── test_metrics.json
```

| File | Nội dung |
| --- | --- |
| `config.yaml` | Cấu hình đã merge và resolve đường dẫn, dùng lại khi resume |
| `metadata.json` | Trạng thái, seed, Git metadata, manifest hash, thời gian, số lần resume, best epoch và monitor |
| `history.json` | `epoch`, `train_loss`, `val_loss`, `val_dice` theo từng epoch |
| `best.pt` | Checkpoint có validation loss thấp nhất |
| `last.pt` | Trạng thái checkpoint epoch gần nhất để resume |
| `test_metrics.json` | `checkpoint: best.pt`, `split: test`, `loss`, `dice` |

Sau khi đặt `PHASE_A_RUN_DIR` như mục 7, xem kết quả bằng terminal:

```bash
python -m json.tool "$PHASE_A_RUN_DIR/test_metrics.json"
python -m json.tool "$PHASE_A_RUN_DIR/metadata.json"
python -m json.tool "$PHASE_A_RUN_DIR/history.json"
```

Run thành công có `metadata.status: completed`, `monitor_name: loss`, `monitor_mode: min`, `best_epoch` và `best_monitor_value`. Giá trị `best_monitor_value` là validation loss; Dice càng cao càng tốt nhưng không quyết định chọn `best.pt` trong baseline này.

Runner tự tính final test. Hiện Phase A chỉ lưu loss và Dice, chưa có ảnh reconstruction hoặc thống kê latent. Script `scripts/evaluation/evaluate.py` dành cho segmentation; `scripts/summarize_experiments.py` yêu cầu đủ sáu metric segmentation, nên hai script đó chưa hỗ trợ đầu ra Phase A hiện tại. Dùng các JSON trong run để đọc kết quả.

## 9. Kiểm thử trước khi chạy dài

Chạy riêng các test Phase A và task contract, không cần dữ liệu ISIC2018 thật:

```bash
python -m unittest discover -s tests -p 'test_shape_models.py' -v
python -m unittest discover -s tests -p 'test_tasks.py' -v
python -m unittest discover -s tests -p 'test_shape_pretraining.py' -v
```

Chạy toàn bộ test suite, bao gồm trainer, evaluator và resume:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Các lệnh trên là hướng dẫn kiểm tra, không phải báo cáo test đã pass. Trước run dài, nên hoàn tất unit tests và smoke run với dữ liệu thật ở mục 4.

## 10. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| `python: command not found` hoặc thiếu module | Kích hoạt môi trường Python 3.10 và cài `requirements.txt` bằng chính Python đó |
| `CUDA was requested...` | Kiểm tra `torch.cuda.is_available()`; nếu chạy CPU, dùng config CPU ở mục 4 |
| GPU không hỗ trợ BF16 | Đặt `training.amp: false` trong config local trước run mới |
| `Tracked dataset manifest not found` | Kiểm tra `dataset.manifest` và file manifest đã chuẩn bị |
| `Image not found` / `Mask not found` | Kiểm tra `dataset.root`, relative path trong manifest và đích symlink ảnh; chạy kiểm tra ở mục 3 |
| CUDA out of memory | Giảm `training.batch_size`; giữ image size 256×256 |
| Lỗi worker/DataLoader hoặc shared memory | Thử `training.num_workers: 0` |
| Lỗi fixed model config hoặc spatial size | Giữ các trường kiến trúc của `isic2018_s0_small_cnn.yaml` và image size `[256, 256]` |
| `Checkpoint monitor configuration ... does not match` | Resume đúng run Phase A với `loss/min`; checkpoint segmentation `dice/max` không phù hợp |
| History và checkpoint không khớp | Khôi phục đúng bộ `history.json` và `last.pt` từ cùng run/epoch |
| Resume không chạy thêm epoch | Xem `training.epochs` trong config của run và epoch cuối của history; run có thể đã hoàn tất |

Nếu run được đánh dấu `failed`, đọc trường `error` trong `metadata.json` và traceback trong terminal/log để xác định nguyên nhân.
