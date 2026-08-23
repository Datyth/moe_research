# Hướng dẫn chạy segmentation end-to-end

Tài liệu này hướng dẫn chạy pipeline UNet trên ISIC 2018 theo luồng:

```text
Chuẩn bị dữ liệu
    → train + validation mỗi epoch
    → chọn best checkpoint
    → test
    → xem metrics và visualization
```

Mọi lệnh bên dưới được chạy từ thư mục gốc của repository.

## 1. Chuẩn bị môi trường

Project sử dụng Python 3.10. Kích hoạt môi trường hiện có:

```bash
source .venv/bin/activate
```

Nếu chưa có môi trường, có thể tạo mới:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra PyTorch và GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Khi dùng GPU, kết quả cần có:

```text
True
NVIDIA ...
```

## 2. Chuẩn bị ISIC 2018

### Trường hợp chưa có dữ liệu

Chạy script tải, giải nén, chuyển mask và tạo dataset manifest:

```bash
bash scripts/02_prepare_isic2018_dataset.sh
```

Dữ liệu được đặt tại:

```text
dataset/isic2018_task1/
├── dataset.json
├── images/
├── labels/
└── pseudo/
```

Ảnh có thể được lưu bằng symbolic link để tránh tạo thêm bản sao lớn.

### Trường hợp đã có dataset

Tạo lại deterministic train/validation split mà không copy hoặc di chuyển file:

```bash
python scripts/data/prepare_isic2018.py \
  --data-root dataset/isic2018_task1 \
  --val-ratio 0.2 \
  --seed 42
```

Với dataset ISIC hiện tại, kết quả mong đợi là:

```text
Training samples: 2075
Validation samples: 519
Test samples: 100
```

Validation được tách từ official training set. Test set chỉ được sử dụng cho đánh giá cuối cùng, không dùng để chọn model hoặc chỉnh hyperparameter.

## 3. Chạy tests

Nên chạy tests trước khi bắt đầu training:

```bash
python -m unittest discover -s src/tests -p 'test_*.py' -v
```

Kết quả hiện tại:

```text
Ran 18 tests
OK
```

## 4. Smoke training một epoch

Chạy một epoch trước để kiểm tra toàn bộ train/validation pipeline:

```bash
python scripts/training/train_unet.py \
  --epochs 1 \
  --image-size 256 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --checkpoint-dir checkpoints/smoke \
  --checkpoint-prefix unet \
  --history-path results/smoke/unet_training_history.json
```

Sau mỗi epoch, terminal hiển thị:

```text
Train Loss
Val Loss
Val Dice
Val IoU
```

Output smoke run:

```text
checkpoints/smoke/
├── unet_best.pt
└── unet_last.pt

results/smoke/
└── unet_training_history.json
```

`unet_best.pt` là model có validation Dice cao nhất. `unet_last.pt` là trạng thái sau epoch gần nhất.

## 5. Training chính thức

Ví dụ train 50 epochs:

```bash
python scripts/training/train_unet.py \
  --epochs 50 \
  --image-size 256 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda
```

Output mặc định:

```text
checkpoints/
├── unet_best.pt
└── unet_last.pt

results/
└── unet_training_history.json
```

Một số option thường dùng:

| Option | Mặc định | Ý nghĩa |
|---|---:|---|
| `--epochs` | `1` | Số epoch training |
| `--image-size` | `256` | Kích thước ảnh vuông đưa vào model |
| `--batch-size` | `8` | Số ảnh trong một batch |
| `--num-workers` | `4` | Số DataLoader workers |
| `--learning-rate` | `1e-4` | Learning rate của AdamW |
| `--base-channels` | `32` | Độ rộng cơ bản của UNet |
| `--device` | `cuda` | Thiết bị training; dùng `cpu` nếu không có GPU |
| `--no-amp` | tắt | Tắt CUDA automatic mixed precision |

## 6. Đánh giá best checkpoint trên test set

Sau khi training xong, evaluate `unet_best.pt`:

```bash
python scripts/evaluation/evaluate.py \
  --data-root dataset/isic2018_task1 \
  --checkpoint checkpoints/unet_best.pt \
  --split test \
  --output-dir results/unet_test \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --num-visualizations 12
```

Terminal hiển thị:

```text
Test Loss
Test Dice
Test IoU
```

Kết quả được lưu tại:

```text
results/unet_test/
├── metrics.json
└── visualizations/
    ├── ISIC_xxx.png
    └── ...
```

Mỗi visualization gồm bốn panel:

1. Input image.
2. Ground-truth mask.
3. Predicted mask.
4. Hai đường biên được overlay lên input.

Màu trong panel Boundary Overlay:

- Xanh lá: ground-truth boundary.
- Đỏ: prediction boundary.
- Vàng: phần boundary trùng nhau.

Khoảng cách giữa đường xanh và đường đỏ thể hiện trực tiếp sai số biên của prediction.

## 7. Đánh giá validation hoặc chạy bằng CPU

Đánh giá validation split:

```bash
python scripts/evaluation/evaluate.py \
  --checkpoint checkpoints/unet_best.pt \
  --split val \
  --output-dir results/unet_val
```

Nếu không có GPU, thay:

```text
--device cuda
```

bằng:

```text
--device cpu
```

Training và evaluation bằng CPU sẽ chậm hơn đáng kể.

## 8. Lỗi thường gặp

### `CUDA was requested, but torch.cuda.is_available() is False`

PyTorch không nhận GPU. Kiểm tra lại CUDA/PyTorch hoặc tạm chạy với:

```bash
--device cpu --no-amp
```

### `Dataset manifest not found`

Kiểm tra file sau có tồn tại:

```text
dataset/isic2018_task1/dataset.json
```

Nếu chưa có, quay lại bước chuẩn bị dataset.

### `Split 'validation' is missing`

Manifest chưa được tách validation. Chạy lại:

```bash
python scripts/data/prepare_isic2018.py --val-ratio 0.2 --seed 42
```

### `Checkpoint not found`

Kiểm tra đường dẫn truyền vào `--checkpoint`. Training thành công phải tạo `unet_best.pt` sau epoch validation đầu tiên.

### Hết GPU memory

Giảm batch size, ví dụ:

```bash
--batch-size 8
```

hoặc:

```bash
--batch-size 4
```

## Luồng lệnh ngắn gọn

Khi môi trường và raw data đã sẵn sàng, toàn bộ workflow gồm ba lệnh chính:

```bash
python scripts/data/prepare_isic2018.py --val-ratio 0.2 --seed 42

python scripts/training/train_unet.py \
  --epochs 50 \
  --image-size 256 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda

python scripts/evaluation/evaluate.py \
  --checkpoint checkpoints/unet_best.pt \
  --split test \
  --output-dir results/unet_test \
  --num-visualizations 12
```
