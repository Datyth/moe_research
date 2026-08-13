# SegMoTE Research

Môi trường và các script hỗ trợ tái chạy SegMoTE trên bộ dữ liệu ISIC 2018.
Lộ trình thực nghiệm chi tiết nằm trong [`PLAN.md`](PLAN.md).

## Yêu cầu

- Conda (Miniconda hoặc Anaconda)
- Python 3.10
- GPU NVIDIA và driver tương thích nếu chạy bằng CUDA

Python 3.10 được chọn để tương thích với PyTorch, MONAI và mã nguồn
SegMoTE hiện tại.

## Cài đặt bằng Conda

Tạo và kích hoạt môi trường:

```bash
conda create --name segmote python=3.10 pip -y
conda activate segmote
```

Nâng cấp pip và cài các dependency:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu chạy trên GPU NVIDIA, hãy chọn bản PyTorch hỗ trợ CUDA phù hợp với driver
đang cài trên máy. Có thể cài PyTorch CUDA trước, sau đó cài các dependency
còn lại từ `requirements.txt`.

Kiểm tra môi trường và GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

## Chuẩn bị checkpoint

Tải checkpoint vào `SegMoTE/checkpoints/`:

```bash
hf download yujielu/SegMoTE \
  --local-dir SegMoTE/checkpoints
```

## Chuẩn bị ISIC 2018

Script sau tải ảnh/mask và chuyển dữ liệu sang định dạng SegMoTE:

```bash
bash scripts/02_prepare_isic2018_dataset.sh
```

Dữ liệu sinh ra nằm tại `SegMoTE/dataset/isic2018_task1/` và không được đưa
vào Git.

## Chạy thử

Chạy smoke training một epoch trên CUDA:

```bash
DEVICE=cuda bash scripts/training/smoke_training.sh
```

Chạy validation:

```bash
DEVICE=cuda bash scripts/valid/valid_isic2018.sh
```

Có thể thay đổi các biến `EPOCHS`, `BATCH_SIZE`, `IMAGE_SIZE` và `CHECKPOINT`
trước lệnh để điều chỉnh cấu hình chạy.
