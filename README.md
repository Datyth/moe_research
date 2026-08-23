# Medical Image Segmentation Research Framework

Framework thực nghiệm deep learning cho medical image segmentation, được thiết kế để huấn luyện và so sánh nhiều kiến trúc model trên cùng dataset, loss và training contract.

## Current Working

> Cập nhật lần cuối: 2026-08-23 +07 (UTC+07:00)
>
> Commit cơ sở: 4aaa854
> Quy ước: cập nhật timestamp và checklist sau mỗi mốc công việc.

### Đã hoàn thành

- [x] Model API dùng chung: BaseSegmentationModel, SegmentationOutput và model registry.
- [x] Baseline UNet cho binary segmentation.
- [x] Dataset API, dataset registry và DatasetConfig dùng chung.
- [x] ISIC 2018 loader với deterministic split train/validation/test và sparse mask.
- [x] Transform đồng bộ image–mask: resize, augmentation và normalization.
- [x] Binary losses: BCE, Dice và BCE + Dice.
- [x] Binary evaluator với sample-mean Dice/IoU và sample-weighted loss.
- [x] Validation mỗi epoch, best/last checkpoint và JSON training history.
- [x] CUDA AMP, AdamW training và checkpoint tự mô tả model/data/loss config.
- [x] Test evaluation CLI và visualization Input/GT/Prediction/Overlay.
- [x] Unit/smoke tests cho model, dataset, transform, losses, evaluator và trainer.
- [x] GPU smoke end-to-end một epoch trên NVIDIA RTX PRO 4000 Blackwell.
  - 2.075 train, 519 validation và 100 test samples; image size 256 × 256.
  - Train loss 0.529859; validation loss 0.468268.
  - Validation Dice/IoU: 0.729197/0.613759.
  - Test loss 0.450153; test Dice/IoU: 0.740456/0.620748.
  - Output local: `checkpoints/smoke/` và `results/smoke/`.

### Tiếp theo

- [ ] Chạy baseline UNet dài hạn và ghi nhận test metrics chính thức.
- [ ] Hỗ trợ resume training và LR scheduler.
- [ ] Thêm model mới thông qua base class và registry.
- [ ] Thêm experiment config chung cho model, dataset, loss, optimizer và trainer.
- [ ] Đưa SegMoTE vào framework dưới dạng model adapter sau khi baseline chung ổn định.

## Nguyên tắc thiết kế

Các thành phần được tách độc lập:

~~~text
DatasetConfig → Dataset/Transform → DataLoader
                                      ↓
ModelConfig   → Model Registry → SegmentationOutput.logits
                                      ↓
Loss          → scalar loss → Trainer → Checkpoint
~~~

Trainer không phụ thuộc trực tiếp vào UNet hoặc SegMoTE. Model hợp lệ chỉ cần trả một Tensor logits hoặc object có thuộc tính logits.

Quy ước binary segmentation:

~~~text
image   : float32 [B, 3, H, W]
mask    : float32 [B, 1, H, W]
logits  : float   [B, 1, H, W]
~~~

Model forward luôn trả raw logits. Không gọi sigmoid trước BCE hoặc BCE + Dice loss.

## Cấu trúc project

~~~text
src/
├── configs/
│   └── dataset.py
├── data/
│   ├── base.py
│   ├── registry.py
│   ├── isic2018.py
│   └── transforms.py
├── models/
│   ├── base.py
│   ├── registry.py
│   └── unet.py
├── losses/
│   ├── bce.py
│   ├── dice.py
│   └── combined.py
├── engine/
│   ├── trainer.py
│   └── evaluator.py
└── tests/
    ├── test_model.py
    ├── test_dataset.py
    ├── test_losses.py
    ├── test_trainer.py
    └── test_evaluator.py

scripts/
├── data/prepare_isic2018.py
├── training/train_unet.py
└── evaluation/evaluate.py

dataset/       # local data, ignored by Git
checkpoints/   # generated checkpoints, ignored by Git
results/       # metrics/history/visualizations, ignored by Git
SegMoTE/       # external/reference implementation
~~~

## Cài đặt

Yêu cầu:

- Python 3.10.
- PyTorch.
- GPU NVIDIA và CUDA nếu training trên GPU.

Tạo môi trường:

~~~bash
conda create --name medical-seg python=3.10 pip -y
conda activate medical-seg
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Kiểm tra PyTorch và CUDA:

~~~bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
~~~

## Dataset ISIC 2018

Generic pipeline hiện đọc dữ liệu tại:

~~~text
dataset/isic2018_task1/
├── dataset.json
├── images/
│   ├── train/
│   └── test/
├── labels/
│   ├── train/
│   └── test/
└── pseudo/
    └── train/
~~~

Mask được lưu dưới dạng sparse NPZ và được đọc bằng scipy.sparse.load_npz.

Tạo hoặc cập nhật deterministic validation split:

~~~bash
python scripts/data/prepare_isic2018.py --val-ratio 0.2 --seed 42
~~~

Manifest chứa `training`, `validation` và `test`. Validation là 20% official training records và tham chiếu các file vật lý trong `images/train`/`labels/train`, nên không nhân bản dữ liệu. Split test đến từ official ISIC 2018 validation archive và chỉ dùng cho đánh giá cuối.

Để tải, giải nén và chuẩn bị dữ liệu từ đầu:

~~~bash
bash scripts/02_prepare_isic2018_dataset.sh
~~~

## Chạy tests

Chạy toàn bộ test suite:

~~~bash
python -m unittest discover -s src/tests -p 'test_*.py' -v
~~~

Chạy từng nhóm:

~~~bash
python src/tests/test_model.py
python src/tests/test_dataset.py
python src/tests/test_losses.py
python src/tests/test_trainer.py
python src/tests/test_evaluator.py
~~~

Trạng thái gần nhất: 18 tests passed.

## Train UNet

Ví dụ training một epoch trên GPU:

~~~bash
python scripts/training/train_unet.py \
  --epochs 1 \
  --image-size 256 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda
~~~

Không nhầm image size với batch size. Với pipeline hiện tại, batch size 16–32 phù hợp hơn batch size 256 vì image decode, sparse-mask conversion và resize vẫn chạy trên CPU.

Mỗi epoch chạy validation và cập nhật:

~~~text
checkpoints/unet_best.pt
checkpoints/unet_last.pt
results/unet_training_history.json
~~~

Checkpoint chứa:

- Epoch.
- Model state dict.
- Optimizer state dict.
- AMP scaler state.
- Train loss, validation loss, Dice và IoU.
- Best validation Dice.
- Trainer config.
- Model/data/loss metadata để evaluation dựng lại đúng experiment.

## Evaluate checkpoint

~~~bash
python scripts/evaluation/evaluate.py \
  --checkpoint checkpoints/unet_best.pt \
  --split test \
  --output-dir results/unet_test \
  --num-visualizations 12
~~~

Output:

~~~text
results/unet_test/
├── metrics.json
└── visualizations/
    └── ISIC_xxx.png
~~~

Mỗi visualization gồm bốn panel: Input, Ground Truth, Prediction và Boundary Overlay. Panel cuối vẽ biên ground truth màu xanh lá, biên prediction màu đỏ và phần biên trùng nhau màu vàng trực tiếp trên input để làm rõ sai số. Checkpoint UNet cũ không có metadata vẫn được hỗ trợ với fallback `base_channels=32`, image size 256 và BCE+Dice 0.5/0.5; có thể override bằng CLI.

## Thêm model mới

1. Kế thừa BaseSegmentationModel.
2. Cài đặt forward và trả SegmentationOutput.
3. Đăng ký model bằng register_model.
4. Import model trong src/models/__init__.py.
5. Thêm smoke test shape/forward.
6. Dùng cùng dataset, loss và Trainer hiện tại.

Ví dụ contract:

~~~python
@register_model("my_model")
class MySegmentationModel(BaseSegmentationModel):
    def forward(self, images, **kwargs):
        logits = self.network(images)
        return SegmentationOutput(logits=logits)
~~~

Mục tiêu là thay đổi model qua config/registry, không thêm nhánh if/elif riêng trong Trainer.

## Losses hiện có

~~~python
from src.losses import BCELoss, DiceLoss, BCEDiceLoss

criterion = BCEDiceLoss(
    bce_weight=0.5,
    dice_weight=0.5,
)

output = model(images)
loss = criterion(output.logits, masks)
~~~

Các loss hiện tại dành cho binary segmentation.

## SegMoTE reference

Thư mục SegMoTE giữ code upstream/reference phục vụ reproduction và nghiên cứu Mixture-of-Experts. Nó không phải kiến trúc trung tâm của framework.

Các script legacy vẫn có thể sử dụng:

~~~bash
DEVICE=cuda bash scripts/training/smoke_training.sh
DEVICE=cuda bash scripts/valid/valid_isic2018.sh
~~~

Lộ trình SegMoTE chuyên biệt trước đây nằm trong [PLAN.md](PLAN.md). Framework chung trong src nên được ưu tiên khi thêm dataset, loss, trainer hoặc model mới.
