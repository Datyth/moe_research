# Medical Image Segmentation Research Framework

Framework thực nghiệm deep learning cho medical image segmentation, được thiết kế để huấn luyện và so sánh nhiều kiến trúc model trên cùng dataset, loss và training contract.

## Current Working

> Cập nhật lần cuối: 2026-08-15 18:03:56 +07 (UTC+07:00)
>
> Commit cơ sở: 49c71a6
> Quy ước: cập nhật timestamp và checklist sau mỗi mốc công việc.

### Đã hoàn thành

- [x] Model API dùng chung: BaseSegmentationModel, SegmentationOutput và model registry.
- [x] Baseline UNet cho binary segmentation.
- [x] Dataset API, dataset registry và DatasetConfig dùng chung.
- [x] ISIC 2018 loader với split train/test và sparse mask.
- [x] Transform đồng bộ image–mask: resize, augmentation và normalization.
- [x] Binary losses: BCE, Dice và BCE + Dice.
- [x] Trainer cơ bản dùng được cho model trả Tensor hoặc output có thuộc tính logits.
- [x] CUDA AMP, AdamW training và lưu checkpoint model/optimizer/scaler.
- [x] Unit/smoke tests cho model, dataset, transform, losses và trainer.
- [x] Train thử UNet một epoch trên NVIDIA RTX PRO 4000 Blackwell.
  - 2.594 training samples.
  - Image size 256 × 256.
  - Mean training loss: 0.495141.
  - Checkpoint local: checkpoints/unet_initial.pt.

### Đang thực hiện

- [ ] Tối ưu data pipeline để giảm thời gian GPU chờ CPU.
- [ ] Giảm các điểm đồng bộ CPU–GPU không cần thiết trong Trainer.
- [ ] Chuẩn hóa entrypoint/config để thay model mà không cần viết lại training script.

### Tiếp theo

- [ ] Thêm validation split độc lập; không dùng test set để tuning.
- [ ] Viết evaluator và metrics Dice/IoU.
- [ ] Hỗ trợ resume training và best/last checkpoint.
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
    └── test_trainer.py

scripts/
├── training/
│   └── train_unet.py
└── ...

dataset/       # local data, ignored by Git
checkpoints/   # generated checkpoints, ignored by Git
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

Lưu ý: split test hiện được tạo từ official validation archive của ISIC 2018. Không sử dụng split này để chọn hyperparameter. Cần tách validation từ training set trước khi chạy thực nghiệm chính thức.

Các script trong scripts/01_prepare_segmote.sh và scripts/02_prepare_isic2018_dataset.sh thuộc workflow baseline/reference cũ. Cần tiếp tục chuẩn hóa data-preparation entrypoint cho generic pipeline.

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
~~~

Trạng thái gần nhất: 10 tests passed.

## Train UNet

Ví dụ training một epoch trên GPU:

~~~bash
python scripts/training/train_unet.py \
  --epochs 1 \
  --image-size 256 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --checkpoint checkpoints/unet_initial.pt
~~~

Không nhầm image size với batch size. Với pipeline hiện tại, batch size 16–32 phù hợp hơn batch size 256 vì image decode, sparse-mask conversion và resize vẫn chạy trên CPU.

Checkpoint chứa:

- Epoch.
- Model state dict.
- Optimizer state dict.
- AMP scaler state.
- Mean training loss.
- Trainer config.

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
