# Hướng dẫn framework và chạy thực nghiệm segmentation

Tài liệu này mô tả contract giữa các module và toàn bộ vòng đời một experiment. Mọi lệnh được chạy từ thư mục gốc repository.

## 1. Cấu trúc framework

### Experiment config

Mỗi experiment được mô tả bởi một YAML, ví dụ [`configs/unet.yaml`](../configs/unet.yaml). File gồm các nhóm sau:

- `experiment`: tên experiment và thư mục gốc chứa runs.
- `seed`: seed cho model initialization, augmentation và DataLoader shuffle.
- `dataset`: dataset registry name, data root local, tracked manifest và preprocessing.
- `model`: model registry name cùng tham số riêng của kiến trúc.
- `loss`: loss registry name cùng tham số constructor.
- `optimizer`: hiện hỗ trợ AdamW.
- `scheduler`: `none`, `cosine` hoặc `reduce_on_plateau`.
- `training`: epochs, batch size, workers, device, AMP và threshold.

Cấu hình mẫu:

```yaml
experiment:
  name: unet_isic2018
  output_root: runs

seed: 42

dataset:
  name: isic2018
  root: dataset/isic2018_task1
  manifest: manifests/isic2018_task1_v1.json
  version: isic2018-task1-v1
  task: binary
  num_classes: 1
  in_channels: 3
  image_size: [256, 256]

model:
  name: unet
  base_channels: 32

loss:
  name: bce_dice
  bce_weight: 0.5
  dice_weight: 0.5

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.00001

scheduler:
  name: cosine
  eta_min: 0.0

training:
  epochs: 50
  batch_size: 16
  num_workers: 8
  device: cuda
  amp: true
  prediction_threshold: 0.5
  log_interval: 20
  gradient_clip_norm: null
```

#### Đổi loss qua YAML

Chỉ thay block `loss`; các phần dataset, model và training giữ nguyên. Ba loss có sẵn:

```yaml
# Binary cross-entropy
loss:
  name: bce
  pos_weight: null
  reduction: mean
```

```yaml
# Dice loss
loss:
  name: dice
  smooth: 1.0
  epsilon: 1.0e-7
```

```yaml
# Kết hợp BCE và Dice
loss:
  name: bce_dice
  bce_weight: 0.5
  dice_weight: 0.5
  pos_weight: null
  dice_smooth: 1.0
  dice_epsilon: 1.0e-7
```

Các kwargs không cần chỉnh có thể bỏ để dùng mặc định. Với `bce`, nên giữ `reduction: mean` để trainer nhận scalar loss. Khi so sánh loss trong research, nên đổi thêm `experiment.name` (ví dụ `unet_isic2018_dice`) để run folder dễ đọc; config fingerprint vẫn tự ngăn summary gộp nhầm các loss khác nhau.

Path tương đối được resolve từ repository root. Runner lưu resolved config thực sự đã dùng vào run folder. Các trường `task`, `in_channels` và `num_classes` lấy từ dataset rồi inject vào model để tránh hai source of truth.

### Module contracts

Luồng build module:

```text
YAML
 ├─ dataset.name ──> DATASET_REGISTRY ──> Dataset/DataLoader
 ├─ model.name   ──> MODEL_REGISTRY   ──> BaseSegmentationModel
 ├─ loss.name    ──> LOSS_REGISTRY    ──> loss module
 ├─ optimizer    ──> AdamW
 └─ scheduler    ──> scheduler builder
```

Mọi registered model phải tuân theo:

```python
output = model(images)
assert isinstance(output, SegmentationOutput)
logits = output.logits
```

Binary segmentation dùng shape:

```text
images : float32 [B, 3, H, W]
masks  : float32 [B, 1, H, W]
logits : float   [B, 1, H, W]
```

Model trả raw logits. Không gọi sigmoid trước BCE, Dice hoặc BCE+Dice. Sigmoid chỉ được dùng khi tính prediction và metrics.

### Frozen dataset split

Dữ liệu ảnh/mask local nằm trong `dataset/` và không được commit. Split chính thức nằm trong:

```text
manifests/isic2018_task1_v1.json
```

Manifest chứa relative paths cho `training`, `validation` và `test`, cùng dataset version, split seed và validation ratio. Loader đọc trực tiếp file này; training seed không tạo hoặc thay đổi split.

Mỗi run lưu cả manifest version và SHA-256 trong metadata. Khi tạo version split mới, tạo file manifest mới và đổi `dataset.version` cùng `dataset.manifest`; không sửa âm thầm version cũ sau khi đã có kết quả research.

### Run folder và checkpoint

Run mới được đặt tại:

```text
runs/<experiment>/<UTC timestamp>_seed-<seed>/
├── config.yaml
├── metadata.json
├── history.json
├── best.pt
├── last.pt
└── test_metrics.json
```

- `best.pt`: epoch có validation Dice cao nhất, dùng cho final test.
- `last.pt`: trạng thái gần nhất để resume.
- `history.json`: train loss, validation loss, Dice và IoU theo epoch.
- `test_metrics.json`: loss, Dice và IoU của `best.pt` trên test.
- `metadata.json`: seed, Git commit/dirty state, manifest hash, device, timestamps, status và config fingerprint.

Checkpoint version mới chứa model, optimizer, AMP scaler, scheduler, epoch và best validation Dice. Checkpoint legacy vẫn evaluate được nhưng không resume được.

## 2. Chạy thực nghiệm

### Chuẩn bị môi trường và dữ liệu

Project dùng Python 3.10 trong môi trường Conda:

```bash
conda create --name medical-seg python=3.10 pip -y
conda activate medical-seg
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ở các phiên làm việc sau, chỉ cần chạy `conda activate medical-seg` trước các lệnh của project.

Chuẩn bị ISIC 2018 từ raw data:

```bash
bash scripts/02_prepare_isic2018_dataset.sh
```

Hoặc cập nhật manifest từ dataset đã có:

```bash
python scripts/data/prepare_isic2018.py \
  --data-root dataset/isic2018_task1 \
  --manifest-output manifests/isic2018_task1_v1.json \
  --val-ratio 0.2 \
  --seed 42
```

Kết quả split hiện tại là 2.075 train, 519 validation và 100 test samples. `--seed` ở lệnh prepare là split-generation seed, khác với training seed trong YAML.

### Fresh run và smoke run

Chạy config chính:

```bash
python scripts/run_experiment.py --config configs/unet.yaml
```

Runner validate config, tạo run folder, train/validate, chọn best checkpoint, tự evaluate test rồi ghi metadata hoàn tất.

Trước một run dài, tạo config smoke riêng hoặc copy config mẫu và đặt:

```yaml
experiment:
  name: unet_isic2018_smoke
training:
  epochs: 1
  device: cpu
  num_workers: 0
```

Các section khác vẫn phải được giữ đầy đủ. Không dùng test metrics để chọn learning rate, architecture hoặc epoch; model selection chỉ dựa trên validation.

Script cũ vẫn chạy:

```bash
python scripts/training/train_unet.py --epochs 1 --device cpu --no-amp
```

Đây là compatibility wrapper và sẽ cảnh báo deprecation. Artifact vẫn đi vào standardized run folder; các path checkpoint/history legacy không còn điều khiển từng file.

### Resume một run

Resume bằng run folder, không truyền config ngoài:

```bash
python scripts/run_experiment.py \
  --resume runs/unet_isic2018/20260824T120000Z_seed-42
```

Runner dùng `config.yaml` trong run, load `last.pt`, đọc history và tiếp tục từ epoch kế tiếp. `training.epochs` là tổng epoch mục tiêu. Resume phù hợp cho run bị gián đoạn; nếu checkpoint đã đạt target, training được bỏ qua và best checkpoint được test lại.

Nếu thay model, dataset, loss, optimizer hoặc scheduler, phải tạo run mới thay vì sửa config đã lưu.

### Multi-seed summary và visualization

Để chạy nhiều seed, tạo các YAML có cùng hyperparameters/experiment name và chỉ đổi `seed`. Mỗi lệnh tạo run folder riêng:

```bash
python scripts/run_experiment.py --config configs/unet_seed42.yaml
python scripts/run_experiment.py --config configs/unet_seed123.yaml
python scripts/run_experiment.py --config configs/unet_seed999.yaml
```

Tổng hợp:

```bash
python scripts/summarize_experiments.py \
  --runs-root runs \
  --experiment unet_isic2018 \
  --output runs/unet_isic2018/summary.csv
```

Script in bảng từng run và mean ± sample standard deviation. Chỉ các run có cùng experiment name và config fingerprint mới được group; seed và output root không thuộc fingerprint. Một run hiển thị standard deviation là `N/A`.

Final test đã chạy tự động nhưng không tạo ảnh. Khi cần visualization:

```bash
python scripts/evaluation/evaluate.py \
  --checkpoint runs/unet_isic2018/<run-id>/best.pt \
  --data-root dataset/isic2018_task1 \
  --split test \
  --output-dir runs/unet_isic2018/<run-id>/visualization \
  --device cuda \
  --num-visualizations 12
```

## 3. Mở rộng module

### Thêm model

1. Kế thừa `BaseSegmentationModel`.
2. Constructor nhận shared fields `in_channels`, `num_classes`, `task`.
3. `forward()` trả `SegmentationOutput(logits=...)`.
4. Đăng ký bằng `@register_model("name")`.
5. Import module trong `src/models/__init__.py`.
6. Thêm shape/forward/backward unit test.

Không thêm nhánh model-specific vào trainer hoặc runner. Model có auxiliary outputs đặt chúng trong `aux_logits`; routing/debug information đặt trong `diagnostics`.

### Thêm loss

Loss nhận `logits, targets` và trả scalar Tensor:

```python
@register_loss("my_loss")
class MyLoss(nn.Module):
    def forward(self, logits, targets):
        return scalar_loss
```

Import loss trong `src/losses/__init__.py`, sau đó dùng:

```yaml
loss:
  name: my_loss
  custom_parameter: 0.5
```

Không gọi sigmoid ở runner. Loss tự quyết định cách chuyển logits nếu cần.

### Thêm dataset

Dataset mới kế thừa `BaseSegmentationDataset`, đăng ký bằng `@register_dataset` và trả tối thiểu:

```python
{
    "image": image_tensor,
    "mask": mask_tensor,
    "sample_id": sample_id,
}
```

Manifest phải là artifact versioned và tracked bởi Git; data root chỉ chứa file local. Dataset class chịu trách nhiệm map `train`, `val`, `test` sang các key manifest phù hợp và báo lỗi nếu split thiếu/rỗng.

### Scheduler và optimizer

Scheduler v1 cố ý chỉ có:

- `none`.
- `cosine`: `T_max` tự lấy tổng epochs, config nhận `eta_min`.
- `reduce_on_plateau`: luôn monitor validation loss với mode `min`, config nhận `factor`, `patience`, `min_lr`.

Optimizer v1 chỉ hỗ trợ AdamW. Chỉ mở rộng builder khi experiment thực tế cần optimizer mới; chưa cần registry hoặc callback system.

## 4. Kiểm thử và xử lý lỗi

### Unit tests và CI

Chạy toàn bộ CPU tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests bao phủ model/loss/dataset/evaluator, strict output contract, config validation, frozen split, standardized run artifacts, scheduler resume và multi-seed summary.

GitHub Actions dùng Python 3.10, PyTorch CPU và `requirements-ci.txt`. CI không tải dataset thật và không chạy GPU training.

### Lỗi config và dữ liệu

- `Experiment config not found`: kiểm tra path sau `--config`.
- `Configuration section ... is missing`: giữ đủ tất cả top-level sections trong YAML.
- `Tracked dataset manifest not found`: chạy prepare script hoặc sửa `dataset.manifest`.
- `Image/Mask not found`: kiểm tra `dataset.root`; paths trong manifest là relative với root này.
- Manifest hash khác giữa runs: không gộp kết quả trước khi xác minh dataset version/split.

### Lỗi device, output và resume

- `CUDA was requested...`: cài đúng CUDA PyTorch hoặc đổi `training.device: cpu` và `training.amp: false`.
- Hết VRAM: giảm batch size hoặc image size; không nhầm batch size với image size.
- `Model forward must return SegmentationOutput`: bọc raw network bằng registered model adapter.
- `Resume requires a version-2 experiment checkpoint`: checkpoint cũ chỉ dùng evaluation; bắt đầu run lifecycle mới để có resume.
- History và checkpoint lệch epoch: không copy `last.pt` giữa các run folder; khôi phục đúng cặp `last.pt` và `history.json`.
- Run có metadata `failed`: đọc trường `error`; artifacts trước lỗi vẫn được giữ để debug.
