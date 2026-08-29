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
- `scheduler`: `none`, `cosine`, `reduce_on_plateau` hoặc `warmup_poly`.
- `training`: epochs, batch size, workers, device, AMP và threshold.

Cấu hình mẫu:

```yaml
experiment:
  name: unet_isic2018
  output_root: runs

seed: 42

dataset:
  name: isic2018
  root: /home/teama/projects/project_01/dataset/isic2018_task1
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
  amp_dtype: float16
  prediction_threshold: 0.5
  boundary_tolerance: 2
  log_interval: 20
  gradient_clip_norm: null
```

`training.amp_dtype` nhận `float16` (mặc định) hoặc `bfloat16`. Đổi sang `bfloat16` khi thấy `FloatingPointError: Non-finite loss detected` giữa chừng training trên GPU hỗ trợ bf16 (Ampere trở lên) — `float16` có dải số mũ hẹp (±65504) nên các kiến trúc có phép nhân không chuẩn hóa ở tầng cuối (ví dụ mask decoder kiểu SAM trong `esam`) dễ tràn số sau vài epoch dù đã bật `gradient_clip_norm`; `bfloat16` có cùng dải số mũ với `float32` nên không gặp lỗi này, đổi lại giảm độ chính xác mantissa. `GradScaler` tự động vô hiệu hoá khi dùng `bfloat16` vì loss-scaling chỉ cần thiết cho `float16`.

#### Chia sẻ hyperparameter chung giữa nhiều config bằng `extends`

Khi nhiều experiment (ví dụ các hàng trong 1 bảng ablation) cần giữ nguyên `dataset`/`loss`/`optimizer`/`scheduler`/`training`, chỉ khác `model`, một file có thể kế thừa file khác thay vì copy toàn bộ:

```yaml
# configs/isic2018_common.yaml — base, thiếu experiment/model nên không tự chạy được
seed: 42
dataset: {...}
loss: {...}
optimizer: {...}
scheduler: {...}
training: {...}
```

```yaml
# configs/isic2018_e1.yaml
extends: isic2018_common.yaml   # path tương đối so với file chứa nó

experiment:
  name: isic2018_e1
  output_root: runs

model:
  name: esam
  ...
```

`load_experiment_config` (dùng bởi `scripts/run_experiment.py`) merge đệ quy: key nào file con khai thì đè lên base ở đúng cấp lồng nhau đó (ví dụ chỉ override `training.epochs` mà không cần chép lại cả `training`), key nào không khai thì giữ nguyên từ base. `extends` có thể chain (base cũng được phép `extends` file khác) nhưng không được vòng lặp. Việc merge diễn ra **trước** validate và trước khi snapshot vào `runs/<experiment>/<run-id>/config.yaml`, nên run đã chạy vẫn giữ nguyên bản config đầy đủ, độc lập với `isic2018_common.yaml` có bị sửa sau này hay không.

Ví dụ thật trong repo: [`configs/isic2018_common.yaml`](../configs/isic2018_common.yaml) là base cho [`isic2018_e0.yaml`](../configs/isic2018_e0.yaml), [`isic2018_e1.yaml`](../configs/isic2018_e1.yaml) và [`isic2018_smoke.yaml`](../configs/isic2018_smoke.yaml) (file này còn override thêm `scheduler`/`training` cho 1 epoch chạy nhanh).

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

Dữ liệu ảnh/mask local nằm trong `/home/teama/projects/project_01/dataset/` và không được commit. Split chính thức nằm trong:

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
- `test_metrics.json`: Loss, Dice, IoU, HD95, ASSD và Boundary F1 của
  `best.pt` trên test.
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
  --data-root /home/teama/projects/project_01/dataset/isic2018_task1 \
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

Summary yêu cầu đủ cả sáu metric. Run cũ thiếu HD95, ASSD hoặc Boundary F1 sẽ
được bỏ qua với warning rõ ràng; script không gán metric thiếu thành 0.

### Ý nghĩa evaluation metrics

- Dice và IoU đo độ chồng lấp vùng segmentation.
- HD95 là percentile 95 của tập hợp hai chiều các khoảng cách gần nhất giữa
  boundary dự đoán và boundary ground truth; đây là robust worst-case boundary
  distance.
- ASSD là trung bình có trọng số theo số boundary pixel của cùng hai tập khoảng
  cách có hướng, thể hiện average boundary distance.
- Boundary F1 đo precision/recall của boundary. Một boundary pixel được match khi
  khoảng cách Euclidean gần nhất tới boundary còn lại **nhỏ hơn hoặc bằng**
  `training.boundary_tolerance`; mặc định là 2 pixel.

Boundary được lấy bằng `mask & ~binary_erosion(mask)`. HD95 và ASSD hiện có đơn vị
pixel, không phải millimeter, vì pipeline ISIC 2D không cung cấp physical pixel
spacing. Nếu cả prediction và ground truth rỗng, HD95/ASSD bằng 0 và Boundary F1
bằng 1. Nếu chỉ một mask rỗng, Boundary F1 bằng 0; HD95 và ASSD nhận finite penalty
theo đường chéo ảnh `sqrt((H - 1)^2 + (W - 1)^2)`.

Final test đã chạy tự động nhưng không tạo ảnh. Khi cần visualization:

```bash
python scripts/evaluation/evaluate.py \
  --checkpoint runs/unet_isic2018/<run-id>/best.pt \
  --data-root /home/teama/projects/project_01/dataset/isic2018_task1 \
  --split test \
  --output-dir runs/unet_isic2018/<run-id>/visualization \
  --device cuda \
  --boundary-tolerance 2 \
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

Khi port kiến trúc từ một repo ngoài (ví dụ `src/models/esam/`), giữ code gốc trong subpackage `<model>/_vendor/` riêng, không sửa trộn vào wrapper. Mỗi file vendor ghi rõ trong docstring: nguồn gốc (link repo), và từng chỗ khác với upstream kèm lý do (thiếu file, bug, không tương thích với contract của framework này...). Wrapper ở `<model>/__init__.py` chỉ chịu trách nhiệm ánh xạ `in_channels`/`num_classes`/`task` sang tham số của kiến trúc gốc và implement `forward()` theo contract chung — không có logic model-specific nào rò rỉ ra ngoài subpackage vendor.

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
- `warmup_poly`: scheduler theo iteration (công thức của MoE-SAM/E-SAM), config nhận `warmup_steps` (mặc định 250) và `power` (mặc định 0.9). Warmup tuyến tính theo số bước optimizer, sau đó suy giảm đa thức `(1 - progress)**power` đến 0 ở cuối huấn luyện; `Trainer` tự step mỗi optimizer step thay vì mỗi epoch.

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
- `FloatingPointError: Non-finite loss detected`: thử `training.gradient_clip_norm` (ví dụ `1.0`) trước; nếu vẫn tràn số sau vài epoch trên `training.amp: true`, đổi `training.amp_dtype: bfloat16` (xem giải thích ở phần cấu hình mẫu phía trên).
- Hết VRAM: giảm batch size hoặc image size; không nhầm batch size với image size.
- `Model forward must return SegmentationOutput`: bọc raw network bằng registered model adapter.
- `Resume requires a version-2 experiment checkpoint`: checkpoint cũ chỉ dùng evaluation; bắt đầu run lifecycle mới để có resume.
- History và checkpoint lệch epoch: không copy `last.pt` giữa các run folder; khôi phục đúng cặp `last.pt` và `history.json`.
- Run có metadata `failed`: đọc trường `error`; artifacts trước lỗi vẫn được giữ để debug.
