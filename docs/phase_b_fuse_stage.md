# Phase B — fuse stage

Phase B kết hợp hai nhánh để tạo biểu diễn định tuyến đặc quyền
`h_q = [h_I ; h_M]`. Đây là mốc "completely fuse stage" trong proposal.
Posterior `q(z | I, M)`, prior `p(z | I)` và Top-K routing **đã được triển
khai ở giai đoạn kế tiếp** — xem
[`docs/phase_b_router.md`](phase_b_router.md); hierarchical enhancement
và việc nối expert vẫn là bước sau.

```
I --> SAM ViT-B (MoE-SAM, E3) --> F^(l), l ∈ {3,6,9,12} --> h_I  ┐
                                                                 ├──> h_q ∈ R^{B×512}
M --> Shape Teacher G_M (Phase A, đóng băng)            --> h_M  ┘
```

## 1. Thành phần

| Ký hiệu | Shape | Module |
| --- | --- | --- |
| `F^(l)` | `[B, 16, 16, 768]` | 12 block output của SAM ViT-B, lấy tại layer 3/6/9/12 |
| `u^(l)` | `[B, 256]` | `P_l` (Conv2d 1×1 riêng từng level) rồi GAP |
| `α_l` | `[B, 4]` | `g_level` (MLP dùng chung) rồi softmax trên chiều level |
| `h_I` | `[B, 256]` | `Σ_l α_l u^(l)` |
| `h_M` | `[B, 256]` | `G_M(M)` — encoder + projector từ Phase A, bỏ decoder |
| `h_q` | `[B, 512]` | `concat([h_I, h_M])` |
| `X^(l)` | `[B, 256, 768]` | token gốc, giữ lại cho MoE enhancement giai đoạn sau |

Code tương ứng:

| File | Vai trò |
| --- | --- |
| [`src/models/phase_b/image_descriptor.py`](../src/models/phase_b/image_descriptor.py) | `MultiLevelImageDescriptor` → `h_I`, `α`, `X^(l)` |
| [`src/models/phase_b/shape_teacher.py`](../src/models/phase_b/shape_teacher.py) | `ShapeTeacher` = `G_M`, `load_shape_teacher` đọc checkpoint Phase A |
| [`src/models/phase_b/fusion.py`](../src/models/phase_b/fusion.py) | `PrivilegedFusion` → `h_q` |
| [`src/models/phase_b/fuse_stage.py`](../src/models/phase_b/fuse_stage.py) | `PhaseBFuseStage`, model đăng ký tên `phase_b_fuse` |
| [`src/tasks/phase_b_fuse.py`](../src/tasks/phase_b_fuse.py) | `PhaseBFuseTask`, truyền mask ground-truth vào model |

## 2. Điều cần biết trước khi chạy

**`h_I` chưa được huấn luyện — riêng ở fuse stage.** Từ mốc router
(`phase_b_router`, xem [`docs/phase_b_router.md`](phase_b_router.md))
trở đi, KL và load-balance loss bắt đầu tác động lên `h_I` và level
attention. Riêng tại mốc fuse stage này chưa có posterior nên không có
loss nào tác động lên `h_I` hay level attention; gradient của chúng bằng 0.
Hệ quả:

- `val_level_weight_entropy` sẽ đứng yên ở `log(4) ≈ 1.386` (phân bố đều).
- Chất lượng segmentation **trùng khớp bit-for-bit với E3** ở cùng seed. Đây
  là tính chất mong muốn: nó chứng minh fuse stage được nối vào mà không làm
  nhiễu baseline. Đã kiểm chứng: `max |logit_E3 − logit_phaseB| = 0.0`.

**`h_M` là thông tin đặc quyền.** Shape Teacher bị đóng băng (0 tham số
trainable), luôn ở chế độ eval, và chỉ hoạt động khi mask được truyền vào.
`PhaseBFuseStage.forward(images)` không có `masks=` vẫn chạy và chỉ đơn giản
không sinh `h_q` — đúng với hành vi ở Phase C và lúc inference.

## 3. Chuẩn bị Shape Teacher

Fuse stage cần một checkpoint Phase A. Bản đang dùng được train 10 epoch để
overfit (chủ ý: `h_M` chỉ cần giữ hình học tổn thương, không cần khả năng
tổng quát hoá):

```bash
python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_10ep.yaml
```

Kết quả tham chiếu (`runs/phase_a_s0_small_cnn_10ep/20260910T185518Z_seed-42`):

| | val Dice | test Dice |
| --- | --- | --- |
| smoke, 1 epoch | 0.8957 | 0.9137 |
| **10 epoch** | **0.9583** (epoch 9) | **0.9672** |

## 4. Chạy Phase B

```bash
# Kiểm tra pipeline trong 1 epoch
python scripts/run_experiment.py --config configs/phase_b/isic2018_fuse_smoke.yaml

# Chạy đầy đủ 50 epoch
python scripts/run_experiment.py --config configs/phase_b/isic2018_fuse.yaml
```

`configs/phase_b/isic2018_fuse.yaml` kế thừa `configs/isic2018_common.yaml`
(ISIC2018 tại `dataset/isic2018_task1`) và thêm:

```yaml
model:
  name: phase_b_fuse
  descriptor_dim: 256              # C_s
  levels: [3, 6, 9, 12]            # L
  shape_teacher_checkpoint: runs/phase_a_s0_small_cnn_10ep/<RUN_ID>/best.pt
  freeze_shape_teacher: true

task:
  name: phase_b_fuse               # truyền mask vào model làm privileged input
```

Đường dẫn trong `model.checkpoint` và `model.shape_teacher_checkpoint` được
resolve theo project root nên config chạy được từ thư mục làm việc bất kỳ.

## 5. Kiểm chứng đã thực hiện

- 22 unit test trong [`tests/test_phase_b_fuse.py`](../tests/test_phase_b_fuse.py):
  shape của `h_I`/`h_M`/`h_q`, `α` là softmax và tổng bằng 1, level index là
  1-based đúng theo `L = {3,6,9,12}`, teacher tái tạo đúng latent của
  autoencoder Phase A, teacher đóng băng không sinh gradient, và model không
  sinh `h_q` thì task báo lỗi rõ ràng thay vì im lặng thoái hoá về MoE-SAM.
- Smoke run 1 epoch trên ISIC2018: test Dice 0.8643, `level_weight_entropy`
  1.3841 (≈ log 4, đúng như dự đoán ở mục 2).
