# Phase B — router: posterior/prior và Top-K

Tài liệu này tiếp nối [`docs/phase_b_fuse_stage.md`](phase_b_fuse_stage.md):
giờ đây trên `h_q` có posterior đặc quyền `q(z | I, M)`, prior triển khai
được `p(z | I)`, và shape-aware Top-K router. Đây là mốc **đầu tiên mà
`h_I` nhận gradient** — KL và load-balance loss quay ngược về descriptor
và level attention của nó.

Router **chưa được nối vào expert nào**: quyết định định tuyến chỉ được ghi
vào diagnostics. Nối `pi`/`K_b` vào shape expert và hierarchical enhancement
(dùng `X^(l)`) là giai đoạn kế tiếp.

```
training (mask có sẵn — privileged):
  I --> SAM ViT-B (MoE-SAM) --> h_I ─┐
                                     ├─> h_q = [h_I ; h_M] ─> q(z|I,M) ─> z_q ~ rsample ─┐
  M --> Shape Teacher G_M (frozen) ──┘                                                    ├─> Top-K router
  I ──────────────────────────> h_I ─> p(z|I) <── KL(sg[q] || p)                         ┘   (prior bị kéo về posterior)

inference (không mask — deployable):
  I ─────> h_I ─> p(z|I) ─> z = mu_p (deterministic) ─> Top-K router
```

Hai luật chuyển đường, chọn theo sự hiện diện của mask trong
`PhaseBRouterStage.forward`:

- **mask có** (training/validation/test của framework): đi đường
  **posterior**; sample `z_q` bằng reparameterization khi đang train;
  sinh thêm `L_latent` và `L_bal`.
- **mask không có** (deploy): đi đường **prior**; `z = mu_p` xác định,
  không phụ thuộc mask. Segmentation vẫn chạy bình thường vì router không
  tác động vào mask decoder.

Đây là pattern privileged information (teacher–student): posterior thấy
ground-truth mask nên có tín hiệu định tuyến tốt hơn; KL chưng cất tín hiệu
đó vào prior chỉ đọc ảnh — nhánh còn sống lúc inference.

## 1. Thành phần

| Ký hiệu | Shape | Module |
| --- | --- | --- |
| `h_I` | `[B, 256]` | image descriptor của fuse stage |
| `h_M` | `[B, 256]` | `G_M(M)`, frozen, không nhận gradient |
| `h_q` | `[B, 512]` | `= [h_I ; h_M]`, input của posterior head |
| `mu, sigma` | `[B, d_z]`, `d_z = 64` | hai Gaussian chéo: `q(z\|I,M)` và `p(z\|I)` |
| `z` | `[B, 64]` | latent định tuyến thực dùng (rsample hoặc mean) |
| `r` | `[B, K]`, `K = 4` | logits router: `r = W_R z + b_R` |
| `pi_bar` | `[B, 4]` | `Softmax(r)` — phân bố dày (pre-Top-K) |
| `K_b` | `[B, k_e]`, `k_e = 2` | chỉ số expert được kích hoạt |
| `pi` | `[B, 4]` | sparse: zero ngoài `K_b`, tổng bằng 1 trên `K_b` |

Code tương ứng:

| File | Vai trò |
| --- | --- |
| [`src/models/phase_b/posterior.py`](../src/models/phase_b/posterior.py) | `GaussianParameterHead` → `(mu, sigma)`; `DiagonalGaussian` (rsample/detach); `gaussian_kl` |
| [`src/models/phase_b/router.py`](../src/models/phase_b/router.py) | `TopKRouter` → `RoutingOutput`; `load_balance_loss` |
| [`src/models/phase_b/phase_b_router.py`](../src/models/phase_b/phase_b_router.py) | `PhaseBRouterHead` (posterior + prior + router); `PhaseBRouterStage`, model `phase_b_router` |
| [`src/tasks/phase_b_router.py`](../src/tasks/phase_b_router.py) | `PhaseBRouterTask`: `L_seg + KL + L_bal`, metric định tuyến |

### Posterior/prior: `GaussianParameterHead`

Một lớp head duy nhất dùng cho cả hai vai trò, khác nhau ở input
(`in_dim` = 512 với posterior, 256 với prior):

- `mean_head`: `Linear(in_dim, d_z)` → `mu`.
- `scale_head`: `Linear(in_dim, d_z)` → `rho`, rồi
  `sigma = Softplus(rho) + std_floor` (`std_floor = 1e-4`).
  Softplus giữ `sigma > 0` và khả vi; floor chặn các số hạng
  `1/sigma_p^2` và `log sigma` trong KL bùng nổ khi scale bị đẩy về 0.

`DiagonalGaussian.rsample` vẽ `z = mu + sigma * eps`, `eps ~ N(0, I)` —
nhiễu nằm ở `eps` (không mang gradient) nên gradient của router chảy
về cả `mu` lẫn `sigma`. `detach()` tạo bản `sg[q]` cho KL.

### KL: `KL(sg[q] || p)`

Closed-form cho Gaussian chéo, per-sample `[B]` (sum theo chiều latent):

```
KL(q || p) = 1/2 * sum_j [ log(sigma_p_j^2 / sigma_q_j^2)
                         + (sigma_q_j^2 + (mu_q_j - mu_p_j)^2) / sigma_p_j^2
                         - 1 ]
```

Tính `log` của tỉ lệ bằng `2*(log sigma_p - log sigma_q)` — ổn định hơn
log một tỉ lệ khi một trong hai scale nhỏ. Code gọi
`gaussian_kl(posterior.detach(), prior).mean()`: **posterior là teacher
cố định, chỉ prior bị kéo**; gradient của KL không chạm posterior head.
(Lưu ý: `detach` chặn gradient *qua* posterior, nhưng trọng số của
posterior head vẫn được học từ `L_bal` qua `z_q` — xem mục 3.)

### Top-K router

```
r      = W_R z + b_R           [B, K]
pi_bar = Softmax(r)
K_b    = TopK(pi_bar_b, k_e)
pi     = pi_bar / sum_{j in K_b} pi_bar_j   (renormalize trên tập được chọn)
```

Điểm cần nhớ:

- Renormalize là **chia phân bố dày cho tổng trên tập được chọn**, không
  phải softmax lại top-k logits — đúng định nghĩa
  `pi_{b,k} = pi_bar_{b,k} / sum_{j in K_b} pi_bar_{b,j}` của proposal.
- Router là **instance-level**: một quyết định cho cả ảnh. Khác với
  `ExpertChoiceTokenSparseMoE` trong encoder E-SAM (MoE-FEB) — router đó
  route theo patch token; router này chọn shape expert cho cả object.
- `active_experts` bị chặn `1 <= k_e <= K`.

### Load-balance loss

```
L_bal = K * sum_k f_k * P_k
```

`f_k` là tỉ lệ *slot* active-expert cả batch gán cho expert k (chuẩn hoá
theo `B * k_e`), `P_k` là xác suất dày trung bình của expert k. Tích này
cực tiểu khi tải trải đều, nên nó chống routing collapse. Đây là balance
kiểu switch-transformer mà proposal chỉ định — **cố tình không dùng** loss
`CV^2` của ShapeMoE.

## 2. Model `phase_b_router` và `RouterStageOutput`

`PhaseBRouterStage` kế thừa `PhaseBFuseStage`: segmentation logits của
backbone đi xuyên qua không chỉnh sửa; phần định tuyến chỉ **ghi thêm**
diagnostic `phase_b_router` vào output. `PhaseBRouterHead` được giữ là
đơn vị composable (không phải `nn.Module` con của model) để test được trên
descriptor tổng hợp mà không phải dựng ViT-B.

`RouterStageOutput` gồm: `source` (`'posterior'`/`'prior'`), `prior`,
`latent` (`z` thực dùng), `routing` (`RoutingOutput`), và ba trường chỉ
tồn tại trên đường posterior: `posterior`, `latent_kl` (scalar),
`balance` (scalar).

Cờ `stochastic` (default `True`): bật thì train dùng `rsample`;
tắt là ablation deterministic `z_q = mu_q` của proposal.

Config keys mới so với fuse stage: `latent_dim: 64`, `num_experts: 4`,
`active_experts: 2`, `std_floor: 1e-4`, `stochastic: true`.

## 3. Gradient chảy đi đâu

Tại mốc này các loss bắt đầu tác động lên `h_I` — bảo chứng "trùng khớp
bit-for-bit với E3" của fuse stage **kết thúc ở đây** (hàm
`images → logits` vẫn không đổi; thay đổi là adapter và các khối dùng
chung nhận thêm gradient từ KL/L_bal qua `h_I`, nên trọng số sẽ lệch
khỏi quỹ đạo E3 ngay khi `lambda_latent`/`lambda_balance` > 0):

| Tham số | `L_seg` | `λ_latent·KL(sg[q]‖p)` | `λ_bal·L_bal` |
| --- | --- | --- | --- |
| Adapter, MoE-FEB, LPEG (image encoder) | ✓ | qua `h_I` | qua `h_I` |
| mask decoder | ✓ | — | — |
| prior head `p(z\|I)` | — | ✓ | — |
| posterior head `q(z\|I,M)` | — | — (bị `sg` chặn) | ✓ (qua `z_q`) |
| router `W_R` | — | — | ✓ |
| Shape Teacher `G_M` | — (frozen) | — (frozen) | — (frozen) |

Ghi chú: mặc dù không có gradient *qua* `h_M` về teacher, posterior head
vẫn học được cách dùng nửa `h_M` của `h_q` từ `L_bal` (giá trị forward là
hằng số, trọng số của head vẫn cập nhật). Trên đường prior (eval không mask)
chỉ còn `L_seg`; `latent_kl`/`balance` là `None`.

## 4. Task: loss và metric

`PhaseBRouterTask` kế thừa `PhaseBFuseTask` (gồm cả việc truyền mask vào
model làm privileged input):

```
L = L_seg + lambda_latent * L_latent + lambda_balance * L_bal
```

- `lambda_latent = 0.1`, `lambda_balance = 0.01` (default; cả hai phải
  ≥ 0).
- `strict_router = true` (default): model không sinh diagnostic
  `phase_b_router` thì task báo lỗi ngay thay vì lặng lẽ lui về seg thuần.
- Metric validation/test: `expert_usage_entropy` (entropy của phân bố
  dùng expert trung bình; gần `log(4) ≈ 1.386` là cân bằng, gần 0 là
  collapse), `latent_kl`, `load_balance`, kế thừa `level_weight_entropy`,
  `level_weight_max` và các metric segmentation.

Lưu ý ngữ cảnh đánh giá: ở validation/test của framework, mask **có sẵn**
nên các metric trên đo trên đường posterior với `z = mu_q` (model đang
eval → deterministic). Chỉ ở deploy thật (không mask) mới đi đường prior.

## 5. Kiểm chứng

- 27 unit/integration test trong
  [`tests/test_phase_b_router.py`](../tests/test_phase_b_router.py), chạy
  trên descriptor tổng hợp (không cần ViT-B): shape của `mu/sigma/z/pi`;
  `rsample` khớp `mu + sigma*eps` với cùng generator và khả vi theo cả
  hai head; KL bằng 0 khi hai phân bố trùng, khớp oracle
  `torch.distributions.kl_divergence`, dương và bất đối xứng; `sg[q]`
  chặn gradient vào posterior head nhưng không chặn prior head; router
  sparse đúng `k_e` phần tử khác 0, tổng 1, expert được chọn là top-k của
  `pi_bar`, gradient tới `W_R`; `L_bal` phạt collapse nặng hơn cân bằng;
  head trả đường posterior/prior đúng field; đường không mask dùng
  `mu_p` làm latent.
- Task integration (trên mock model, không cần ViT-B): `PhaseBRouterTask`
  cộng đúng KL + `L_bal` vào loss (đối chứng với `lambda = 0`), truyền
  mask đặc quyền vào model, trả đủ 11 khóa metric, mất diagnostic thì báo
  lỗi (trừ khi tắt `strict_router`), trọng số âm bị chặn; config chấp nhận
  `phase_b_router`, default và giữ nguyên `lambda_latent`/`lambda_balance`
  tường minh, chặn giá trị âm (kiểm chứng thêm trong
  [`tests/test_experiment.py`](../tests/test_experiment.py)).
- 1 test full-model (dựng backbone thật) gated sau
  `RUN_BACKBONE_TESTS=1`: xác nhận diagnostic `phase_b_router` xuất hiện
  với `source='posterior'` khi có mask và `source='prior'` khi không.

## 6. Chạy Phase B router

Model `phase_b_router` chạy được qua `run_experiment.py`:

```bash
# Kiểm tra pipeline trong 1 epoch
python scripts/run_experiment.py --config configs/phase_b/isic2018_router_smoke.yaml

# Chạy đầy đủ 50 epoch (kế thừa mọi thứ từ fuse config)
python scripts/run_experiment.py --config configs/phase_b/isic2018_router.yaml
```

`configs/phase_b/isic2018_router.yaml` kế thừa
`configs/phase_b/isic2018_fuse.yaml` (từ đó kế thừa
`configs/isic2018_common.yaml`) và chỉ thêm:

```yaml
model:
  name: phase_b_router
  latent_dim: 64            # d_z
  num_experts: 4            # K
  active_experts: 2         # k_e
  std_floor: 1.0e-4
  stochastic: true          # false = ablation z_q = mu_q

task:
  name: phase_b_router
  lambda_latent: 0.1        # KL(sg[q] || p)
  lambda_balance: 0.01      # L_bal
```

Hai trọng số loss nằm ở section `task` (không phải `model`), được validate
non-negative ngay khi load config và ghi lại vào run folder. Bỏ hai dòng
lambda để dùng default `0.1/0.01`.

Chưa có run tham chiếu — đây là việc còn thiếu duy nhất. Khi chạy smoke lần
đầu, đối chiếu:

- Dice phải **trùng fuse stage** (router chưa đụng mask decoder).
- `val_expert_usage_entropy` khởi đầu gần `log(4) ≈ 1.386` (router khởi
  tạo ngẫu nhiên nên dùng đều 4 expert), rồi dịch chuyển khi `L_bal` và
  KL tác động.
- `val_latent_kl`/`val_load_balance` khác 0.
- `h_I` bắt đầu nhận gradient — quan sát qua `val_level_weight_entropy`
  lệch dần khỏi `log(4)` sau vài epoch (điều fuse stage không bao giờ làm).

Sau đó là giai đoạn nối expert: cấp `pi`/`K_b` cho shape expert trên
`X^(l)` (hierarchical enhancement) — lúc đó output của router mới ảnh
hưởng đến segmentation.
