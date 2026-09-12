# Phase B — expert enhancement: nối expert và tiêm vào decoder

Tài liệu này tiếp nối [`docs/phase_b_router.md`](phase_b_router.md). Routing
latent `z` giờ đây không chỉ ghi nhận trong diagnostics mà **điều khiển
các expert** tinh chỉnh token nhiều tầng (`X^(l)`), kết quả được nén qua
neck và **tiêm residually vào SAM decoder**. Đây là mốc đầu tiên mà
loss segmentation backpropagate vào expert và `g_layer` — và là lần đầu
hàm `images → logits` của Phase B **cố ý tách khỏi E3**.

```
                      g_layer([v^(l) ; z ; e_k])  (Eqs. 40-44)
   X^(l) ─────────┐        │
   (l ∈ {3,6,9,12})│        ▼  β_{b,k,l}  (softmax trên levels)
        │         │        │
        │    E_k(X^(l))    │            π_{b,k}, K_b (từ router stage)
        │         │        │                │
        ▼         ▼        ▼                ▼
   X̂^(l) = X^(l) + Σ_{k∈K_b} π_{b,k} β_{b,k,l} E_k(X^(l))      (Eq. 47)
        │
        │   γ_{b,l} = Σ_{k∈K_b} π_{b,k} β_{b,k,l}                (Eqs. 48/49)
        ▼
   Z_fused = Σ_l γ_{b,l} X̂^(l)                                  (Eq. 50)
        │
        ▼  Neck (Eq. 53): [B, P, 768] → [B, 256, 16, 16]
   E_aux
        │
E_SAM ──┴─► E_enh = E_SAM + E_aux                               (Eq. 54)
        │
        ▼  LPEG prompt + D_SAM                                   (Eqs. 55-56)
       M̂
```

Hai thứ được giữ nguyên từ hai giai đoạn trước: SAM backbone vẫn đóng băng
(expert **đọc** `X^(l)` nhưng không ghi lại vào encoder), và đường inference
deployable (không mask → prior mean `μ_p` điều khiển; `β` và expert chỉ cần
đầu vào phía ảnh — `v^(l)`, `z`, `e_k` đều image-side).

## 1. Thành phần

| Ký hiệu | Shape | Module |
| --- | --- | --- |
| `E_k(X)` | `[B, P, 768]` → `[B, P, 768]` | `FeedForwardExpert` (Eq. 45): LN → 768→3072 → GELU → 3072→768 |
| `v^(l)` | `[B, 768]` | `GAP(X^(l))` — **raw**, không qua `P_l` (khác `u^(l)` của descriptor) |
| `e_k` | `[768]` | `nn.Parameter` riêng trong scorer (Eq. 41), khởi tạo ×0.02 |
| `β_{b,k,l}` | `[B, k_e, L]` | `LayerPreferenceScorer`: softmax trên levels (Eqs. 40-44) |
| `X̂^(l)` | `[B, P, 768]` | Eq. 47 — token đã enhance của level l |
| `γ_{b,l}` | `[B, L]` | Eq. 48/49 — tổng bằng 1 trên levels (hệ quả π, β chuẩn hoá) |
| `Z_fused` | `[B, P, 768]` | Eq. 50 — token đã fuse nhiều level |
| `E_aux` | `[B, 256, 16, 16]` | Neck (Eq. 53) |
| `E_enh` | `[B, 256, 16, 16]` | `E_SAM + E_aux` (Eq. 54) |

Code tương ứng:

| File | Vai trò |
| --- | --- |
| [`src/models/phase_b/experts.py`](../src/models/phase_b/experts.py) | `FeedForwardExpert` (Eq. 45); `ExpertBank` (K expert) |
| [`src/models/phase_b/layer_attention.py`](../src/models/phase_b/layer_attention.py) | `LayerPreferenceScorer` = `g_layer` (Eqs. 40-44); owns `e_k` |
| [`src/models/phase_b/moe_enhancement.py`](../src/models/phase_b/moe_enhancement.py) | `HierarchicalMoEEnhancement` (Eqs. 47-50) — sparse dispatch |
| [`src/models/phase_b/phase_b_moe.py`](../src/models/phase_b/phase_b_moe.py) | `PhaseBMoEStage`, model `phase_b_moe`: neck + residual injection + decoder (Eqs. 53-56) |
| [`src/tasks/phase_b_moe.py`](../src/tasks/phase_b_moe.py) | `PhaseBMoETask`: task `phase_b_moe` — strict check + metric |

## 2. Các quyết định thiết kế

**`g_layer` khác `g_level` ở câu hỏi being asked.** `g_level` (fuse stage)
hỏi *"level l liên quan đến ảnh này bao nhiêu?"* — một trọng số mỗi level,
dùng chung cho mọi expert, sinh `α_l`. `g_layer` hỏi *"level l liên quan đến
**expert k** trên **mẫu này** bao nhiêu?"* — sinh `β_{b,k,l}`. Cùng một mẫu
điều hướng tới expert {0, 3} có thể rút shape-evidence từ level khác nhau
cho từng expert — texture sớm cho một expert, semantics muộn cho expert kia.
Đó là phần "hierarchical": việc chọn level được uỷ quyền cho từng expert,
không cố định toàn cục.

**`v^(l)` là pooling raw, không tái dùng `u^(l)`.** Descriptor fuse stage
đã có `u^(l) = GAP(P_l(F^(l)))` [B, 256]. `g_layer` dùng
`v^(l) = GAP(X^(l))` [B, 768] — không qua `P_l` — để `β` độc lập với đường
`α` của descriptor. Hai luồng giữ tách biệt; tái dùng `u^(l)` rẻ hơn một
projection nhưng sẽ buộc level-preference của expert đi qua cùng biến đổi
mà `h_I` dùng.

**Sparse dispatch thật — không loop trên K.** Sample được nhóm theo expert
được route trong batch; mỗi expert chạy **một** call batched trên các
(sample, slot) pair gán cho nó — gather-then-run chuẩn sparse-MoE. Hệ quả:
expert ngoài `K_b` (2/4 với `active_experts=2`) không tốn compute **và**
không nhận gradient — được chứng minh trực tiếp bằng
`test_unrouted_experts_get_no_gradient`.

**`e_k` là `nn.Parameter` riêng của scorer.** Bản nháp đầu lấy e_k từ trung
bình cột `fc1.weight` của expert (tránh bảng embedding riêng) nhưng điều đó
lẫn lộn hai vai trò: danh tính dùng để *score* và biến đổi dùng để
*transform*. Refactor thành `nn.Parameter` [K, C] riêng trong
`LayerPreferenceScorer`, khởi tạo ×0.02 để lúc đầu `β` theo data terms.

**Neck mirror `neck5` của MoE-FEB.** Token-level MoE trong encoder đã dùng
đúng pattern này: tokens → Conv1×1 768→256 → LN → Conv3×3 → LN → residual
add vào `image_embeddings` (`sam_moe.py`, dòng `image_embeddings +
neck5(...)`). Giai đoạn này là consumer **thứ hai** của cùng injection
point — instance-routed, level-aware, shape-conditioned. Mask decoder
hoàn toàn không đụng đến (khác ShapeMoE duplicate hypernetwork decoder-side;
xem mục 2 của [`phase_b_router.md`](phase_b_router.md) vì sao bản proposal
chọn encoder-side).

**Forward gọi network theo từng giai đoạn.** `PhaseBMoEStage.forward` không
gọi `backbone.network(...)` nguyên thứ (sẽ chạy decoder trước khi kịp tiêm
`E_aux`): nó tự chạy `image_encoder` → MoE-FEB path (nguyên văn từng dòng
vendored code) → descriptor/fuse/router → enhancement → neck →
`prompt_encoder` + `mask_decoder`. Phần chung với E3 giữ nguyên đúng từng
dòng nên chỉ khác biệt nằm ở `E_aux`.

## 3. Gradient chảy đi đâu

Bảng này tiếp nối mục 3 của
[`docs/phase_b_router.md`](phase_b_router.md) — cột mới là `L_seg` **qua
enhancement**:

| Tham số | `L_seg` (qua E_aux) | `λ_latent·KL` | `λ_bal·L_bal` |
| --- | --- | --- | --- |
| shape expert `E_k` (k ∈ K_b) | ✓ | — | — |
| `g_layer` + `e_k` | ✓ | — | — |
| prior head `p(z\|I)` | ✓ (qua `z = μ_p` trên đường prior / `z_q` train) | ✓ | — |
| posterior head, router `W_R` | ✓ (chỉ đường train, qua `z_q`) | — | ✓ |
| `P_l`, `g_level` (h_I) | ✓ (qua `h_I` → posterior/prior) | ✓ | ✓ |
| Adapter, MoE-FEB, LPEG | ✓ | qua `h_I` | qua `h_I` |
| Shape Teacher `G_M` | — (frozen) | — | — |

Đây là câu trả lời cho "lần đầu `h_I` nhận gradient" của router doc: tại
router stage chỉ KL và `L_bal` chạm `h_I`; từ giai đoạn này `L_seg` cũng
chảy về `h_I` qua chuỗi
`logits ← E_enh ← E_aux ← Z_fused ← X̂^(l) ← π/β ← z ← h_I`.

Trên đường **prior** (deploy, không mask): `z = μ_p` vẫn điều khiển
enhancement đầy đủ — expert, `g_layer` nhận gradient từ `L_seg` khi finetune
trên đường này (không có trong config hiện tại; mọi config của framework
đi đường posterior vì mask luôn có ở train/val/test).

## 4. Chạy giai đoạn này

```bash
# Kiểm tra pipeline trong 1 epoch
python scripts/run_experiment.py --config configs/phase_b/isic2018_moe_smoke.yaml

# Chạy đầy đủ 50 epoch (kế thừa mọi thứ từ fuse config)
python scripts/run_experiment.py --config configs/phase_b/isic2018_moe.yaml
```

`configs/phase_b/isic2018_moe.yaml` kế thừa
`configs/phase_b/isic2018_fuse.yaml` (từ đó kế thừa
`configs/isic2018_common.yaml`) và thêm:

```yaml
model:
  name: phase_b_moe
  latent_dim: 64            # d_z (kế thừa router stage)
  num_experts: 4            # K
  active_experts: 2         # k_e
  stochastic: true          # false = ablation z_q = mu_q
  expert_hidden_ratio: 4     # 768 → 3072 → 768

task:
  name: phase_b_moe
  lambda_latent: 0.1        # KL(sg[q] || p)
  lambda_balance: 0.01      # L_bal
```

Task không thêm loss mới ngoài hai routing loss của router stage —
enhancement nằm trong forward path của model nên `L_seg` **tự** backpropagate
qua expert. Task thêm strict check diagnostic `phase_b_moe` (mất thì báo lỗi
thay vì lặng lẽ lui về router stage) và hai metric:

| Metric | Ý nghĩa |
| --- | --- |
| `enhancement_aux_ratio` | mean ‖E_aux‖/‖E_enh‖ — 0 nghĩa là enhancement chết hoặc neck zero |
| `fused_level_weight_entropy` | entropy của `γ` — gần `log 4 ≈ 1.386` là đều, gần 0 là collapse |

kế thừa toàn bộ metric của router/fuse stage (`expert_usage_entropy`,
`latent_kl`, `load_balance`, `level_weight_entropy` — chú ý `α` từ
`g_level` và `γ` từ `g_layer` là **hai đại lượng khác nhau**, đều được báo).

Chưa có run tham chiếu — khi chạy smoke lần đầu, đối chiếu:

- Dice **không** còn trùng router stage (E_aux chủ ý nhiễu decoder); theo
  kỳ vọng khởi tạo thì `enhancement_aux_ratio` nhỏ và Dice gần baseline,
  sai lệch lớn nghĩa là neck/expert khởi tạo xấu.
- `val_fused_level_weight_entropy` khởi đầu gần `log(4)` (β khởi tạo theo
  data terms nhỏ) rồi dịch chuyển khi expert bắt đầu phân hoá.
- Gradient check: trọng số expert thay đổi sau epoch 1 (khác router stage
  nơi router chỉ học từ KL/L_bal).

## 5. Kiểm chứng

- 18 unit test trong [`tests/test_phase_b_moe.py`](../tests/test_phase_b_moe.py),
  chạy trên tensor tổng hợp (không cần ViT-B): shape của `β`/`γ`/`Z_fused`/
  `E_k(X)`; `β` và `γ` là softmax (tổng 1); **identity khi expert output
  zero** — `X̂^(l) = X^(l)` và `Z_fused` đúng bằng tổng γ-weighted của token
  gốc; gradient đến expert, `g_layer`, `e_k`; **không gradient cho expert
  ngoài `K_b`**; các error contract về shape sai.
- 3 backbone test (gated sau `RUN_BACKBONE_TESTS=1`, dựng ViT-B thật):
  - forward đủ chain: logits thay đổi, `aux_norm_ratio > 0`, shape đúng
    `[B, 1, 256, 256]`;
  - gradient đến enhancement nhưng ViT block đóng băng vẫn không
    `requires_grad` (chỉ Adapter được train);
  - inference không mask: vẫn đủ enhancement, router đi đường prior
    (`source='prior'`).
- Toàn bộ suite: 167 test pass (18 mới + 3 backbone của giai đoạn này).

## 6. Chưa làm

- **Routing distillation**: KL hiện chưng cất latent `z` (posterior →
  prior); chưng cất **phân phối routing** (`π̄_q` → `π̄_p`) là bước kế —
  `RoutingOutput.dense_probs` đã được trả chính xác để serving bước này.
- **Phase C**: mọi thứ trong proposal sau mốc này.
