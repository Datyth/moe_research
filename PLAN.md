# SegMoTE Research Plan

## Mục tiêu

Tái chạy SegMoTE trên GPU NVIDIA RTX PRO 4000 Blackwell, kiểm tra cơ chế expert routing, sau đó phát triển hướng cải tiến dựa trên expert specialization.

---

## Phase 1 — Environment & Repository Setup

**Input**
- SegMoTE repository đã clone
- NVIDIA RTX PRO 4000 Blackwell
- SAM-B và SegMoTE checkpoints

**Tasks**
- Tạo môi trường Python/PyTorch
- Cài dependencies
- Kiểm tra CUDA và GPU
- Chạy import/model loading

**Output**
- Môi trường chạy ổn định
- File `environment.txt`
- Checkpoints đặt đúng thư mục

**Definition of Done**
- `torch.cuda.is_available() == True`
- Model load thành công, không lỗi dependency

**Expected Outcome**
- Repo sẵn sàng cho inference và training

---

## Phase 2 — Dataset Preparation

**Input**
- ISIC 2018 Task 1 images và masks
- Data loader của SegMoTE

**Tasks**
- Chuyển ISIC sang format repo yêu cầu
- Tạo `dataset.json`
- Chuyển masks sang `.npz`
- Kiểm tra image–mask matching
- Tạm dùng pseudo mask chỉ cho smoke test

**Output**
- `dataset/isic2018_task1/`
- Train/validation split hợp lệ
- Dataset validation script

**Definition of Done**
- Data loader đọc được toàn bộ samples
- Image, label và pseudo có shape đúng
- Không có missing hoặc corrupted files

**Expected Outcome**
- Dataset sẵn sàng để chạy baseline

---

## Phase 3 — Baseline Reproduction

**Input**
- Dataset đã chuẩn hóa
- Released SegMoTE checkpoint

**Tasks**
- Chạy inference trên validation set
- Ghi Dice, IoU và thời gian inference
- Lưu một số prediction examples
- Chạy smoke training 1–2 epochs

**Output**
- Baseline metrics
- Prediction visualizations
- Training logs và checkpoint thử nghiệm

**Definition of Done**
- `test.py` chạy hết validation set
- `train.py` chạy không OOM, NaN hoặc crash
- Kết quả được lưu có tổ chức

**Expected Outcome**
- Xác nhận pipeline SegMoTE hoạt động trên NVIDIA RTX PRO 4000 Blackwell

> Đây là code-level reproduction, chưa phải exact reproduction toàn bộ MedSeg-HQ.

---

## Phase 4 — Clean Training Baseline

**Input**
- Pipeline baseline đã chạy
- Ground-truth masks

**Tasks**
- Làm rõ cách pseudo mask được tạo trong repo
- Tạo hai cấu hình:
  1. GT-only baseline
  2. GT + pseudo baseline hợp lệ
- Fine-tune trên ISIC
- Giữ cùng cấu hình để so sánh

**Output**
- Baseline đáng tin cậy cho nghiên cứu
- Config, checkpoint và metrics
- Báo cáo ảnh hưởng của pseudo supervision

**Definition of Done**
- Không dùng `pseudo = GT` cho kết quả nghiên cứu
- Có ít nhất một baseline huấn luyện hợp lệ
- Kết quả có thể tái chạy từ config

**Expected Outcome**
- Có mốc so sánh công bằng trước khi sửa routing

---

## Phase 5 — Expert Routing Audit

**Input**
- Baseline SegMoTE ổn định
- Validation set có ground truth

**Tasks**
- Log router logits và selected expert
- Force inference với từng expert
- Tính:
  - expert usage
  - oracle expert performance
  - selected-expert regret
  - output similarity
  - routing stability
- Phân tích theo object size, boundary complexity và boundary contrast

**Output**
- File per-sample/per-expert results
- Routing analysis figures
- Evidence về expert specialization

**Definition of Done**
- Biết expert có thật sự khác nhau hay không
- Biết router có chọn expert phù hợp hay không
- Xác định bottleneck chính: router, expert redundancy hoặc shortcut

**Expected Outcome**
- Chọn đúng hướng cải tiến thay vì sửa architecture theo cảm tính

---

## Phase 6 — Proposed Improvement

**Input**
- Kết quả audit từ Phase 5

**Tasks**
- Chọn một hướng:
  - competence-aware routing
  - pattern-aware routing
  - routing stability regularization
  - expert diversity objective
- So sánh với original SegMoTE
- Chạy ablation tối thiểu

**Output**
- Improved SegMoTE model
- Baseline comparison
- Ablation table và error analysis

**Definition of Done**
- Cải thiện hard-case hoặc worst-group performance
- Không tăng quá nhiều trainable parameters
- Kết quả ổn định qua nhiều seed hoặc split

**Expected Outcome**
- Có kết quả đủ rõ để viết workshop/short paper

---

## Thứ tự ưu tiên hiện tại

1. Hoàn thành Phase 1
2. Chuẩn hóa ISIC trong Phase 2
3. Chạy checkpoint baseline ở Phase 3
4. Không phát triển method mới trước khi hoàn thành routing audit
