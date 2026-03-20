# Integral Vision Transformer



## Ý tưởng chính

Attention logits $QK^T$ trong cơ chế attention tiêu chuẩn chứa **nhiễu** (noise), làm giảm chất lượng trọng số attention. Dự án này triển khai ba chiến lược khử nhiễu:

| Mô hình | Chiến lược | Độ phức tạp |
|---------|-----------|-------------|
| **Integral Attention** | Trung bình hóa S nhóm tín hiệu logit trước softmax | $O(N^2)$ |
| **Integral-Diff Attention** | Trừ vi sai + trung bình hóa | $O(N^2)$ |
| **Taylor Integral Attention** | Tuyến tính hóa bằng Taylor kernel + trung bình hóa | $O(N)$ |

### Integral Attention

Chia mỗi head thành S nhóm tín hiệu và trung bình hóa logits của chúng:

$$\text{Attn} = \text{softmax}\!\left(\frac{1}{S}\sum_{s=1}^{S} Q_s K_s^T\right) V$$

### Integral-Diff Attention

Ghép cặp tín hiệu và trừ nhau để triệt tiêu nhiễu tương quan, sau đó trung bình hóa:

$$\text{diff}_i = Q_{2i}K_{2i}^T - \lambda_i \cdot Q_{2i+1}K_{2i+1}^T$$

$$\text{Attn} = \text{softmax}\!\left(\frac{2}{S}\sum_{i} \text{diff}_i\right) V$$

trong đó $\lambda_i$ là tham số học được theo từng head và từng cặp (khởi tạo ≈ 0.8).


## Các biến thể mô hình

| Biến thể | Embed Dim | Depth | Heads | Tham số |
|----------|-----------|-------|-------|---------|
| Tiny     | 192       | 12    | 3     | ~5M     |
| Small    | 384       | 12    | 6     | ~22M    |
| Base     | 768       | 12    | 12    | ~86M    |

## Cấu trúc dự án

```
├── models/
│   ├── integral_attention.py           # Integral Attention (O(N²))
│   ├── integral_diff_attention.py      # Integral-Diff Attention (O(N²))
│   ├── taylor_integral_attention.py    # Taylor Integral Attention (O(N))
│   ├── multihead_attention.py          # Multi-head attention tiêu chuẩn
│   ├── deit_integral_attention.py      # DeiT + IntegralAttention
│   ├── deit_integral_diff.py           # DeiT + IntegralDiffAttention
│   └── deit_linear_taylor_integral.py  # DeiT + TaylorIntegralAttention
├── configs/
│   ├── pet.yaml                        # Taylor Integral trên Oxford-IIIT Pet
│   ├── pet_integral.yaml               # Integral Attention trên Pet
│   ├── pet_integral_diff.yaml          # Integral-Diff trên Pet
│   ├── deit_linear_taylor_integral.yaml# Taylor Integral trên ImageNet-1K
│   ├── imagenet_integral_hf.yaml       # Integral Attention trên ImageNet-1K (HF)
│   └── imagenet_integral_diff_hf.yaml  # Integral-Diff trên ImageNet-1K (HF)
├── utils/
│   └── dataset.py                      # Bộ nạp dữ liệu Parquet & HuggingFace
├── train.py                            # Script huấn luyện
└── inference.py                        # Script suy luận
```

## Cách sử dụng

### Huấn luyện

```bash
# Oxford-IIIT Pet (37 lớp)
python train.py --config configs/pet_integral_diff.yaml
python train.py --config configs/pet_integral.yaml
python train.py --config configs/pet.yaml

# ImageNet-1K qua HuggingFace
python train.py --config configs/imagenet_integral_diff_hf.yaml

# ImageNet-1K qua parquet cục bộ
python train.py --config configs/deit_linear_taylor_integral.yaml
```

### Ghi đè tham số qua dòng lệnh

```bash
python train.py --config configs/pet_integral_diff.yaml \
    --variant small \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 200 \
    --output_dir ./checkpoints/my_run
```

### Tiếp tục huấn luyện từ checkpoint

```bash
python train.py --config configs/pet_integral_diff.yaml \
    --resume ./checkpoints/deit_integral_diff_pet/checkpoint_epoch_50.pth
```

## Tính năng huấn luyện

- **Mixed precision** (fp16/bf16) qua PyTorch AMP
- **Mixup & CutMix** augmentation kèm label smoothing
- **Cosine annealing** lịch trình learning rate với linear warmup
- **Stochastic depth** (DropPath) với tỉ lệ drop tăng tuyến tính
- **Lưu checkpoint** (định kỳ + mô hình tốt nhất theo accuracy validation)
- Hỗ trợ **distillation token** kiểu DeiT (tùy chọn)

## Cấu hình

### Oxford-IIIT Pet (quy mô nhỏ)

| Cài đặt | Giá trị |
|---------|---------|
| Epochs | 100 |
| Batch size | 64 |
| Learning rate | 5e-4 |
| Warmup | 5 epochs |
| Optimizer | AdamW (wd=0.05) |
| Mixup α | 0.2 |

### ImageNet-1K (quy mô lớn)

| Cài đặt | Giá trị |
|---------|---------|
| Epochs | 300 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Warmup | 5 epochs |
| Optimizer | AdamW (wd=0.05) |
| Mixup α | 0.8 |
| CutMix α | 1.0 |

## Tài liệu tham khảo

- [Integral Transformer: Denoising Attention, Not Too Much Not Too Little](https://arxiv.org/abs/2409.13560) (Kobyzev et al., 2025)
- [DeiT: Training data-efficient image transformers](https://arxiv.org/abs/2012.12877) (Touvron et al., 2021)
- [Diff Transformer](https://arxiv.org/abs/2410.05258) (Ye et al., 2024)
