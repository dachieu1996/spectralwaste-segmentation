# FuseMoE for Semantic Segmentation

Đây là implementation của **FuseMoE** (Mixture-of-Experts) được adapt cho bài toán **semantic segmentation** trên dataset SpectralWaste.

## 📄 Paper gốc

**FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion**
- Paper: https://arxiv.org/abs/2402.03226
- Original repo: `FuseMoE/` (trong project root)

## 🏗️ Kiến trúc

```
Input (RGB/HSI) → Conv2d Projection → Reshape to Tokens → MoE Layer → Reshape to Image → Output Logits
     [B,C,H,W]         [B,D,H,W]         [B*H*W, D]      [B*H*W, K]      [B,K,H,W]
```

Trong đó:
- `B`: batch size
- `C`: input channels (3 cho RGB, 6 cho RGB+HSI)
- `H, W`: height, width
- `D`: embedding dimension (default: 64)
- `K`: num_classes (7 cho SpectralWaste)

## 📁 Files

- **`model.py`**: Wrapper class `FuseMoESegmentation` cho segmentation task
- **`sparse_moe.py`**: Core MoE implementation với sparse gating
- **`config.py`**: Configuration class `MoEConfig`
- **`activations.py`**: Activation functions (GELU, SiLU, Mish, etc.)

## 🔧 Các chỉnh sửa từ FuseMoE gốc

### 1. Adaptation cho Segmentation (model.py)

```python
# EDIT: Thêm Conv2d projection
self.input_proj = torch.nn.Conv2d(total_in_channels, embed_dim, kernel_size=1)

# EDIT: Reshape image → tokens
tokens = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)

# EDIT: Reshape tokens → image
logits = logits_tokens.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

# EDIT: Store aux loss
self.last_aux_loss = aux_loss
```

### 2. Output format (sparse_moe.py)

```python
# EDIT: Giữ outputs ở dạng logits (không apply exp/log)
# Line 106, 114, 133
self.log_soft = nn.Identity()  # Thay vì LogSoftmax
```

### 3. Import paths (sparse_moe.py)

```python
# EDIT: Relative imports
from .activations import ACT2FN
from .config import MoEConfig
```

## 🎛️ Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 64 | Embedding dimension |
| `num_experts` | 8 | Số lượng expert networks |
| `top_k` | 2 | Số experts được activate |
| `noisy_gating` | False | Có dùng noisy gating không |
| `router_type` | "joint" | Loại router: joint/permod/disjoint |
| `gating` | "softmax" | Gating function: softmax/laplace/gaussian |

## 💡 Cách sử dụng

### Trong Python code:

```python
from spectralwaste_segmentation.models.fusemoe import FuseMoESegmentation

# Tạo model
model = FuseMoESegmentation(
    in_channels=[3, 3],      # RGB + HSI
    num_classes=7,
    embed_dim=64,
    num_experts=8,
    top_k=2,
    noisy_gating=False
)

# Forward pass
rgb = torch.randn(2, 3, 256, 256)
hsi = torch.randn(2, 3, 256, 256)
output = model([rgb, hsi])  # Shape: [2, 7, 256, 256]

# Lấy auxiliary loss
aux_loss = model.last_aux_loss  # Load balancing loss
```

### Training:

```bash
# Unimodal
python -m scripts.train_model --model fusemoe --input-mode rgb

# Multimodal
python -m scripts.train_model --model fusemoe --input-mode rgb,hyper_pca3
```

## 🧪 Testing

```bash
python -m scripts.test_fusemoe
```

## 📊 Model Complexity

| Config | embed_dim | num_experts | Parameters |
|--------|-----------|-------------|------------|
| Small  | 32        | 4           | ~21K       |
| Medium | 64        | 8           | ~149K      |
| Large  | 128       | 16          | ~1.1M      |

## 🔬 MoE Components

### SparseDispatcher
- Phân phối inputs đến các experts dựa trên gates
- Chỉ activate top-k experts cho mỗi token
- Combine outputs với weighted sum

### MLP Expert
- 2-layer feedforward network
- Hidden size = 4 × embed_dim
- Activation: GELU (default)
- Dropout: 0.1

### Gating Network
- Learnable weight matrix: `w_gate` [embed_dim, num_experts]
- Optional noisy gating với `w_noise`
- Top-k selection với softmax

## ⚠️ Lưu ý

1. **Memory usage**: MoE xử lý mỗi pixel như một token → memory tăng với image size
2. **Auxiliary loss**: Cần thêm `model.model.last_aux_loss` vào training loss để balance experts
3. **Router types**: 
   - `joint`: Single router cho tất cả modalities (hiện tại)
   - `permod`: Separate router cho mỗi modality
   - `disjoint`: Disjoint experts cho mỗi modality

## 📚 References

```bibtex
@article{wang2024fusemoe,
  title={FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion},
  author={Wang, Xing Han and others},
  journal={arXiv preprint arXiv:2402.03226},
  year={2024}
}
```

