import torch
import time
from torch.utils.cpp_extension import load

# 🔥 載入 CUDA Tensor Core 核心
cuda_attention_tc = load(
    name="cuda_attention_tensorcore",
    sources=["cuda_attention_tensorcore.cu"],
    verbose=True
)

# ✅ 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Transformer 參數設定
batch_size = 64
seq_len = 128
d_model = 512  # 確保是 16 的倍數，如 256, 512, 1024
nhead = 8
num_layers = 24

# ✅ 創建測試數據
Q = torch.randn(batch_size, seq_len, d_model, device=device).to(torch.half).contiguous()
K = torch.randn(batch_size, seq_len, d_model, device=device).to(torch.half).contiguous()
V = torch.randn(batch_size, seq_len, d_model, device=device).to(torch.half).contiguous()

# 🚀 開始計時並運行 CUDA Tensor Core Attention
start_time = time.time()
output = cuda_attention_tc.attention_tensor_core(Q, K, V, num_layers, nhead)
end_time = time.time()

print(f"🚀 使用 Tensor Cores CUDA 多頭 Attention (24 層, 8 Heads) 運行時間: {end_time - start_time:.6f} 秒")
