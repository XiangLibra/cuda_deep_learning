import torch
import time
from torch.utils.cpp_extension import load

# 🔥 載入 CUDA 核心
cuda_attention = load(name="cuda_attention_deep", sources=["cuda_attention_deep.cu"], verbose=True)

# ✅ 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Transformer Attention 參數
batch_size = 64
seq_len = 128
d_model = 512
nhead = 8
num_layers = 24  
learning_rate = 0.01

# ✅ 創建數據
Q = torch.randn(batch_size, seq_len, d_model, device=device)
K = torch.randn(batch_size, seq_len, d_model, device=device)
V = torch.randn(batch_size, seq_len, d_model, device=device)

# ✅ 目標輸出 (模擬 ground truth)
target = torch.randn(batch_size, nhead, seq_len, seq_len, device=device)

# ✅ 初始化權重
weights = torch.randn(d_model, d_model, device=device, requires_grad=True)

# 🚀 訓練 5 Epoch
start_time = time.time()
output = cuda_attention.attention_cuda_train(Q, K, V, target, weights, num_layers, nhead, learning_rate)
end_time = time.time()

print(f"🚀 訓練完成！總時間: {end_time - start_time:.6f} 秒")
