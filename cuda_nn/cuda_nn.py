import torch
import time
from torch.utils.cpp_extension import load

# 載入 CUDA 擴展
cuda_kernel = load(name="cuda_kernel", sources=["cuda_kernel.cu"], verbose=True)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化輸入
input_size = 1024
output_size = 512
x = torch.randn(100000, input_size, device=device)
w = torch.randn(input_size, output_size, device=device)

# 計時
start_time = time.time()
y = cuda_kernel.matmul_cuda(x, w)  # 呼叫我們的 CUDA 核心函數
end_time = time.time()

print(f"使用自行編寫 CUDA 核心的運行時間: {end_time - start_time:.6f} 秒")
