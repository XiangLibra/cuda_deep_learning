import torch
import time

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立簡單的神經網路層（線性變換）
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# 初始化模型並搬移到 GPU
input_size = 1024
output_size = 512
model = SimpleNN(input_size, output_size).to(device)

# 測試輸入
x = torch.randn(100000, input_size, device=device)

# 計時
start_time = time.time()
y = model(x)  # 直接使用 PyTorch 的 CUDA 加速
end_time = time.time()

print(f"使用 PyTorch 內建 CUDA 加速的運行時間: {end_time - start_time:.6f} 秒")
