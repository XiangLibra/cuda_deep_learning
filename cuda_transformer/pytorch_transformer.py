import torch
import time

# ✅ 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 24 層的 Transformer，nhead=8
class DeepTransformer(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(DeepTransformer, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers
        )

    def forward(self, src, tgt):
        return self.transformer(src, tgt)

# ✅ 設定 Transformer 參數
d_model = 512
nhead = 8  # 8 個注意力頭
num_layers = 24  # 深度 24 層
learning_rate = 0.001  # 設定學習率
epochs = 10  # 訓練 5 個 Epoch

# ✅ 初始化模型
model = DeepTransformer(d_model, nhead, num_layers).to(device)

# ✅ 損失函數 & 優化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ✅ 創建訓練數據
seq_len = 128
batch_size = 64
start_time = time.time()
# 🚀 訓練 5 個 Epoch
for epoch in range(epochs):
    src = torch.randn(seq_len, batch_size, d_model, device=device)
    tgt = torch.randn(seq_len, batch_size, d_model, device=device)
    target = torch.randn(seq_len, batch_size, d_model, device=device)  # Ground Truth
    
    # ✅ Forward 傳播
   
    output = model(src, tgt)
    
    # ✅ 計算 Loss
    loss = criterion(output, target)
    
    # ✅ 反向傳播與更新權重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    


    # ✅ 顯示當前 Epoch 的訓練結果
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
end_time = time.time()
print(f"時間: {end_time - start_time:.6f} 秒")

print("🚀 訓練完成！")
