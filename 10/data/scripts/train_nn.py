import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import yaml

# 加载参数
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["nn"]

data_path = params["data_path"]
model_path = params["model_output"]
feature_path = params["feature_columns_output"]
loss_curve_path = params["loss_plot_output"]
train_data_path = params["train_data_output"]
tensorboard_log_dir = params["tensorboard_log_dir"]

os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# 加载数据
df = pd.read_csv(data_path)
X = df.drop(columns=[params["target"]])
y = df[params["target"]]
X = pd.get_dummies(X)

# 划分训练集
X_train, _, y_train, _ = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"])
X_train = X_train.astype('float32')

# 保存特征列
X_train.to_csv(train_data_path, index=False)
with open(feature_path, "w", encoding="utf-8") as f:
    for col in X_train.columns:
        f.write(f"{col}\n")

# 转换为 tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# 定义模型
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = Net(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

# 训练模型
writer = SummaryWriter(log_dir=tensorboard_log_dir)
epochs = params["epochs"]
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    writer.add_scalar("Loss/train", loss.item(), epoch)
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}] Loss: {loss.item():.4f}")
writer.close()

# 保存模型
torch.save(model.state_dict(), model_path)

# 保存训练曲线
plt.figure(figsize=tuple(params["plot_figsize"]))
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_curve_path)
plt.show()