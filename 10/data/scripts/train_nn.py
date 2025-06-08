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
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

# 构建 params.yaml 的正确路径（向上三级）
current_dir = os.path.dirname(__file__)
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

# 加载参数
with open(params_path, encoding='utf-8') as f:
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
        self.dropout1 = nn.Dropout(p=params["dropout"])
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=params["dropout"])
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

model = Net(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

# 训练模型
writer = SummaryWriter(log_dir=tensorboard_log_dir)
epochs = params["epochs"]
losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    # 添加每层的权重直方图
    for name, param in model.named_parameters():
        if 'weight' in name:
            writer.add_histogram(f"Weights/{name}", param, epoch)
        if 'bias' in name:
            writer.add_histogram(f"Biases/{name}", param, epoch)
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f}")
writer.close()

# ----------- 可视化前20重要特征的权重 ----------

# 提取第一层权重
fc1_weights = model.fc1.weight.detach().numpy()  # [128, n_features]
mean_abs_weights = np.mean(np.abs(fc1_weights), axis=0)  # 对每个输入特征求平均绝对值

# 加载特征列
with open(feature_path, "r", encoding="utf-8") as f:
    feature_names = [line.strip() for line in f.readlines()]

# 映射权重和特征
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": mean_abs_weights
}).sort_values(by="Importance", ascending=False)

# 选前20个特征可视化
top_n = 20
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(top_n))
plt.title("Top 20 Important Features (fc1 weights)")
plt.tight_layout()
plt.savefig("results/nn_feature_importance_top20.png", dpi=300)
plt.close()

# ---------- 保存每层权重分布直方图 ----------
layers = [("fc1", model.fc1), ("fc2", model.fc2), ("fc3", model.fc3)]
for name, layer in layers:
    weights = layer.weight.detach().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(weights, bins=50, alpha=0.7)
    plt.title(f"Weight Distribution: {name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/nn_weight_hist_{name}.png", dpi=300)
    plt.close()

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