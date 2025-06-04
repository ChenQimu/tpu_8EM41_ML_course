import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yaml

# 读取配置
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["nn"]

# 加载数据
df = pd.read_csv(params["data_path"])
X = df.drop(columns=[params["target"]])
y = df[params["target"]]
X = pd.get_dummies(X)

# 加载特征列（确保与训练一致）
with open(params["feature_columns_output"], "r", encoding="utf-8") as f:
    feature_columns = [line.strip() for line in f.readlines()]

# 对齐特征列
X = X.reindex(columns=feature_columns, fill_value=0).astype("float32")
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# 定义模型结构
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

# 加载模型
model = Net(input_dim=len(feature_columns))
model.load_state_dict(torch.load(params["model_output"]))
model.eval()

# 预测与评估
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("【全数据评估】")
print(f"MSE: {mse:.4f}")
print(f"R² : {r2:.4f}")

# 保存评估结果
with open(params["eval_metrics_output_full"], "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R² : {r2:.4f}\n")

# 可视化预测结果
plt.figure(figsize=tuple(params["plot_figsize"]))
plt.scatter(y, y_pred, alpha=params["plot_alpha"])
plt.plot(params["plot_line_range"], params["plot_line_range"], 'r--')
plt.xlabel("True G3")
plt.ylabel("Predicted G3")
plt.title("NN Full Data Evaluation: True vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(params["eval_plot_output_full"], dpi=300)
plt.show()