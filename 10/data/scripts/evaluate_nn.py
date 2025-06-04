import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yaml
import os

# 加载参数
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["nn"]

# 参数路径
data_path = params["data_path"]
model_path = params["model_output"]
feature_path = params["feature_columns_output"]
eval_data_path = params["eval_data_output"]
result_txt = params["eval_metrics_output"]
result_fig = params["eval_plot_output"]

# 加载数据
df = pd.read_csv(data_path)
X = pd.get_dummies(df.drop(columns=[params["target"]]))
y = df[params["target"]]

# 加载特征列
with open(feature_path, "r", encoding="utf-8") as f:
    feature_columns = [line.strip() for line in f.readlines()]

# 划分评估集
_, X_eval, _, y_eval = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"])
X_eval = X_eval.reindex(columns=feature_columns, fill_value=0).astype("float32")

# 保存评估集
pd.concat([X_eval, y_eval], axis=1).to_csv(eval_data_path, index=False)

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

model = Net(input_dim=len(feature_columns))
model.load_state_dict(torch.load(model_path))
model.eval()

# 转换为 tensor 并预测
X_tensor = torch.tensor(X_eval.values, dtype=torch.float32)
y_tensor = torch.tensor(y_eval.values, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

# 评估
mse = mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)
print(f"评估结果：\nMSE: {mse:.4f}\nR² : {r2:.4f}")
with open(result_txt, "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\nR² : {r2:.4f}\n")

# 可视化
plt.figure(figsize=tuple(params["plot_figsize"]))
plt.scatter(y_eval, y_pred, alpha=params["plot_alpha"])
plt.plot(params["plot_line_range"], params["plot_line_range"], 'r--')
plt.xlabel("True G3")
plt.ylabel("Predicted G3")
plt.title("NN Prediction vs True")
plt.grid(True)
plt.tight_layout()
plt.savefig(result_fig)
plt.show()