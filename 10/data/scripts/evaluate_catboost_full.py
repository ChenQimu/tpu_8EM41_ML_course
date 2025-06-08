import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 获取当前脚本的路径
current_dir = os.path.dirname(__file__)
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

# 读取配置文件
with open(params_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)["catboost"]

# 加载 CatBoost 模型
model = CatBoostRegressor()
model.load_model(params["model_output"])

# 加载完整数据
df = pd.read_csv(params["data_path"])

# 转换分类特征为 category 类型
categorical_cols = params["categorical_features"]
df[categorical_cols] = df[categorical_cols].astype("category")

# 特征和目标
X = df.drop(columns=[params["target"]])
y = df[params["target"]]

# 预测
y_pred = model.predict(X)

# 计算评估指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 打印结果
print("【全数据评估】")
print("MSE:", mse)
print("R² :", r2)

# 确保结果目录存在
if not os.path.exists("results"):
    os.makedirs("results")

# 保存评估结果文本
with open("results/catboost_eval_result_full.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R² : {r2:.4f}\n")

# 可视化
plt.figure(figsize=tuple(params["plot_figsize"]))
plt.scatter(y, y_pred, alpha=params["plot_alpha"])
plt.plot(params["plot_line_range"], params["plot_line_range"], color='red', linestyle='--')
plt.xlabel("True Values (G3)")
plt.ylabel("Predictions (G3)")
plt.title("CatBoost (Full Data) - True vs Predicted Grades")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost_eval_plot_full.png", dpi=300)
plt.close()

print("使用全数据完成评估，已保存 CSV、评估指标和图像。")