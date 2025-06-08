import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 获取当前脚本目录
current_dir = os.path.dirname(__file__)
# 构造 params.yaml 的绝对路径（项目根目录）
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

# 读取配置文件
with open(params_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)["catboost"]

# 加载训练好的 CatBoost 模型
model = CatBoostRegressor()
model.load_model(params["model_output"])

# 加载清理后的数据
df = pd.read_csv(params["data_path"])

# 转换分类特征为 category 类型
categorical_cols = params["categorical_features"]
df[categorical_cols] = df[categorical_cols].astype("category")

# 特征与目标
X = df.drop(columns=[params["target"]])
y = df[params["target"]]

# 划分评估集（30%）
_, X_eval, _, y_eval = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"])

# 合并评估数据并保存
eval_data = pd.concat([X_eval, y_eval], axis=1)
eval_data.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/catboost_eval_data.csv", index=False)

# 预测
y_pred = model.predict(X_eval)

# 评估指标
mse = mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)

# 打印评估结果
print("评估集的 MSE:", mse)
print("评估集的 R²:", r2)

# 确保结果保存目录存在
if not os.path.exists("results"):
    os.makedirs("results")

# 保存评估结果为文本文件
with open("results/catboost_eval_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R² : {r2:.4f}\n")

# 可视化预测结果
plt.figure(figsize=tuple(params["plot_figsize"]))
plt.scatter(y_eval, y_pred, alpha=params["plot_alpha"])
plt.plot(params["plot_line_range"], params["plot_line_range"], color='red', linestyle='--')
plt.xlabel("True Values (G3)")
plt.ylabel("Predictions (G3)")
plt.title("CatBoost - True vs Predicted Grades")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost_eval_plot.png", dpi=300)
plt.close()

print("评估集数据已保存为 catboost_eval_data.csv，结果图与指标已输出至 results 文件夹。")