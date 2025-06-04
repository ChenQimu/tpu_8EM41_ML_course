import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yaml

# 读取参数
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["xgboost"]

# 加载数据
df = pd.read_csv(params["data_path"])

# 特征与目标
X = df.drop(columns=[params["target"]])
y = df[params["target"]]
X = pd.get_dummies(X)

# 获取 30% 评估集
_, X_eval, _, y_eval = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)
X_eval = X_eval.reindex(columns=X.columns, fill_value=0)

# 保存评估数据
eval_df = pd.concat([X_eval, y_eval], axis=1)
eval_df.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/xgboost_eval_data.csv", index=False)

# 加载模型
model = xgb.XGBRegressor()
model.load_model(params["model_output"])

# 模型评估
y_pred = model.predict(X_eval)
mse = mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)

print("XGBoost 评估结果：")
print("MSE:", mse)
print("R² :", r2)

# 保存评估结果
with open("results/xgboost_eval_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R² : {r2:.4f}\n")

# 特征重要性图
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

# 设置更高图像高度与更小字体，限制显示前若干重要特征（如前30）
top_n = 30
top_df = importance_df.tail(top_n)

plt.figure(figsize=(12, top_n * 0.35))  # 高度动态调整
bars = plt.barh(top_df["Feature"], top_df["Importance"], color='steelblue')
plt.xlabel("Importance")
plt.title(f"Top {top_n} XGBoost Feature Importance")
plt.tight_layout()

# 添加数值标注
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=8)

plt.savefig(params["importance_plot_output"], dpi=300)
plt.show()