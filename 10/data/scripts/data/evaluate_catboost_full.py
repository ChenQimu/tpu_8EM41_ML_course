import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取清洗后的完整数据
df = pd.read_csv("data/student_combined_cleaned.csv")

# 特征和目标
X = df.drop(columns=["G3"])
y = df["G3"]
cat_features = X.select_dtypes(include="category").columns.tolist()

# 划分评估集
_, X_eval, _, y_eval = train_test_split(X, y, test_size=0.3, random_state=42)

# 保存评估集数据
eval_df = pd.concat([X_eval, y_eval], axis=1)
eval_df.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/catboost_eval_data_full.csv", index=False)

# 加载模型
model = CatBoostRegressor()
model.load_model("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/models/catboost_model_full.cbm")

# 预测
y_pred = model.predict(X_eval)

# 评估
mse = mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)

print(f"\n评估结果：\nMSE: {mse:.4f}\nR² : {r2:.4f}")

# 保存评估结果
with open("data/catboost_eval_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

# 可视化真实 vs 预测
plt.figure(figsize=(8, 6))
plt.scatter(y_eval, y_pred, alpha=0.6)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel("True G3")
plt.ylabel("Predicted G3")
plt.title("CatBoost Prediction vs True (Full Features)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 获取特征重要性
importances = model.get_feature_importance(prettified=True)

# 保存为 CSV
importances.to_csv("data/catboost_feature_importance_eval.csv", index=False)

# 可视化并保存图像
plt.figure(figsize=(10, 6))
plt.barh(importances["Feature Id"], importances["Importances"])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance (Eval)")
plt.tight_layout()
plt.savefig("data/catboost_feature_importance_eval.png", dpi=300)