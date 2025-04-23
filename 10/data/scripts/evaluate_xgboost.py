import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取清洗后的完整数据
df = pd.read_csv("data/student_combined_cleaned.csv")

# 特征与目标
X = df.drop(columns=["G3"])
y = df["G3"]
X = pd.get_dummies(X)

# 使用同样的划分方式获取 30% 评估集
_, X_eval, _, y_eval = train_test_split(X, y, test_size=0.3, random_state=42)
X_eval = X_eval.reindex(columns=X.columns, fill_value=0)

# 保存评估集数据
eval_df = pd.concat([X_eval, y_eval], axis=1)
eval_df.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/xgboost_eval_data_full.csv", index=False)

# 加载模型
model = xgb.XGBRegressor()
model.load_model("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/models/xgboost_model_full.json")

# 模型评估
y_pred = model.predict(X_eval)
mse = mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)

print("XGBoost 评估结果：")
print("MSE:", mse)
print("R² :", r2)

# 保存评估结果到 txt 文件
with open("data/xgboost_eval_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

# 特征重要性图
importances = model.feature_importances_
features = X.columns

# 构造 DataFrame 并按重要性排序
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(12, max(6, len(features) * 0.25)))  # 自动拉高图形防止拥挤
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/xgboost_feature_importance_full.png", dpi=300)
plt.show()