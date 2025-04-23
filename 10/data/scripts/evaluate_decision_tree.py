import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# 加载已训练好的决策树模型
model = joblib.load("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/models/tree_model.pkl")

# 加载处理后的数据
df = pd.read_csv("data/student_combined_cleaned.csv")
X = df.drop(columns=['G3'])
y = df['G3']

# 独热编码
X = pd.get_dummies(X)

# 特征对齐
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

# 模型预测
y_pred = model.predict(X)

# 评估指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("R²:", r2)

# 保存评估结果
with open("data/decision_tree_eval_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse}\n")
    f.write(f"R²: {r2}\n")

# 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([0, 20], [0, 20], color='red', linestyle='--')
plt.xlabel("True Values (G3)")
plt.ylabel("Predictions (G3)")
plt.title("Decision Tree: True vs Predicted Grades")

# 保存图像
plt.savefig("data/decision_tree_eval_plot.png", dpi=300)
plt.show()