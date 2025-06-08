import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import yaml

# 获取当前脚本目录
current_dir = os.path.dirname(__file__)
# 构造 params.yaml 的绝对路径（项目根目录）
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

with open(params_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)['decision_tree']

# 加载训练好的决策树模型
model = joblib.load(params['model_output'])

# 读取清理后的数据
df = pd.read_csv(params['data_path'])

# 特征和目标
X = df.drop(columns=['G3'])
y = df['G3']

# 对分类特征进行独热编码
X = pd.get_dummies(X)

# 划分评估集：30%用于评估
_, X_eval, _, y_eval = train_test_split(X, y, test_size=0.3, random_state=42)

# 合并评估集数据并保存
eval_data = pd.concat([X_eval, y_eval], axis=1)
eval_data.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/decision_tree_eval_data.csv", index=False)

# 使评估集特征与训练集一致
X_eval = X_eval.reindex(columns=model.feature_names_in_, fill_value=0)

# 预测评估集
y_pred = model.predict(X_eval)

# 评估模型
mse = mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)

# 打印评估结果
print("评估集的MSE:", mse)
print("评估集的R²:", r2)

#  保存评估结果为文本文件
with open("results/decision_tree_eval_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R² : {r2:.4f}\n")

# 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(y_eval, y_pred, alpha=0.6)
plt.plot([0, 20], [0, 20], color='red', linestyle='--')  # 绘制理想的完美拟合线
plt.xlabel("True Values (G3)")
plt.ylabel("Predictions (G3)")
plt.title("True vs Predicted Grades")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/decision_tree_eval_plot.png", dpi=300)  # 保存图像
plt.close()

print("评估集数据已保存为 decision_tree_eval_data.csv。")