import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import yaml

# 获取当前脚本的路径
current_dir = os.path.dirname(__file__)
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

# 读取参数配置
with open(params_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)['linear_regression']

# 加载训练好的线性回归模型
model = joblib.load(params['model_output'])

# 读取清理后的数据
df = pd.read_csv(params['data_path'])

# 特征和目标
X = df.drop(columns=['G3'])
y = df['G3']

# 对分类特征进行独热编码，确保与训练集特征一致
X = pd.get_dummies(X)

# 对齐列：使特征与训练时模型一致（避免缺失或顺序错误）
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

# 预测全体样本
y_pred = model.predict(X)

# 模型评估
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("全数据集评估结果：")
print("MSE:", mse)
print("R² :", r2)

with open("results/linear_regression_eval_full_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

# 可视化预测结果
plt.figure(figsize=tuple(params['plot_figsize']))
plt.scatter(y, y_pred, alpha=params['plot_alpha'])
plt.plot(params['plot_line_range'], params['plot_line_range'], color='red', linestyle='--')  # 理想预测线
plt.xlabel("True Values (G3)")
plt.ylabel("Predicted Values (G3)")
plt.title("True vs Predicted Grades (Full Dataset)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/linear_regression_eval_full_plot.png", dpi=300)
plt.close()