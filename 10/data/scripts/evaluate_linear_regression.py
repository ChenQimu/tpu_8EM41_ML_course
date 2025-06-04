import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yaml

# 读取参数配置
with open("params.yaml", "r") as f:
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

# 划分评估集（30%用于评估）
_, X_eval, _, y_eval = train_test_split(
    X, y,
    test_size=params['test_size'],
    random_state=params['random_state']
)

# 合并评估集的特征和目标列
eval_data = pd.concat([X_eval, y_eval], axis=1)

# 保存评估集为csv文件
eval_data.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/linear_regression_eval_data.csv", index=False)

# 使测试数据与训练数据的特征一致，填补缺失的列（如果有）
X_eval = X_eval.reindex(columns=model.feature_names_in_, fill_value=0)

# 预测评估集
y_pred = model.predict(X_eval)

# 评估模型
mse = mean_squared_error(y_eval, y_pred)
r2 = r2_score(y_eval, y_pred)

# 打印评估结果
print("评估集的MSE:", mse)
print("评估集的R²:", r2)

# 保存评估指标
with open("results/linear_regression_eval_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

# 可视化预测结果
plt.figure(figsize=tuple(params['plot_figsize']))
plt.scatter(y_eval, y_pred, alpha=params['plot_alpha'])
plt.plot(params['plot_line_range'], params['plot_line_range'], color='red', linestyle='--')  # 绘制理想的完美拟合线
plt.xlabel("True Values (G3)")
plt.ylabel("Predictions (G3)")
plt.title("True vs Predicted Grades")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/linear_regression_eval_plot.png", dpi=300)
plt.close()

print("评估集数据已保存为 linear_regression_eval_data.csv。")