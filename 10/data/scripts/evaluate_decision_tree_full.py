import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yaml
import os

# 获取当前脚本目录
current_dir = os.path.dirname(__file__)
# 构造 params.yaml 的绝对路径（项目根目录）
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

# 读取参数配置
with open(params_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)['decision_tree']

# 加载训练好的决策树模型
model = joblib.load(params['model_output'])

# 加载清洗后的数据
df = pd.read_csv(params['data_path'])

# 特征与目标分离
X = df.drop(columns=['G3'])
y = df['G3']

# 分类变量进行独热编码
X = pd.get_dummies(X)

# 特征对齐：确保与训练集一致（填补缺失列）
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

# 模型预测
y_pred = model.predict(X)

#  模型评估
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("全数据集评估结果：")
print("MSE:", mse)
print("R² :", r2)

#  保存评估结果为文本文件
with open("results/decision_tree_eval_full_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R² : {r2:.4f}\n")

#  绘图可视化
plt.figure(figsize=tuple(params['plot_figsize']))
plt.scatter(y, y_pred, alpha=params['plot_alpha'])
plt.plot(params['plot_line_range'], params['plot_line_range'], color='red', linestyle='--')
plt.xlabel("True Values (G3)")
plt.ylabel("Predicted Values (G3)")
plt.title("True vs Predicted Grades (Full Dataset)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/decision_tree_eval_full_plot.png", dpi=300)  # 保存图像
plt.close()

print("全数据集评估结果已保存为文本和图像文件。")