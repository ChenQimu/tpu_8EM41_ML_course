import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
import yaml  # 用于读取 params.yaml 中的参数

# 读取参数配置
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)['decision_tree']

# 读取清理后的数据
df = pd.read_csv(params['data_path'])

# 特征和目标
X = df.drop(columns=['G3']) # 选择所有列作为特征，去除目标变量 G3
y = df['G3'] # 将 G3 作为目标变量

# 对分类特征进行独热编码
X = pd.get_dummies(X)

# 划分训练集：70%用于训练
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# 合并训练集数据并保存
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/decision_tree_train_data.csv", index=False)

# 训练决策树回归模型
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# 输出模型的权重
print("训练完毕，决策树模型已训练。")

# 确保模型保存的目录存在
if not os.path.exists('models'):
    os.makedirs('models')

# 绘制决策树的前几个节点
plt.figure(figsize=(16, 12))  # 增加图形大小
plot_tree(tree_model, feature_names=X.columns, filled=True, max_depth=3, fontsize=10)  # 调整字体大小
plt.title("Decision Tree - First Few Nodes")
plt.show()

# 保存训练好的决策树模型
joblib.dump(tree_model, params['model_output'])

print("训练集数据已保存为 student_tree_train_data.csv，决策树模型已保存为 decision_tree_model.pkl。")