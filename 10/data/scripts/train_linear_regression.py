import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import yaml

# 构建 params.yaml 的正确路径（向上三级）
current_dir = os.path.dirname(__file__)
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

# 读取参数配置
with open(params_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)['linear_regression']

# 从参数文件中读取路径
df = pd.read_csv(params['data_path'])

# 特征和目标
X = df.drop(columns=['G3'])
y = df['G3']

# 对分类特征进行独热编码
X = pd.get_dummies(X)

# 划分训练集（70%用于训练，30%用于评估，从params中读取）
X_train, _, y_train, _ = train_test_split(
    X, y,
    test_size=params['test_size'],
    random_state=params['random_state']
)

# 合并特征和目标列
train_data = pd.concat([X_train, y_train], axis=1)

# 保存训练集为csv文件
train_data.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/linear_regression_train_data.csv", index=False)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 输出线性回归模型的权重（w）
print("训练完毕，模型权重 (w):", model.coef_)

# 保存训练好的线性回归模型
joblib.dump(model, params['model_output'])  # 用参数控制输出路径

print("训练集数据已保存为 linear_regression_train_data.csv，线性回归模型已保存为 linear_regression_model.pkl。")