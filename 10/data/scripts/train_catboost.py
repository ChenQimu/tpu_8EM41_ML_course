import pandas as pd
import yaml
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# 构建 params.yaml 的正确路径（向上三级）
current_dir = os.path.dirname(__file__)
params_path = os.path.abspath(os.path.join(current_dir, '../../../params.yaml'))

# 读取配置
with open(params_path, encoding='utf-8') as f:
    params = yaml.safe_load(f)["catboost"]

# 加载清洗后的数据
df = pd.read_csv(params["data_path"])

# 转换为 category 类型（CatBoost 必需）
categorical_cols = params["categorical_features"]
df[categorical_cols] = df[categorical_cols].astype("category")

# 特征与目标
X = df.drop(columns=[params["target"]])
y = df[params["target"]]

# 划分训练集（保留70%作为训练）
X_train, _, y_train, _ = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

# 合并训练集并保存
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/catboost_train.csv", index=False)

# 构建 CatBoost 的训练 Pool
train_pool = Pool(X_train, y_train, cat_features=categorical_cols)

# 初始化 CatBoost 模型
model = CatBoostRegressor(
    iterations=params["iterations"],
    learning_rate=params["learning_rate"],
    depth=params["depth"],
    loss_function=params["loss_function"],
    random_seed=params["random_state"],
    verbose=params["verbose"]
)

# 模型训练
model.fit(train_pool)

# 确保模型保存目录存在
model_dir = os.path.dirname(params["model_output"])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 保存模型
model.save_model(params["model_output"])

print("训练完毕，CatBoost 模型已训练。")

# 获取特征重要性并保存
importances = model.get_feature_importance(prettified=True)
importances.to_csv(params["importance_output"], index=False)

# 可视化前若干特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importances["Feature Id"], importances["Importances"])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()

print("训练集数据与特征重要性已保存，模型保存至：", params["model_output"])