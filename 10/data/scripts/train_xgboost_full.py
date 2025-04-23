import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

# 读取清洗后的完整数据
df = pd.read_csv("data/student_combined_cleaned.csv")

# 特征与目标
X = df.drop(columns=["G3"])
y = df["G3"]

# 对分类特征独热编码
X = pd.get_dummies(X)

# 划分 70% 训练集
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.3, random_state=42)

# 保存训练集数据
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/xgboost_train_data_full.csv", index=False)

# 训练 XGBoost 模型
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 保存模型
model.save_model("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/models/xgboost_model_full.json")

print("XGBoost 模型训练完成，训练集数据与模型已保存。")