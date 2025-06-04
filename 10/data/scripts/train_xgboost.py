import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import yaml

# 读取参数
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["xgboost"]

# 读取清洗后的数据
df = pd.read_csv(params["data_path"])

# 特征与目标
X = df.drop(columns=[params["target"]])
y = df[params["target"]]

# 对分类特征进行独热编码
X = pd.get_dummies(X)

# 划分训练集和评估集
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

# 保存训练集数据
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/xgboost_train_data.csv", index=False)

# 训练 XGBoost 模型
model = xgb.XGBRegressor(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    learning_rate=params["learning_rate"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

# 保存模型
model.save_model(params["model_output"])

print("XGBoost 模型训练完成，训练集数据与模型已保存。")