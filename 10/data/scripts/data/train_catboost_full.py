import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# 读取清洗后的完整数据
df = pd.read_csv("data/student_combined_cleaned.csv")

# 明确指定哪些列是类别特征
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                    'nursery', 'higher', 'internet', 'romantic', 'class']

# 把这些列转换为 category 类型（否则 CatBoost 会报错）
df[categorical_cols] = df[categorical_cols].astype("category")

# 特征和目标
X = df.drop(columns=["G3"])
y = df["G3"]
cat_features = categorical_cols  # 显式告诉 CatBoost 哪些是 categorical

# 划分训练集
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# 保存训练集数据
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/catboost_train_data_full.csv", index=False)

# 创建 Pool 对象
train_pool = Pool(X_train, y_train, cat_features=cat_features)

# 初始化 CatBoost 模型
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

# 训练模型
model.fit(train_pool)

# 保存模型
model.save_model("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/models/catboost_model_full.cbm")

# 特征重要性可视化与保存
importances = model.get_feature_importance(prettified=True)
importances.to_csv("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/catboost_feature_importance_full.csv", index=False)

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature Id'], importances['Importances'])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.savefig("E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/catboost_feature_importance_full.png")
plt.show()

print("训练完成并保存模型和训练集数据。")