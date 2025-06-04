import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yaml

# 加载参数
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["xgboost"]

# 加载完整数据
df = pd.read_csv(params["data_path"])

# 特征与目标
X = df.drop(columns=[params["target"]])
y = df[params["target"]]
X = pd.get_dummies(X)

# 重新补齐列（冗余健壮性）
X = X.reindex(columns=X.columns, fill_value=0)

# 加载模型
model = xgb.XGBRegressor()
model.load_model(params["model_output"])

# 全数据预测
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("【全数据评估】")
print(f"MSE: {mse}")
print(f"R² : {r2}")

# 保存评估结果
with open("results/xgboost_eval_result_full.txt", "w", encoding="utf-8") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R² : {r2:.4f}\n")

# 特征重要性图
# 获取特征重要性并绑定特征名称（使用模型内建顺序确保一致性）
booster = model.get_booster()
importance_dict = booster.get_score(importance_type='weight')

# 转换为 DataFrame 并排序（按值从小到大，便于水平条形图显示）
importance_df = pd.DataFrame({
    "Feature": list(importance_dict.keys()),
    "Importance": list(importance_dict.values())
}).sort_values(by="Importance", ascending=True)

# 限制显示前 top_n 个特征
top_n = 30
top_df = importance_df.tail(top_n)

# 绘制图形
plt.figure(figsize=(12, top_n * 0.35))
bars = plt.barh(top_df["Feature"], top_df["Importance"], color='steelblue')
plt.xlabel("Importance")
plt.title(f"Top {top_n} XGBoost Feature Importance (Full Data)")
plt.tight_layout()

# 添加数值注释
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", va='center', fontsize=8)

# 保存图像
plt.savefig(params["importance_plot_output_full"], dpi=300)
plt.show()