import os
import pandas as pd

# 所有模型的评估结果路径
eval_files = {
    "Linear Regression": "data/linear_regression_eval_result.txt",
    "Decision Tree": "data/decision_tree_eval_result.txt",
    "CatBoost": "data/catboost_eval_result.txt",
    "XGBoost": "data/xgboost_eval_result.txt",
    "Neural Network": "data/nn_eval_result.txt",
}

results = []

# 读取每个模型的评估文件
for model_name, file_path in eval_files.items():
    if not os.path.exists(file_path):
        print(f"[Warning] 文件未找到：{file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        mse = float(lines[0].split(":")[1].strip())
        r2 = float(lines[1].split(":")[1].strip())
        results.append({
            "Model": model_name,
            "MSE": mse,
            "R²": r2
        })

# 输出为表格
df = pd.DataFrame(results)
df = df.sort_values(by="R²", ascending=False)
print("\n模型评估指标对比：")
print(df.to_string(index=False))