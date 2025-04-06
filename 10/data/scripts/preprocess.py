import pandas as pd

# 读取数据
df_math = pd.read_csv(r"E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/student-mat.csv", sep=",")
df_por = pd.read_csv(r"E:/TPU_Work/jidian_shujufenxi/tpu_8EM41_ML_course/10/data/raw/student-por.csv", sep=",")

# 添加 class 列区分数据来源
df_math["class"] = "math"
df_por["class"] = "por"

# 合并数据集
df_combined = pd.concat([df_math, df_por], ignore_index=True)

# 处理分类变量
categorical_cols = df_combined.select_dtypes(include=['object']).columns
df_combined[categorical_cols] = df_combined[categorical_cols].apply(lambda x: x.str.strip())

# 筛选出数值类型的列
numeric_cols = df_combined.select_dtypes(include=['number']).columns

# 填充数值列的缺失值
df_combined[numeric_cols] = df_combined[numeric_cols].fillna(df_combined[numeric_cols].median())

# 对分类列使用众数进行填充
categorical_cols = df_combined.select_dtypes(include=['object']).columns
df_combined[categorical_cols] = df_combined[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# 保存清理后的数据
df_combined.to_csv("data/student_combined_cleaned.csv", index=False)