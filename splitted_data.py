import pandas as pd
from datasets import Dataset

# 配置你的原始 CSV 文件路径
CSV_FILE = "germanic_parallel_dataset.csv"

# 1. 读取数据
df = pd.read_csv(CSV_FILE)

# 2. 格式化数据（与训练时完全一致）
def format_input(row):
    parts = []
    if row["Modern_English"] != "-":
        parts.append(f"English: {row['Modern_English']}")
    if row["Modern_German"] != "-":
        parts.append(f"German: {row['Modern_German']}")
    if row["Modern_Dutch"] != "-":
        parts.append(f"Dutch: {row['Modern_Dutch']}")
    if row["Modern_Swedish"] != "-":
        parts.append(f"Swedish: {row['Modern_Swedish']}")
    return " | ".join(parts)

df["input_text"] = df.apply(format_input, axis=1)
df["target_text"] = df["Target_Old_English"].astype(str)

# 3. 转换为 Hugging Face Dataset 格式并使用相同的 seed=42 进行划分
raw_dataset = Dataset.from_pandas(df[["input_text", "target_text"]])
dataset_split = raw_dataset.train_test_split(test_size=0.15, seed=42)

# 4. 使用官方标准的 .to_pandas() 导出
train_df = dataset_split["train"].to_pandas()
val_df = dataset_split["test"].to_pandas()

# 5. 保存为 CSV 文件
train_df.to_csv("actual_training_set.csv", index=False)
val_df.to_csv("actual_validation_set.csv", index=False)

print("--- 🎉 导出成功！ ---")
print(f"已生成真实的训练集：'actual_training_set.csv' （共 {len(train_df)} 条）")
print(f"已生成真实的验证集：'actual_validation_set.csv' （共 {len(val_df)} 条，模型闭卷考试的题库）")