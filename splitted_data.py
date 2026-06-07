import pandas as pd
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# ==========================================
# 1. 配置参数
# ==========================================
MODEL_NAME = "google/byt5-small"  # 强烈推荐的字节级/字符级模型
CSV_FILE = "germanic_parallel_dataset.csv"
OUTPUT_DIR = "./byt5_old_english_reconstructor"
MAX_INPUT_LENGTH = 128            # 输入的最大字符长度
MAX_TARGET_LENGTH = 32            # 目标输出（古英语）的最大字符长度

# ==========================================
# 2. 数据读取与格式化
# ==========================================
df = pd.read_csv(CSV_FILE)

# 将多国现代语言拼接成一个输入字符串
# 格式：English: stone | German: Stein | Dutch: steen | Swedish: sten
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

# 转换为 Hugging Face Dataset 格式并划分训练集/验证集
raw_dataset = Dataset.from_pandas(df[["input_text", "target_text"]])
dataset_split = raw_dataset.train_test_split(test_size=0.15, seed=42)
train_dataset = dataset_split["train"]
val_dataset = dataset_split["test"]

print(f"数据加载完成。训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
print("输入样例:", train_dataset[0]["input_text"])
print("目标样例:", train_dataset[0]["target_text"])

# ==========================================
# 3. 加载 Tokenizer 与模型
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ==========================================
# 4. 数据预处理（Tokenization）
# ==========================================
def preprocess_function(examples):
    # ByT5 处理的是字节，所以这里直接将字符转换为 token IDs
    model_inputs = tokenizer(
        examples["input_text"], 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True
    )
    
    # 准备目标（古英语）
    labels = tokenizer(
        text_target=examples["target_text"], 
        max_length=MAX_TARGET_LENGTH, 
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 编码数据集
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

# 动态 Padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# 将洗牌划分后的 Dataset 还原为 Pandas DataFrame
train_df = pd.DataFrame(dataset_split["train"])
val_df = pd.DataFrame(dataset_split["test"])

# 保存为新的 CSV 文件
train_df.to_csv("actual_training_set.csv", index=False)
val_df.to_csv("actual_validation_set.csv", index=False)

print("保存成功！你可以在文件栏下载 'actual_training_set.csv'（训练集）")
print("和 'actual_validation_set.csv'（验证集/测试集）进行查看。")