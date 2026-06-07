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

# ==========================================
# 5. 设置针对 T4 GPU 优化的训练参数
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,            # ByT5 微调通常需要稍大一些的学习率
    per_device_train_batch_size=8, # T4 上设为 8 非常安全，不易 OOM
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=15,           # 15个 Epoch 足够让 3000 条数据收敛
    predict_with_generate=True,
    fp16=True,                     # 核心：开启 T4 硬件加速的半精度训练
    logging_steps=50,
    load_best_model_at_end=True,   # 训练结束时加载验证集上表现最好的模型
    metric_for_best_model="loss",
    report_to="none"
)

# ==========================================
# 6. 开始训练
# ==========================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\n--- 开始在 T4 GPU 上训练 ---")
trainer.train()

# 保存最佳模型
trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
print("训练完成，最佳模型已成功保存！")