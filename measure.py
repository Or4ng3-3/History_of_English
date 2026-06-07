import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm # 进度条库

# ==========================================
# 1. 纯 Python 实现的莱文斯坦编辑距离计算
# ==========================================
def edit_distance(s1, s2):
    """
    计算两个字符串之间的最小编辑距离（Levenshtein Distance）
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

# ==========================================
# 2. 环境与模型加载
# ==========================================
MODEL_PATH = "./byt5_old_english_reconstructor/best_model"
VAL_CSV = "actual_validation_set.csv"
OUTPUT_CSV = "validation_evaluation_results.csv"

print("正在加载模型和验证集数据...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

df_val = pd.read_csv(VAL_CSV)
print(f"验证集载入成功，共 {len(df_val)} 条测试词。开始批量评估...")

# ==========================================
# 3. 批量推理与编辑距离计算
# ==========================================
results = []
total_edit_distance = 0
total_norm_edit_distance = 0
perfect_matches = 0

# 统计不同编辑距离范围内的单词数量
dist_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, "4+": 0}

for idx, row in tqdm(df_val.iterrows(), total=len(df_val)):
    input_text = row["input_text"]
    target_oe = str(row["target_text"]).strip()
    
    # 模型编码与生成
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=32, num_beams=5, early_stopping=True)
    pred_oe = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 计算编辑距离
    dist = edit_distance(pred_oe, target_oe)
    norm_dist = dist / max(len(target_oe), 1) # 归一化编辑距离
    
    total_edit_distance += dist
    total_norm_edit_distance += norm_dist
    
    # 统计区间分布
    if dist == 0:
        perfect_matches += 1
    if dist <= 4:
        dist_counts[dist] += 1
    else:
        dist_counts["4+"] += 1
        
    # 保存该行结果供后续错误分析使用
    results.append({
        "input_text": input_text,
        "ground_truth_target": target_oe,
        "predicted_target": pred_oe,
        "edit_distance": dist,
        "normalized_edit_distance": round(norm_dist, 4)
    })

# ==========================================
# 4. 统计与生成报告 (对应 Ab Antiquo 论文 Table 1 格式)
# ==========================================
total_samples = len(df_val)
avg_edit_dist = total_edit_distance / total_samples
avg_norm_edit_dist = total_norm_edit_distance / total_samples

# 计算累加百分比 (即：编辑距离 <= k 的单词百分比)
acc_0 = (perfect_matches / total_samples) * 100
acc_1 = (sum(dist_counts[i] for i in range(2)) / total_samples) * 100
acc_2 = (sum(dist_counts[i] for i in range(3)) / total_samples) * 100
acc_3 = (sum(dist_counts[i] for i in range(4)) / total_samples) * 100
acc_4 = (sum(dist_counts[i] for i in range(5)) / total_samples) * 100

print("\n" + "="*50)
print("             验证集量化评估报告 (HEL Project)")
print("="*50)
print(f"测试总词条数: {total_samples}")
print(f"平均编辑距离 (Avg Edit Distance): {avg_edit_dist:.4f}")
print(f"平均归一化编辑距离 (Avg Norm Edit Dist): {avg_norm_edit_dist:.4f}")
print("-"*50)
print("编辑距离分布累加率 (Edit Distance Distribution Cumulative):")
print(f"  距离 = 0 (完全完美重构): {acc_0:.2f}%")
print(f"  距离 <= 1             : {acc_1:.2f}%")
print(f"  距离 <= 2             : {acc_2:.2f}%")
print(f"  距离 <= 3             : {acc_3:.2f}%")
print(f"  距离 <= 4             : {acc_4:.2f}%")
print("="*50)

# ==========================================
# 5. 保存详细分析结果
# ==========================================
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"评估完成！详细的对齐和对比数据已保存至：'{OUTPUT_CSV}'")
print("你可以用 Excel 打开该文件，筛选 'edit_distance > 0' 的行直接开始进行历史音变错误分析。")