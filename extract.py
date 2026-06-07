import json
import csv

# 我们想要收集的“子孙语言”配置
# 格式：{"语言代码": "保存到CSV的列名"}
TARGETS = {
    "ang": "Target_Old_English",  # 古英语 (我们的预测目标)
    "en": "Modern_English",       # 现代英语
    "de": "Modern_German",        # 现代德语
    "nl": "Modern_Dutch",         # 现代荷兰语
    "sv": "Modern_Swedish"        # 现代瑞典语
}

def extract_all_branches(desc_list, current_row):
    """
    递归遍历后代树，把所有目标语言的单词填入对应的列
    """
    if not desc_list:
        return
    
    for item in desc_list:
        lang_code = item.get("lang_code")
        word = item.get("word")
        
        # 如果是我们要的语言，且有单词，填入对应的列
        if lang_code in TARGETS and word:
            # 过滤掉带有 [1], [2] 这种维基词典脚注的单词
            clean_word = word.split("[")[0].strip()
            # 如果该列还没填过，或者填的是"-"，则写入
            if current_row[TARGETS[lang_code]] == "-":
                current_row[TARGETS[lang_code]] = clean_word
                
        # 递归向下寻找
        if "descendants" in item:
            extract_all_branches(item["descendants"], current_row)

# 主程序
input_file = "kaikki.org-dictionary-ProtoGermanic.jsonl"  # 你下载的原始日耳曼语包
output_file = "germanic_parallel_dataset.csv"

aligned_dataset = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
        except:
            continue
            
        # 过滤掉非词汇词条
        if data.get("pos") not in ["noun", "verb", "adj", "adv"]:
            continue
            
        pgmc_word = data.get("word") # 原始日耳曼语词，如 *stainaz
        descendants = data.get("descendants", [])
        
        if not descendants:
            continue
            
        # 初始化一行数据
        row = {val: "-" for val in TARGETS.values()}
        row["Proto_Germanic_Ancestor"] = pgmc_word
        row["Meaning"] = data.get("senses", [{}])[0].get("glosses", [""])[0]
        
        # 提取所有分支
        extract_all_branches(descendants, row)
        
        # 过滤条件：只有当【古英语】存在，且现代语言（en, de, nl, sv）中至少有2个存在时，这一行才保留
        has_oe = row["Target_Old_English"] != "-"
        modern_count = sum(1 for lang in ["Modern_English", "Modern_German", "Modern_Dutch", "Modern_Swedish"] if row[lang] != "-")
        
        if has_oe and modern_count >= 2:
            aligned_dataset.append(row)

# 写入 CSV
headers = ["Proto_Germanic_Ancestor", "Meaning", "Target_Old_English", "Modern_English", "Modern_German", "Modern_Dutch", "Modern_Swedish"]
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(aligned_dataset)

print(f"完成！成功提取了 {len(aligned_dataset)} 组多语言对齐的同源词！")