import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载你刚才训练好的最佳模型
MODEL_PATH = "./byt5_old_english_reconstructor/best_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# 如果有 GPU，放回 GPU 上运行推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def reconstruct_old_english(english="-", german="-", dutch="-", swedish="-"):
    """
    输入现代日耳曼语言，预测古英语祖先
    """
    parts = []
    if english != "-": parts.append(f"English: {english}")
    if german != "-": parts.append(f"German: {german}")
    if dutch != "-": parts.append(f"Dutch: {dutch}")
    if swedish != "-": parts.append(f"Swedish: {swedish}")
    
    input_text = " | ".join(parts)
    
    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 生成预测
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=32, 
            num_beams=5, # 使用 Beam Search 提高重构质量
            early_stopping=True
        )
        
    # 解码输出
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# ==========================================
# 测试一些例子（你可以自己改词测试）
# ==========================================
# 测试 1: "free" 的重构
print("预测1 [free]:", reconstruct_old_english(english="free", german="frei", dutch="vrij", swedish="fri"))
# 测试 2: "now" 的重构
print("预测2 [now]:", reconstruct_old_english(english="now", german="nun", dutch="nou", swedish="nu"))
# 测试 3: 假设缺失了瑞典语和荷兰语，只给英语和德语，看能否预测出古英语的 "father"
print("预测3 [father]:", reconstruct_old_english(english="father", german="Vater"))