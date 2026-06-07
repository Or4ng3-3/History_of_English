import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载你刚才训练好的最佳模型
MODEL_PATH = "./byt5_old_english_reconstructor/checkpoint-825"
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
# 交互式用户输入主循环
# ==========================================
print("====================================================")
print("欢迎使用古英语智能重构系统 (History of English HEL Project)")
print("说明：输入现代单词。若该语言无同源词或你想模拟数据缺失，请【直接按回车】跳过。")
print("输入 'q' 或 'exit' 可退出程序。")
print("====================================================")

while True:
    print("\n--- 请输入现代日耳曼语同源词 ---")
    eng = input("1. 现代英语 (English) [默认无]: ").strip() or "-"
    if eng.lower() in ['q', 'exit']: 
        print("感谢使用，系统已退出。")
        break
        
    ger = input("2. 现代德语 (German)  [默认无]: ").strip() or "-"
    if ger.lower() in ['q', 'exit']: break
    
    dut = input("3. 现代荷兰语 (Dutch)   [默认无]: ").strip() or "-"
    if dut.lower() in ['q', 'exit']: break
    
    swe = input("4. 现代瑞典语 (Swedish) [默认无]: ").strip() or "-"
    if swe.lower() in ['q', 'exit']: break
    
    # 检查是否全部留空
    if eng == "-" and ger == "-" and dut == "-" and swe == "-":
        print("⚠️ 错误：你必须至少提供一门语言的单词！")
        continue
        
    # 运行模型重构
    try:
        predicted_oe = reconstruct_old_english(english=eng, german=ger, dutch=dut, swedish=swe)
        print(f"\n👉 AI 重构的【古英语 (Old English)】形式为: \033[1;32m{predicted_oe}\033[0m")
    except Exception as e:
        print(f"发生错误: {e}")