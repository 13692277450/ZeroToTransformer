import json
import random

# 定义变量池
personas = ["齐天大圣孙悟空", "花果山水帘洞美猴王", "斗战胜佛"]
topics = ["天庭", "妖怪", "唐僧", "金箍棒", "如来佛祖", "取经","西游记", "孙悟空的冒险", "孙悟空", "美猴王","七十二变", "大闹天宫", "沙僧", "猪八戒"]
questions = [
    "你是怎么看待{topic}的？",
    "跟我说说{topic}的事情吧。",
    "{topic}到底是个什么东西？",
    "如果遇到{topic}，你会怎么做？"
]
answers = [
    "俺老孙当年在{topic}时，那可是威风八面！",
    "说起{topic}，俺老孙最清楚了，妖怪们见了我都得绕道走。",
    "哼，那{topic}算什么，当年大闹天宫我都不怕！",
    "师傅常教导我，对待{topic}要和善些，不过俺老孙脾气暴。"
]

def generate_data(num_samples=50000):
    dataset = []
    for _ in range(num_samples):
        persona = random.choice(personas)
        topic = random.choice(topics)
        
        # 随机组合 System, User, Bot
        system_msg = f"你正在扮演{persona}。"
        user_msg = random.choice(questions).format(topic=topic)
        bot_msg = random.choice(answers).format(topic=topic)
        
        # 构造 Alpaca 格式
        item = {
            "instruction": user_msg,
            "input": "",
            "output": bot_msg,
            "system": system_msg
        }
        dataset.append(item)
        
    # 保存为 JSON 文件
    with open("sft_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ 成功生成 {num_samples} 条数据到 sft_data.json")

# 运行
generate_data()