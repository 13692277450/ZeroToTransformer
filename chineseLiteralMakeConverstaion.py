import re
import json
from pathlib import Path

# =========配置=========

# system提示：古风问答人设
SYSTEM_PROMPT = "你熟读红楼梦,西游记,水浒传,三国演义,聊斋志异,喻世明言,警世通言,醒世恒言,初刻拍案惊奇，二刻拍案惊奇，回答尽量贴合原著，语言偏古风简洁，依据书中内容作答。"



INPUT_FILE = "allChineseLiteral.txt"
OUTPUT_FILE = "allChineseLiteral_sft_custom.json"

QUESTION_TEMPLATES = [
    "请简述{}这段故事的经过？",
    "{}这个情节发生的背景是什么？",
    "在{}一回中，人物都做了哪些事？",
    "书中{}一事的结果是什么？",
    "{}事件里面主要出场人物有谁？",
    "请解释《三国演义》中{}的故事情节。",
]

def clean_raw_text(text: str) -> str:
    patterns = [
        r"← 阿星星的首页", r"↑", r"© 2016‑.*?版权所有",
        r"文学著作‑三国演义", r"首页", r"下一节：.*", r"上一节：.*",
        r"（原文）", r"中华民族文学著作：.*开源在此。",
        r"三国演义，罗贯中 著.*介绍：.*",
        r"第[一二三四五六七八九十百]+回.*?\n",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text)
    text = re.sub(r"\n{3,}", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_paragraphs(text: str, chunk_len=280):
    paragraphs = []
    start = 0
    while start < len(text):
        end = start + chunk_len
        chunk = text[start:end]
        cut_pos = chunk.rfind("。")
        if cut_pos > 50:
            end = start + cut_pos + 1
            chunk = text[start:end]
        paragraphs.append(chunk.strip())
        start = end
    return paragraphs

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    clean_text = clean_raw_text(raw)
    chunks = split_paragraphs(clean_text)
    all_data = []
    max_total = 12000

    for ck in chunks:
        if len(ck) <80:
            continue
        event_key = ck[:22].strip("，。！？ ")
        for qt in QUESTION_TEMPLATES[:2]:
            human_q = qt.format(event_key)
            item = {
                "system": SYSTEM_PROMPT,
                "instruction": human_q,
                "output": ck
            }
            all_data.append(item)
            if len(all_data)>=max_total:
                break
        if len(all_data)>=max_total:
            break
    # 输出完整json数组，适配你json.load()
    with open(OUTPUT_FILE,"w",encoding="utf-8") as fw:
        json.dump(all_data, fw, ensure_ascii=False, indent=2)
    print(f"生成样本 {len(all_data)} 条，输出：{OUTPUT_FILE}")

if __name__=="__main__":
    main()
