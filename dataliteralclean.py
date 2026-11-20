import os
import re
import json
from collections import Counter

OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

SRC = "allChineseLiteral_clean.txt"

with open(SRC, "r", encoding="utf-8") as f:
    raw_text = f.read()


def normalize_text(t):
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t\u3000]+", "", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def remove_boilerplate(t):
    patterns = [
        r"阿星星\s*·\s*版权所有",
        r"©\s*\d{4}-?",
        r"[-—]{2,}",
        r"中华民族文学著作[：:].{0,300}?开源在此。",
        r"红楼梦西游记水浒传三国演义聊斋志异喻世明言警世通言醒世恒言初刻拍案惊奇二刻拍案惊奇",
        r"文学著作(红楼梦|西游记|水浒传|三国演义|聊斋志异|喻世明言|警世通言|醒世恒言|初刻拍案惊奇|二刻拍案惊奇){1,5}",
        r"(红楼梦|西游记|水浒传|三国演义|聊斋志异|喻世明言|警世通言|醒世恒言|初刻拍案惊奇|二刻拍案惊奇)原文版\1?",
        r"原文版(红楼梦|西游记|水浒传|三国演义|聊斋志异|喻世明言|警世通言|醒世恒言|初刻拍案惊奇|二刻拍案惊奇)",
        r"(红楼梦|西游记|水浒传|三国演义|聊斋志异|喻世明言|警世通言|醒世恒言|初刻拍案惊奇|二刻拍案惊奇)[：:]目录",
        r"上一节[：:].{0,80}",
        r"下一节[：:].{0,80}",
        r"[：:]目录",
        r"目录原文",
    ]

    for p in patterns:
        t = re.sub(p, "", t)

    return t


def remove_dense_toc(t):
    """
    删除密集目录区。
    例如：
    第一回xxx第二回xxx第三回xxx...
    这种连续目录不适合预训练。
    """
    pat = re.compile(r"第[一二三四五六七八九十百零〇两\d]{1,8}[回卷章节]")
    out = []
    i = 0
    n = len(t)

    while i < n:
        window = t[i:i + 160]
        hits = len(pat.findall(window))

        # 160字内出现3个以上“第X回/卷”，基本就是目录
        if hits >= 3:
            j = i
            while j < n:
                w = t[j:j + 160]
                if len(pat.findall(w)) >= 2:
                    j += 80
                else:
                    break
            i = j
        else:
            out.append(t[i])
            i += 1

    return "".join(out)


def remove_intro_blocks(t):
    """
    删除网站式介绍块。
    注意：这不是删所有介绍，只删“文学著作xxx，作者著介绍：”这种网页摘要。
    """
    t = re.sub(
        r"(红楼梦|西游记|水浒传|三国演义|聊斋志异|喻世明言|警世通言|醒世恒言|初刻拍案惊奇|二刻拍案惊奇)，[^。]{0,20}著介绍[：:].{0,800}?原文版",
        "",
        t,
        flags=re.S,
    )
    return t


def final_clean(t):
    t = normalize_text(t)
    t = remove_boilerplate(t)
    t = remove_intro_blocks(t)
    t = remove_dense_toc(t)
    t = remove_boilerplate(t)
    t = normalize_text(t)

    # 去掉特别短的碎片行
    lines = []
    for line in t.split("\n"):
        line = line.strip()
        if len(line) >= 20:
            lines.append(line)
        elif re.search(r"[。！？；]$", line) and len(line) >= 8:
            lines.append(line)

    return "\n".join(lines)


clean_text = final_clean(raw_text)

save_path = os.path.join(OUT_DIR, "pretrain_body_clean.txt")
with open(save_path, "w", encoding="utf-8") as f:
    f.write(clean_text)

print("✅ 已保存:", save_path)
print(f"原始字数: {len(raw_text):,}")
print(f"清洗后字数: {len(clean_text):,}")
print(f"保留比例: {100 * len(clean_text) / len(raw_text):.2f}%")

print("=" * 80)
print("清洗后开头 500 字:")
print(clean_text[:500])
print("=" * 80)
print("清洗后结尾 500 字:")
print(clean_text[-500:])
