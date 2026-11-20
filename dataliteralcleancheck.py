import re
from collections import Counter

CLEAN = r"pretrain_body_clean.txt"   # Windows 下用你实际路径
text = open(CLEAN, encoding="utf-8").read()

checks = {
    "上一节":        len(re.findall(r"上一节", text)),
    "下一节":        len(re.findall(r"下一节", text)),
    "原文版":        len(re.findall(r"原文版", text)),
    "目录":          len(re.findall(r"目录", text)),
    "版权所有":      len(re.findall(r"版权所有", text)),
    "阿星星":        len(re.findall(r"阿星星", text)),
    "文学著作":      len(re.findall(r"文学著作", text)),
    "著介绍":        len(re.findall(r"著介绍", text)),
    "©":            len(re.findall(r"©", text)),
    "第X回总数":     len(re.findall(r"第[一二三四五六七八九十百零〇\d]{1,8}回", text)),
    "第X卷总数":     len(re.findall(r"第[一二三四五六七八九十百零〇\d]{1,8}卷", text)),
}
for k, v in checks.items():
    print(f"{k:12s} {v:8d}")

# 密集目录检测:统计有多少个"目录区"
pat = re.compile(r"第[一二三四五六七八九十百零〇\d]{1,8}[回卷]")
dense = sum(1 for i in range(0, len(text)-160, 80)
            if len(pat.findall(text[i:i+160])) >= 3)
print(f"\n疑似目录窗口数: {dense}  (每窗口约80字 → 约 {dense*80} 字)")

# 高频 8-gram,看重复污染
c = Counter(text[i:i+8] for i in range(0, len(text)-8, 4))
print("\n最高频 8-gram:")
for s, n in c.most_common(20):
    print(f"  {n:6d}  {s}")


text = open(r"D:\Project\AI\ZeroToTransformer\pretrain_body_clean.txt", encoding="utf-8").read()

for k in ["滚滚长江东逝水", "浪花淘尽英雄", "是非成败转头空",
          "青山依旧在", "白发渔樵江渚上", "惯看秋月春风",
          "一壶浊酒喜相逢", "话说天下大势"]:
    print(f"{k:12s} {'✅' if k in text else '❌ 缺失'}")

print("\n开头 200 字:\n", text[:200])
