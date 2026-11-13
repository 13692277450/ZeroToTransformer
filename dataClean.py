import re
import os
import shutil

with open("allChineseLiteral.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
    raw_text = raw_text.replace("\n", "")
# 假设 raw_text 是你读取的一大段文本
# 1. 删掉常见的网页导航和版权信息
raw_text = re.sub(r'← 阿星星的首页', '', raw_text)
raw_text = re.sub(r'阿星星 · 版权所有', '', raw_text)
raw_text = re.sub(r'© 2016-', '', raw_text)
raw_text = re.sub(r'（原文）', '', raw_text)
raw_text = re.sub(r'首页', '', raw_text)

# 2. 删掉所有的网页链接（如果有的话）
raw_text = re.sub(r'http[s]?://\S+', '', raw_text)

# 3. 如果还有多余的符号（比如 ←、↑），可以只保留中文字符、数字和标点
# 这行正则极其霸道：除了中文、数字和标准标点，其他所有符号全部剔除
raw_text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9，。！？：；“”‘’（）《》、\n]', '', raw_text)

# 4. 去掉多余的连续空行
raw_text = re.sub(r'\n+', '\n', raw_text)

print("清理后的文本示例：")
print(raw_text[:500])
with open("allChineseLiteral_clean.txt", "w", encoding="utf-8") as file:
    file.write(raw_text)
print("Data clean done.")
