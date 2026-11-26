# import json

# def convert_qa_txt_to_jsonl(txt_path, out_jsonl_path):
#     """
#     支持两种格式：
#     1. 已经是 {"book":...,"instruction":...,"output":...} 单行json（跳过）
#     2. Q: xxx\nA: xxx 格式，自动解析
#     """
#     records = []
#     current_book = None
#     with open(txt_path, "r", encoding="utf-8") as f:
#         lines = [line.strip() for line in f if line.strip()]

#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         # 如果本身就是json对象，直接加载
#         if line.startswith("{"):
#             rec = json.loads(line)
#             current_book = rec["book"]
#             records.append(rec)
#             i += 1
#             continue
#         # 匹配 Q: ...
#         if line.startswith("Q:"):
#             if i + 1 >= len(lines):
#                 break
#             next_line = lines[i+1]
#             if next_line.startswith("A:"):
#                 q = line[2:].strip()
#                 a = next_line[2:].strip()
#                 records.append({
#                     "book": current_book,
#                     "instruction": q,
#                     "output": a
#                 })
#                 i += 2
#                 continue
#         i += 1

#     # 写入jsonl
#     with open(out_jsonl_path, "w", encoding="utf-8") as f:
#         for r in records:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")
#     print(f"完成，共 {len(records)} 条，输出到 {out_jsonl_path}")
#     return records


# # ========== 使用示例 ==========
# if __name__ == "__main__":
#     # 注意：txt里面前面最好先有一条json样例，用来读取book名称，后面的Q/A自动继承这个book
#     # 或者你直接固定book名字
#     BOOK_NAME = "三国演义"
#     in_txt = "sft_data2.txt"
#     out_jsonl = "sft_data2_out.jsonl"

#     # 如果不想靠文件里的json记录book，直接固定book版本：
#     records = []
#     with open(in_txt, "r", encoding="utf-8") as f:
#         lines = [line.strip() for line in f if line.strip()]
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         if line.startswith("Q:") and i+1 < len(lines) and lines[i+1].startswith("A:"):
#             q = line[2:].strip()
#             a = lines[i+1][2:].strip()
#             records.append({
#                 "book": BOOK_NAME,
#                 "instruction": q,
#                 "output": a
#             })
#             i += 2
#         else:
#             i += 1

#     with open(out_jsonl, "w", encoding="utf-8") as f:
#         for r in records:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")
#     print(f"固定book={BOOK_NAME}，一共 {len(records)} 条")


import json

jsonl_file = "sft_data2_out.jsonl"
out_json = "sft_new_data.json"

all_data = []
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_data.append(json.loads(line))

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
