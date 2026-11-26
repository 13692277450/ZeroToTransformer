import json, os, hashlib, time
from openai import OpenAI

client = OpenAI(api_key="sk-a7db2ca8048899fb36695bfba607ea29f590f156e3114c7314eb168c6c74654c", base_url="https://api.qnaigc.com/v1")  # 任意 OpenAI 兼容接口

BOOKS = {
    "红楼梦,西游记,水浒传,三国演义,聊斋志异,喻世明言,警世通言,醒世恒言,初刻拍案惊奇，二刻拍案惊奇等古书": "D:/Project/AI/ZeroToTransformer/pretrain_body_clean.txt",

}

GEN_PROMPT = """你是一位精通{book}的学者。请基于下面这段原文，生成{n}条SFT训练数据。
要求：
1. 问题必须能从原文内容回答，严禁编造原文没有的情节；
2. 问题类型多样化：人物性格、情节概括、细节理解、主题分析、人物关系、语言赏析；
3. 答案简洁准确（50~150字），语言半文半白；
4. 不要生成"作者是谁""全书主线"这类全书级常识题，聚焦本段内容；
5. 严格输出JSON数组，格式：[{{"instruction": "...", "output": "..."}}, ...]

原文：
{chunk}
"""


def gen_one_book(book_name, txt_path, target=4500, chunk_chars=3000):
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
    chunks = [text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars)]

    seen = set()
    results = []
    for ci, chunk in enumerate(chunks):
        if len(results) >= target:
            break
        try:
            resp = client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b0-free",
                messages=[{"role": "user", "content": GEN_PROMPT.format(
                    book=book_name, n=5, chunk=chunk)}],
                temperature=0.8,
            )
            items = json.loads(resp.choices[0].message.content) # type: ignore
        except Exception as e:
            print(f"chunk {ci} 失败: {e}")
            continue

        for it in items:
            q = it.get("instruction", "").strip()
            a = it.get("output", "").strip()
            if not q or not a:
                continue
            h = hashlib.md5(q.encode()).hexdigest()   # 去重
            if h in seen:
                continue
            seen.add(h)
            print(f"Q: {q}")
            print(f"A: {a}")
            results.append({"book": book_name, "instruction": q, "output": a})

        if ci % 20 == 0:
            print(f"{book_name}: {len(results)}/{target}")
            with open(f"sft_{book_name}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        time.sleep(0.5)   # 控制请求频率

    return results

# 11本书 × ~4500条 ≈ 50000条
for book, path in BOOKS.items():
    gen_one_book(book, path, target=4500)