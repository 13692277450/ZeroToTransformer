# -*- coding: utf-8 -*-
"""
批量清洗当前目录（及子目录）下所有 .html 文件，提取正文纯文本，保存为同名 .txt。
依赖：beautifulsoup4（已装 bs4 4.15.0），lxml 可选（缺省用内置 html.parser）。
"""

import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# ---------------------- 配置 ----------------------
# 待扫描的根目录：默认脚本所在目录；传命令行参数可覆盖
ROOT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
# 是否递归子目录
RECURSIVE = True
# 输出 txt 保存目录：默认与 html 同目录；若要集中存放可改成 ROOT_DIR / "txt"
OUTPUT_DIR = None  # 例: ROOT_DIR / "txt"

# 需要从正文中剔除的标签（整段删除其内容）
DROP_TAGS = [
    "script", "style", "noscript", "iframe", "nav", "footer", "header",
    "form", "button", "input", "select", "textarea", "img", "svg", "aside",
]
# 只保留这些标签内的正文（None 表示不限制）
# 你的样例正文集中在 h1/h2/p，可按需打开：
KEEP_TAGS = None  # ["h1", "h2", "p", "td", "th", "li"]

# 样例里要剔除的导航/版权类固定文案（正则，匹配到的整行删除）
DROP_LINE_PATTERNS = [
    r"上一节[：:].*",
    r"下一节[：:].*",
    r"首页$",
    r"目录$",
    r"←\s*阿星星的首页",
    r"阿星星.*版权所有",
    r"©\s*\d{4}[-–]?\d*.*",
    r"回到顶部",
    r"↑$",
    r"^\s*html\s*$",
]

# 章节标题（h1）里去掉的尾部说明，如“ 原文”
TITLE_SUFFIX_STRIP = "原文"

# ---------------------- 合并配置 ----------------------
# 清洗完成后，是否把所有 txt 合并成一个文件
MERGE_ENABLED = True
# 合并后的文件名（保存在 OUTPUT_DIR 或 ROOT_DIR 下）
MERGE_FILE = "all_merged.txt"
# 合并时是否在每篇正文前加上来源文件名作为标题
ADD_SOURCE_TITLE = False
# 每篇正文之间的分隔符
MERGE_SEPARATOR = "\n\n\n"
# --------------------------------------------------


def clean_text(text: str) -> str:
    """清理单段文本：压缩空白、去掉多余符号。"""
    if text is None:
        return ""
    # 统一换行/空白
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t　]+", " ", text)
    # 去掉行首尾空白
    text = "\n".join(line.strip() for line in text.splitlines())
    # 合并连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_drop_line(line: str) -> bool:
    """判断一行是否属于导航/版权等噪音。"""
    s = line.strip()
    if not s:
        return True
    for pat in DROP_LINE_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def extract_from_html(html_path: Path) -> str:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # 1. 删除脚本、样式、导航、页脚等
    for tag in soup(DROP_TAGS):
        tag.decompose()

    # 2. 提取段落
    if KEEP_TAGS:
        nodes = soup.find_all(KEEP_TAGS)
    else:
        # 直接取 body，没有 body 就用整份 soup
        nodes = soup.body.find_all(["h1", "h2", "h3", "p", "li", "td", "th", "div"]) if soup.body \
            else soup.find_all(["h1", "h2", "h3", "p", "li", "td", "th", "div"])

    paragraphs = []
    for node in nodes:
        # 跳过被 drop 标签包裹的（已 decompose，这里保险）
        if node.find_parent(DROP_TAGS):
            continue
        txt = node.get_text(separator=" ", strip=True)
        txt = clean_text(txt)
        if not txt:
            continue
        if should_drop_line(txt):
            continue
        paragraphs.append(txt)

    # 3. 去重（相邻重复段）
    dedup = []
    for p in paragraphs:
        if not dedup or dedup[-1] != p:
            dedup.append(p)

    # 4. 标题段去掉尾部“原文”
    lines = []
    for p in dedup:
        if p.endswith(TITLE_SUFFIX_STRIP):
            p = p[: -len(TITLE_SUFFIX_STRIP)].strip()
        lines.append(p)

    return "\n\n".join(lines).strip()


def collect_txt_files() -> list[Path]:
    """扫描目录下所有已生成的 .txt（排除合并文件本身）。"""
    pattern = "rglob" if RECURSIVE else "glob"
    files = sorted(getattr(Path(ROOT_DIR), pattern)("*.txt"))
    return [f for f in files if f.name != MERGE_FILE]


def merge_txt_files(txt_items: list[tuple[str, str]], merge_path: Path):
    """把 [(来源名, 正文), ...] 合并写入一个文件。"""
    blocks = []
    for source_name, text in txt_items:
        text = clean_text(text)
        if not text:
            continue
        if ADD_SOURCE_TITLE:
            # 用文件名（去扩展名）作为小节标题
            title = Path(source_name).stem
            # blocks.append(f"# {title}\n\n{text}")
        else:
            blocks.append(text)

    merged = MERGE_SEPARATOR.join(blocks).strip()
    merge_path.write_text(merged, encoding="utf-8")
    print(f"\n[合并] 共 {len(blocks)} 篇 -> {merge_path.name} ({len(merged)} 字)")


def main():
    # 命令行开关：--merge-only 只合并已有 txt，不清洗 html
    merge_only = "--merge-only" in sys.argv

    if merge_only:
        txt_files = collect_txt_files()
        if not txt_files:
            print(f"未在 {ROOT_DIR} 下找到 .txt 文件")
            return
        items = [(f.name, f.read_text(encoding="utf-8", errors="ignore")) for f in txt_files]
        out_root = Path(OUTPUT_DIR) if OUTPUT_DIR else ROOT_DIR
        merge_txt_files(items, out_root / MERGE_FILE)
        return

    if RECURSIVE:
        html_files = sorted(ROOT_DIR.rglob("*.html"))
    else:
        html_files = sorted(ROOT_DIR.glob("*.html"))

    if not html_files:
        print(f"未在 {ROOT_DIR} 下找到 .html 文件")
        return

    out_root = Path(OUTPUT_DIR) if OUTPUT_DIR else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    ok = 0
    merged_items: list[tuple[str, str]] = []
    for html_path in html_files:
        try:
            text = extract_from_html(html_path)
            if not text:
                print(f"[跳过] {html_path.name} 无正文")
                continue

            if out_root:
                rel = html_path.relative_to(ROOT_DIR)
                out_path = out_root / rel.with_suffix(".txt")
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = html_path.with_suffix(".txt")

            out_path.write_text(text, encoding="utf-8")
            print(f"[完成] {html_path.name} -> {out_path.name} ({len(text)} 字)")
            merged_items.append((out_path.name, text))
            ok += 1
        except Exception as e:
            print(f"[失败] {html_path.name}: {e}")

    print(f"\n共处理 {ok}/{len(html_files)} 个文件")

    # 合并成一个文件
    if MERGE_ENABLED and merged_items:
        merge_root = out_root if out_root else ROOT_DIR
        merge_txt_files(merged_items, merge_root / MERGE_FILE)


if __name__ == "__main__":
    main()
