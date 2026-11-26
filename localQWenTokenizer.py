# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.net"
# from transformers import AutoTokenizer

# from huggingface_hub import hf_hub_url

# url = hf_hub_url(
#     repo_id="Qwen/Qwen2.5-0.5B-Instruct",
#     filename="config.json"
# )
# print("实际请求URL：", url)


# tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
# tok.save_pretrained("./qwen05b-tokenizer")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.net"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# 设置http超时
os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"

import requests

os.makedirs("./qwen05b-tokenizer", exist_ok=True)
base_url = "https://hf-mirror.net/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"
file_list = [
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json"
]

for fname in file_list:
    save_path = os.path.join("./qwen05b-tokenizer", fname)
    if os.path.exists(save_path):
        print(f"{fname} 已存在，跳过")
        continue
    print(f"正在下载 {fname} ...")
    url = f"{base_url}/{fname}"
    # stream 流式下载，不依赖HEAD的Content-Length
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print(f"{fname} 下载完成")

# 加载本地tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./qwen05b-tokenizer", trust_remote_code=True)
print("✅ tokenizer加载成功")
