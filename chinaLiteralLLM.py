# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')
import torch, gc
gc.collect()
torch.cuda.empty_cache() # 👈 强制清空显存碎片
#=========================================================
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

                         
#=======================================================
!pip install webdataset livelossplot
import webdataset as wds
import json
import re
import torch.nn as nn
import pandas as pd
import tiktoken
import torch
import os
import glob
from datetime import datetime
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 只让程序看到第 1 号显卡（即物理上的第 2 张卡）。此时程序里的 cuda:0 就是物理上的 cuda:1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
# 此时 torch 只能看到一张卡，默认就是你要的 cuda:1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import logging
from rich.logging import RichHandler
from livelossplot import PlotLosses
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# from mainScore import format_input, check_if_running, generate_model_scores, query_model, console
from rich.console import Console
console = Console()

############################ MODEL CONFIG ##########################
GPT_CONFIG_124M = {
    "model": "gpt2",
    "context_length": 1024,
    "n_layers": 12,
    "n_heads": 12,
    "emb_dim": 768,
    "drop_rate": 0.1,
    "temperature": 0.7,
    "vocab_size": 50257,
    "qkv_bias": True
}
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(MODEL_CONFIGS[model_name])
############################ Tokenizer ##########################
class SimpleTokenizer():
    def __init__(self, vocab):
        # 初始化词汇表, 编码 str to int
        self.str_to_int = vocab
        # 初始化词汇表, 解码 int to str
        self.int_to_str = {i:s for s, i in vocab.items()}
    # 编码文本， 返回 int 序列
    def encode(self, text):
        # 对文本进行预处理，将标点符号和空格作为单独的 token
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            # 对每个 token 进行预处理，将空格替换为下划线
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        # 对每个 token 进行编码
        ids = [self.str_to_int[s] for s in preprocessed]
        # 返回编码后的 int 序列
        return ids
    # 解码 int 序列， 返回文本字符串
    def decode(self, ids):
        # 对每个 id 进行解码
        text = "".join([self.int_to_str[i] for i in ids])
        # 对文本进行后处理，将下划线替换为空格
        text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text)
        # 返回解码后的文本字符串
        return text
############################ Tokenizer Gpt2##########################
tokenizer = tiktoken.get_encoding("gpt2")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# 如果你是在计算 API 费用或对接 ChatGPT，千万别用 GPT-2 的编码，必须用 tiktoken.get_encoding("cl100k_base")（对于更新的 GPT-4o 则是 o200k_base）。词表也变大了（cl100k_base 通常约 10 万级别）。
############################ FeedForward ############################
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # 768 -> 3072
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
            # 3072 -> 768
        )
    def forward(self, x):
        return self.layers(x)

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################ LayerNorm ##############################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
        

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################ MultiHeadAttention ######################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                    diagonal=1)
        )
    def forward(self, x, attention_mask=None):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # type: ignore
        
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        if attention_mask is not None:
            extended_mask = attention_mask[:,None, None,:]
            attn_scores = attn_scores.masked_fill(extended_mask == 0, -torch.inf)
        
        attn_weights = torch.softmax(
            attn_scores /keys.shape[-1]**0.5, dim=-1
        )
        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec        
        
############################ CausalAttention ######################

        

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################ Transformer Block ######################
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
        d_in = cfg["emb_dim"],
        d_out = cfg["emb_dim"],
        context_length = cfg["context_length"],
        num_heads = cfg["n_heads"],
        dropout = cfg["drop_rate"],
        qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x, attention_mask=None):
        # 第一个子层：多头因果注意力
        shortcut = x            # 1. 备份原始输入
        x = self.norm1(x)       # 2. 层归一化（稳定数值）
        x = self.att(x, attention_mask)         # 3. 多头因果注意力（提取上下文关系）
        x = self.drop_shortcut(x) # 4. Dropout（训练时防止过拟合，推理时关闭）
        x = x + shortcut        # 5. 第一次残差连接（将处理后的特征与原始输入相加）
        
        # 第二个子层：前馈神经网络
        shortcut = x             # 🚨 注意：这里拼写错了！应该是 shortcut
        x = self.norm2(x)       # 6. 再次层归一化
        x = self.ff(x)          # 7. 前馈网络（先膨胀到 4 倍，GELU 激活，再收缩回原维度）
        x = self.drop_shortcut(x) # 8. Dropout
        x = x + shortcut        # 9. 第二次残差连接
        return x
    

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################ GPT MODEL ##########################
class GPT_MODEL(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self,  input_ids, attention_mask=None, labels=None):
        batch_size, seq_len= input_ids.shape
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=input_ids.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, attention_mask) ######书上没有，必须
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        
        return {"logits": logits, "loss": loss}
############################ MODEL ############################

# %% [code]
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length +1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# %% [markdown]
# 

# %% [code] {"jupyter":{"outputs_hidden":false}}
def create_dataloader(txt, batch_size=2, max_length=256,
                     stride=128, shuffle=True, drop_last=True,
                     num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
    

# %% [code]
with open("/kaggle/input/datasets/mangzhang/allchineseliteral-clean/allChineseLiteral_clean.txt") as f:
    raw_text = f.read()
dataloader = create_dataloader(
    raw_text, batch_size=2, max_length=1024, stride=1, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)


# %% [code] {"jupyter":{"outputs_hidden":false}}
############################ WebDataset ############################
def create_text_webdataset(file_path, output_dir, chunk_size = 2000,max_shards_size=10000):
    """
    把文本文件 转换成 WebDataset 的 .tar 分片。
    
    :param file_path: 包含文本数据的文件路径
    :param output_dir: 输出路径，例如 "shards/retail-%06d.tar"
    :param max_shards_size: 每个分片包含的样本数
    """
    # SharedWriter 会自动按maxcount分片
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "literal-%06d.tar")
    sample_idx =0
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    # raw_text = "".join(raw_text)
    total_len = len(raw_text)
    print(f"total text len: {total_len}")
    split_pattern = re.compile(r'([。!?])')
    sentences = split_pattern.split(raw_text)
    pieces = []
    buf = ""
    for seg in sentences:
        buf += seg
        if len(buf) >= chunk_size:
            pieces.append(buf)
            buf = ""
    if len(buf.strip()) > 0:
        pieces.append(buf)
        
    with wds.ShardWriter(output_pattern, maxcount=max_shards_size) as sink: # type: ignore
        # 遍历 DataFrame 中的每一行
        for block in pieces:
            if len(block) < 50:
                continue
            sample = {
                "__key__": f"chunk_{sample_idx:08d}",
                "text": block.encode("utf-8") # type: ignore
            }
            sink.write(sample)
            sample_idx += 1
    print(f"Total samples blocks: {sample_idx}")
            # console.print(sample.values())

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################Prepare Data############################
def decode_txt(sample):
    """
    自定义解码函数：当样本里包含 'txt' 字段时，将字节流解码为字符串。
    """
    if "text" in sample:
        sample["text"] = sample["text"].decode("utf-8")
    return sample
def prepare_data():
    file_path = "D:\\Project\\AI\\ZeroToTransformer\\allchineseliteral_clean.txt"
    # 生成分片，每个分片 10000 条样本
    create_text_webdataset(file_path,  "shardsTxt", max_shards_size=10000)
############################ MODEL#################################

model = GPT_MODEL(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)
# model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
############################Train data ############################
# 1. 定义一个专门读取散装 text 文件的 Dataset
class KaggleTextDataset(Dataset):
    def __init__(self, folder_path):
        # 自动找到文件夹下所有以 .text 结尾的文件
        folder_path = "/kaggle/input/datasets/mangzhang/allchineseliteral-clean/"
        self.file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
        if not self.file_paths:
            raise FileNotFoundError(f"在 {folder_path} 下没有找到任何 .text 文件，请检查路径！")
        print(f"✅ 成功找到 {len(self.file_paths)} 个文本文件")
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        # 每次读取一个文件
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            text = f.read()
        # 返回一个元组，和之前 WebDataset 的 .to_tuple("text") 格式保持一致
        return (text,)
def collate_fn(batch):
    texts = [item[0].strip() for item in batch]
    encoded_texts = [tokenizer.encode(text)[:1024] for text in texts]
    max_len = max(len(ids) for ids in encoded_texts)
    pad_token_id = 50256
    padded_ids = []
    attention_masks = []
    for ids in encoded_texts:
        padded = ids + [pad_token_id] * (max_len -len(ids))
        mask = [1] * len(ids) + [0] * (max_len -len(ids))
        padded_ids.append(padded)
        attention_masks.append(mask)
    input_ids = torch.tensor(padded_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

# %% [code]
import torch
from torch.utils.data import Dataset, DataLoader

class SingleFileTextDataset(Dataset):
    def __init__(self, file_path, chunk_size=1000):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # 切分成一个个 chunk 列表
        self.chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
        # 过滤掉太短的
        self.chunks = [c for c in self.chunks if len(c.strip()) > 10]
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return (self.chunks[idx],)


# 1. 实例化整个 Dataset
full_dataset = SingleFileTextDataset("/kaggle/input/datasets/mangzhang/allchineseliteral-clean/allChineseLiteral_clean.txt")
total_size = len(full_dataset) # 比如 2918

# 2. 计算切分点
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)
# test_size 就是剩下的 10%

# 3. 你完全可以像切列表一样切 Dataset！（因为底层数据是 List）
train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(full_dataset, range(train_size + val_size, total_size))

# 4. 直接扔进 DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# %% [code]
# import os

# # 直接遍历你上传的那个大文件夹，看看真实的路径到底长什么样
# for dirname, _, filenames in os.walk('/kaggle/input/datasets/mangzhang'):
#     for filename in filenames:
#         if filename.endswith('.txt'):
#             print("真实绝对路径:", os.path.join(dirname, filename))

# %% [code]
def train_dataTxt():
    # folder_path ="/kaggle/input/datasets/mangzhang/shardstxt3/"
    folder_path = "/kaggle/input/datasets/mangzhang/allchineseliteral-clean/"
    # train_dataset = (
    #     wds.WebDataset(tar_path, shardshuffle=1000) # pyright: ignore[reportAttributeAccessIssue]
    #     .map(decode_txt)      # 👈 用 .map() 来执行自定义的解码逻辑
    #     .to_tuple("text") 
    # )
    with open("/kaggle/input/datasets/mangzhang/allchineseliteral-clean/allChineseLiteral_clean.txt","r",encoding="utf-8") as f:
        text_data = f.read()
    
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    train_dataloader = DataLoader(
       dataset, #train_dataset, 
       batch_size=2, 
       shuffle=True, 
       collate_fn=collate_fn)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    stop_train = False
    for epoch in range(1):
        # if stop_train == True:
        #     # save_model()
        #     break
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            # plot_chart(loss.item(), epoch)
            if step % 10 == 0:
                print("Time now is ---- ",datetime.now())
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
                #Grad——Norm：{model.out_head.weight.grad.norm().item()}")
                # if loss.item() < 0.4 :
                #     stop_train = True
                #     save_model()
                #     break
                print(f"========== 第 Step:{step} batch: {batch} 轮训练完成！ ==========")
    print(f"========== 第 epoch: {epoch} 轮训练彻底完成！ ==========")



# %% [markdown]
# 

# %% [code] {"jupyter":{"outputs_hidden":false}}
def train_data():
    # folder_path ="/kaggle/input/datasets/mangzhang/shardstxt3/"
    folder_path = "/kaggle/input/datasets/mangzhang/allchineseliteral-clean/"
    # train_dataset = (
    #     wds.WebDataset(tar_path, shardshuffle=1000) # pyright: ignore[reportAttributeAccessIssue]
    #     .map(decode_txt)      # 👈 用 .map() 来执行自定义的解码逻辑
    #     .to_tuple("text") 
    # )
    train_dataset = KaggleTextDataset(folder_path)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=2, 
        num_workers=0,
        collate_fn = collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    stop_train = False
    for epoch in range(1):
        if stop_train == True:
            # save_model()
            break
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print("Time now is ---- ",datetime.now())
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
                plot_chart(loss.item(), epoch)
                #Grad——Norm：{model.out_head.weight.grad.norm().item()}")
                if loss.item() < 0.4 :
                    stop_train = True
                    save_model()
                    break

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################Log status############################
# 1. 创建 Logger
logger = logging.getLogger("mainFineTune")
logger.setLevel(logging.INFO)

# 2. 终端 Handler（Rich 负责好看，带时间戳）
console_handler = RichHandler(show_time=True, rich_tracebacks=True)
# 注意：RichHandler 的时间格式也由 datefmt 控制
console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

# 3. 文件 Handler（标准格式，带时间戳）
file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_formatter = logging.Formatter(
    # 👇 asctime 会自动读取 datefmt 的格式
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

# 4. 挂载
logger.addHandler(console_handler)
logger.addHandler(file_handler)


########################Tensorboard chart########################


# 默认会创建一个 ./runs/ 文件夹来存日志
#writer = SummaryWriter("runs/transformer_finetune_experiment")

#for epoch in range(num_epochs):
    # ... 你的训练逻辑 ...
    
    # 记录训练指标（Scalar 就是标量，比如 loss, acc）
    # writer.add_scalar("Loss/train", train_loss, epoch)
    # writer.add_scalar("Accuracy/train", train_acc, epoch)
    
    # 验证集指标同理
    # writer.add_scalar("Loss/val", val_loss, epoch)
    # writer.add_scalar("Accuracy/val", val_acc, epoch)

# 训练结束后，关闭 writer（或者手动 flush 确保数据写入磁盘）
#writer.close()

# %% [code] {"jupyter":{"outputs_hidden":false}}
#########################Simple Plot chart#########################
plotlosses = PlotLosses()

def plot_chart(train_loss, epoch):
    """
    绘制损失和精度曲线。
    :param loss: 训练损失列表
    :param val_loss: 验证损失列表
    :param train_acc: 训练精度列表
    :param val_acc: 验证精度列表
    :param epoch: 当前训练轮次
    """
    # 绘制损失曲线
    plotlosses.update({
        'train_loss': train_loss, 
        # 'val_loss': val_loss, 
        # 'train_acc': train_acc, 
        # 'val_acc': val_acc,
        'epoch': epoch
        })
    # 绘制精度曲线
    plotlosses.send()
plotlosses = PlotLosses()

#########################Tqdm Progress Bar#########################
# from tqdm import tqdm
# for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Processing"):
#     # 处理每个条目
#     input_text = format_input(entry)
#     token_ids = generate(
#         model=model,
#         idx=text_to_token_ids(input_text, tokenizer).to(device),
#         max_new_tokens=256,
#         context_size=BASE_CONFIG["context_length"],
#         eos_id=50256
#     )
#     generated_text = token_ids_to_text(token_ids, tokenizer)
#     response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
#     # 去掉输入部分，只保留生成的回复)
#     test_data[i]["model_response"] = response_text
# with open("instruction-data-with-response.json", "w", encoding="utf-8") as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=4)


############################计算损失批次############################

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算一个批次的损失。
    :param input_batch: 输入数据的批次
    :param target_batch: 目标数据的批次
    :param model: 模型
    :param device: 设备（CPU 或 GPU）
    :return: 损失值
    """
    # 将输入和目标数据移动到指定设备
    pass
    # 前向传播
    
    # 获取损失值

    # 返回损失值

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################计算损失加载器############################

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算整个数据加载器的平均损失。
    :param data_loader: 数据加载器
    :param model: 模型
    :param device: 设备（CPU 或 GPU）
    :param num_batches: 要计算的批次数量（可选）
    :return: 平均损失值
    """
    # 初始化总损失和批次计数
    pass
    # 遍历数据加载器

        # 检查是否需要停止遍历
    
        # 计算当前批次的损失
        
        # 累加损失值
        
        # 增加批次计数

    # 计算平均损失值
    
    # 返回平均损失值

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################训练分类检验############################

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    # 初始化训练损失， 验证损失， 训练精度， 验证精度列表 变量
    pass
    # 初始化example_seen， global_step
    
    # 遍历每个epoch
        # 设置模型为训练模式
        
        # 遍历输入，输出训练数据加载器中的每个批次
        
            # 优化器清除梯度

            # 计算损失
            
            # 反向传播
            
            # 优化器更新参数
            
            # track examples instead of tokens
            
            # 步长增加1
            
            
            # 如果步长是eval_freq的倍数
                # 计算训练损失和验证损失
                
                # 增加到训练损失变量
                
                # 增加到验证损失变量
                
                # 打印当前epoch，global_step，训练损失，验证损失
            
            # 每个批次后的精度计算
            # 训练精度计算
            
            # 验证精度计算
            
            # 打印当前epoch的训练精度，验证精度
            
            # 训练精度变量增加此次的训练精度
            
            # 验证精度变量增加此次的验证精度
            
        # 返回训练损失，验证损失，训练精度，验证精度， example_seen
        
        # 在代码里，global_step 是每次循环 +1（代表走了多少步），而 examples_seen 是每次加上 batch_size（代表看了多少数据）。
        # 如果你发现 loss 降得很慢，你可以看 examples_seen。如果 # examples_seen 已经很大了（比如看了十几万条），但 Loss 还不降，那就说明是模型欠拟合或者学习率有问题，而不是训练时间不够。
    

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################评估模式############################

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估模型在训练集和验证集上的性能。
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param device: 设备（CPU 或 GPU）
    :param eval_iter: 评估的批次数量
    :return: 训练集和验证集的平均损失值
    """
    # 设置模型为评估模式
    pass
    # 禁止梯度下降，计算训练集的平均损失
    
    # 计算验证集的平均损失
    
    # 训练模式
    
    # 返回训练集和验证集的平均损失值



############################优化器设计############################
# 学习率设计

# 优化器设计
optimizer  = {}

# 训练轮次初始化

# 训练分类器的简单训练函数

# 打印训练损失，验证损失，训练精度，验证精度， example_seen， global_step， examples_seen

############################SAVE MODEL############################
import re
def save_model():
    
 # 1. 获取当前时间戳（比如 20261113_153022）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. 把你配置里的一些关键参数也带在文件名里（可选）
    model_size = NEW_CONFIG.get("n_layers", 12)
    emb_dim = NEW_CONFIG.get("emb_dim", 768)
    
    # 3. 组装文件名：包含时间戳，绝对不会覆盖
    # file_name = f"allChineseLiteral_L{model_size}_E{emb_dim}_{timestamp}.pth"
    file_name = f"allChineseLiteral.pth"
    # 或者如果你想把原来的 CHOOSE_MODEL 恢复：
    # file_name = f"{re.sub(r'[ ()]', '', model_name)}_sft.pth"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"旧的 {file_name} 已删除。")
    else:
        print(f"{file_name} 不存在，无需删除。")
    torch.save(model.state_dict(), file_name)
    logger.info(f"Model saved to {file_name}")




##########################MAIN############################

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    status = "Training"
    # logger.info("Starting the prepare data...")
    # prepare_data()
    logger.info("Starting the train data...")
    console.print("Training data...", style='blue')
    train_dataTxt()
    logger.info("Starting the save model...")
    save_model()
    
    if status == "Training":
        # prepare_data()
        pass
    else:   
        #  check_if_running("ollama")
        pass

    # generate_model_scores(test_data, "model_response")

# %% [code] {"jupyter":{"outputs_hidden":false}}
model=GPT_MODEL(NEW_CONFIG)
model.to(device)
model.eval()
model.load_state_dict(torch.load("allChineseLiteral.pth"))

# %% [code] {"jupyter":{"outputs_hidden":false}}
def generate_text(prompt, max_new_tokens=100):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    top_k = 50

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :] / 0.8
            top_logits, top_indices = torch.topk(next_token_logits, top_k)
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_tensor = torch.cat([input_tensor, next_token_id], dim=-1)
    return tokenizer.decode(input_tensor[0].tolist())

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(generate_text("孙悟空"))

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch

def generate_safe(prompt, max_new_tokens=30, temperature=0.8, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # 限制输入长度，防止越界
    if input_tensor.shape[1] > 512:
        input_tensor = input_tensor[:, -512:]
        
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            logits = outputs["logits"]
            
            # 1. 取最后一个词，并除以温度
            next_token_logits = logits[:, -1, :] / temperature
            
            # 2. Top-K 过滤
            if top_k is not None:
                top_logits, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_logits[:, -1]] = float('-inf')
            
            # 3. Softmax 转概率
            probas = torch.softmax(next_token_logits, dim=-1)
            
            # 4. 随机采样（打破死循环）
            next_token_id = torch.multinomial(probas, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token_id], dim=-1)
            
    # 过滤掉特殊 token 再解码
    output_ids = input_tensor[0].tolist()
    output_text = tokenizer.decode(output_ids)
    return output_text

# 测试
print("测试1:", generate_safe("话说唐三藏"))
print("测试2:", generate_safe("孙悟空一听，"))

# %% [code] {"jupyter":{"outputs_hidden":false}}
def test_generate(prompt, max_tokens=30):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            # 截断输入
            idx_cond = input_tensor[:, -512:]
            outputs = model(input_ids=idx_cond)
            logits = outputs["logits"][:, -1, :] / 0.7 # 加入温度 0.7
            
            # Top-K 过滤，只从最高的 50 个词里选
            top_logits, _ = torch.topk(logits, 50)
            min_val = top_logits[:, -1]
            logits[logits < min_val] = float('-inf')
            
            # 随机采样，打破死循环
            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)
            
    # 解码并强制过滤掉所有特殊符号 (50256 等)
    output_ids = input_tensor[0].tolist()
    clean_ids = [i for i in output_ids if i < 50256]
    return tokenizer.decode(clean_ids)

# 测试！
print("输出:", test_generate("话说孙悟空"))

# %% [code]
