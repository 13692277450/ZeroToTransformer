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

# kagglehub.dataset_download('<owner>/<dataset-slug>')
import torch, gc
gc.collect()
torch.cuda.empty_cache() # 👈 强制清空显存碎片
#=========================================================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#=======================================================
# !pip install webdataset livelossplot
# import webdataset as wds
import json
import re
import torch.nn as nn
import pandas as pd
# import tiktoken
import torch
import os
import glob
from datetime import datetime
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
# 只让程序看到第 1 号显卡（即物理上的第 2 张卡）。此时程序里的 cuda:0 就是物理上的 cuda:1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import torch
# 此时 torch 只能看到一张卡，默认就是你要的 cuda:1
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import logging
from rich.logging import RichHandler
# from livelossplot import PlotLosses
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# from mainScore import format_input, check_if_running, generate_model_scores, query_model, console
from rich.console import Console
console = Console()
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "12355"
# dist.init_process_group("nccl", rank=rank, world_size=world_size)
# device = torch.device(f"cuda:{rank}")
# torch.cuda.set_device(rank)
############################ MODEL CONFIG ##########################
# GPT_CONFIG_124M = {
#     "model": "gpt2",
#     "context_length": 1024,
#     "n_layers": 12,
#     "n_heads": 12,
#     "emb_dim": 768,
#     "drop_rate": 0.1,
#     "temperature": 0.7,
#     "vocab_size": 50260,
#     "qkv_bias": True
# }

# 2. 加载 Qwen2.5 的分词器（以 0.5B 版本为例，词表大小约 15 万）
# trust_remote_code=True 对于 Qwen 系列通常是必须的
tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/input/datasets/mangzhang/qwen25tokenizer",
    # "Qwen/Qwen2.5-0.5B-Instruct", 
    trust_remote_code=True,
    local_files_onlu=True

    # "Qwen/Qwen2.5-0.5B-Instruct", 
    # trust_remote_code=True,
    # timeout=40
)
eos_endoftext_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
eos_im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
qwen_vocab_size = len(tokenizer)  # 通常是 151,646 或 151,936 左右

GPT_CONFIG_124M = {
    "model": "gpt2",
    "context_length": 1024,
    "n_layers": 12,
    "n_heads": 12,
    "emb_dim": 768,
    "drop_rate": 0.1,
    "temperature": 0.7,
    "vocab_size": qwen_vocab_size,
    "qkv_bias": True
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
model_name = "gpt2-small (124M)"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# base_enc = tiktoken.get_encoding("gpt2")
# new_special_tokens = {"additional_special_tokens": ["<|persona|>", "<|user|>", "<|bot|>"]}
# new_special_tokens = {
#     "<|persona|>": 50257,
#     "<|user|>": 50258,
#     "<|bot|>": 50259
# }
# tokenizer.add_special_tokens(special_tokens)
# tokenizer = tiktoken.Encoding(
#     name="gpt2-custom",
#     pat_str=base_enc._pat_str,
#     mergeable_ranks=base_enc._mergeable_ranks,
#     special_tokens={
#         **base_enc._special_tokens,
#         **new_special_tokens
#     }
# )



# 3. 调整模型嵌入层大小，容纳新 token
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(MODEL_CONFIGS[model_name])
############################ Tokenizer ##########################
# class SimpleTokenizer():
#     def __init__(self, vocab):
#         # 初始化词汇表, 编码 str to int
#         self.str_to_int = vocab
#         # 初始化词汇表, 解码 int to str
#         self.int_to_str = {i:s for s, i in vocab.items()}
#     # 编码文本， 返回 int 序列
#     def encode(self, text):
#         # 对文本进行预处理，将标点符号和空格作为单独的 token
#         preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
#         preprocessed = [
#             # 对每个 token 进行预处理，将空格替换为下划线
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         preprocessed = [item if item in self.str_to_int
#                         else "<|unk|>" for item in preprocessed]
#         # 对每个 token 进行编码
#         ids = [self.str_to_int[s] for s in preprocessed]
#         # 返回编码后的 int 序列
#         return ids
#     # 解码 int 序列， 返回文本字符串
#     def decode(self, ids):
#         # 对每个 id 进行解码
#         text = "".join([self.int_to_str[i] for i in ids])
#         # 对文本进行后处理，将下划线替换为空格
#         text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text)
#         # 返回解码后的文本字符串
#         return text
# ############################ Tokenizer Gpt2##########################
# print(tokenizer.encode("<|bot|>", allowed_special={"<|bot|>"}))

# %% [code] {"jupyter":{"outputs_hidden":false}}
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

CN_TZ = ZoneInfo("Asia/Shanghai")  # or timezone(timedelta(hours=8))

def dtcn(dt):
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt, tz=timezone.utc)
    elif isinstance(dt, str):
        dt = datetime.fromisoformat(dt)  # handle Z
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # assume naive is UTC
    return dt.astimezone(CN_TZ)
print(dtcn(datetime.now()))

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
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
    
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=input_ids.device)
        )
    
        x = self.drop_emb(tok_embeds + pos_embeds)
    
        for block in self.trf_blocks:
            x = block(x, attention_mask)
    
        x = self.final_norm(x)
        logits = self.out_head(x)
    
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
    
            valid_targets = shift_labels != -100
            if not valid_targets.any():
                raise ValueError(
                    "labels 全部为 -100；请检查 token ID 和 collate_fn"
                )
    
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
    
        return {"logits": logits, "loss": loss}



        
    # def forward(self,  input_ids, attention_mask=None, labels=None):
    #     batch_size, seq_len= input_ids.shape
    #     tok_embeds = self.tok_emb(input_ids)
    #     pos_embeds = self.pos_emb(
    #         torch.arange(seq_len, device=input_ids.device)
    #     )
    #     x = tok_embeds + pos_embeds
    #     x = self.drop_emb(x)
    #     for block in self.trf_blocks:
    #         x = block(x, attention_mask) ######书上没有，必须
    #     x = self.final_norm(x)
    #     logits = self.out_head(x)
        
    #     loss = None
    #     if labels is not None:
    #         loss = torch.nn.functional.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             labels.view(-1),
    #             ignore_index=-100
    #         )
        
        
    #     return {"logits": logits, "loss": loss}
############################ MODEL ############################

# %% [code] {"jupyter":{"outputs_hidden":false}}
# class GPTDatasetV1(Dataset):
#     def __init__(self, txt, tokenizer, max_length, stride):
#         self.input_ids = []
#         self.target_ids = []
#         token_ids = tokenizer.encode(txt)
#         for i in range(0, len(token_ids) - max_length, stride):
#             input_chunk = token_ids[i:i + max_length]
#             target_chunk = token_ids[i + 1: i + max_length +1]
#             self.input_ids.append(torch.tensor(input_chunk))
#             self.target_ids.append(torch.tensor(target_chunk))
#     def __len__(self):
#         return len(self.input_ids)
#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.target_ids[idx]

# %% [code] {"jupyter":{"outputs_hidden":false}}
# def create_dataloader(txt, batch_size=2, max_length=256,
#                      stride=128, shuffle=True, drop_last=True,
#                      num_workers=0):
#     tokenizer = tiktoken.get_encoding("gpt2")
#     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         drop_last=drop_last,
#         num_workers=num_workers
#     )
#     return dataloader

# %% [code] {"jupyter":{"outputs_hidden":false}}

model = GPT_MODEL(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

# %% [code] {"jupyter":{"outputs_hidden":false}}
############################Prepare Data############################
# def decode_txt(sample):
#     """
#     自定义解码函数：当样本里包含 'txt' 字段时，将字节流解码为字符串。
#     """
#     if "text" in sample:
#         sample["text"] = sample["text"].decode("utf-8")
#     return sample
############################ MODEL#################################

# model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
############################Train data ############################
# 1. 定义一个专门读取散装 text 文件的 Dataset
# class KaggleTextDataset(Dataset):
#     def __init__(self, folder_path):
#         # 自动找到文件夹下所有以 .text 结尾的文件
#         folder_path = "/kaggle/input/datasets/mangzhang/allchineseliteralcleaned/"
#         self.file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
#         if not self.file_paths:
#             raise FileNotFoundError(f"在 {folder_path} 下没有找到任何 .text 文件，请检查路径！")
#         print(f"✅ 成功找到 {len(self.file_paths)} 个文本文件")
        
#     def __len__(self):
#         return len(self.file_paths)
        
#     def __getitem__(self, idx):
#         # 每次读取一个文件
#         with open(self.file_paths[idx], "r", encoding="utf-8") as f:
#             text = f.read()
#         # 返回一个元组，和之前 WebDataset 的 .to_tuple("text") 格式保持一致
#         return (text,)
def collate_fn(batch):
    texts = [item[0].strip() for item in batch]

    # for text in texts:
    #     ids = tokenizer(
    #         text,
    #         allowed_special={"<|persona|>", "<|user|>", "<|bot|>"},
            
    # #     )
    # max_len = max(len(ids) for ids in encoded_texts)
    # pad_token_id = 50256
    # padded_ids = []
    # attention_masks = []
    # for ids in encoded_texts:
    #     padded = ids + [pad_token_id] * (max_len -len(ids))
    #     mask = [1] * len(ids) + [0] * (max_len -len(ids))
    #     padded_ids.append(padded)
    #     attention_masks.append(mask)

    batch_enc = tokenizer(
        texts,
        truncation=True,
        truncation_side="left",
        max_length=1024,
        padding=True,
        return_tensors="pt"
    ) # 输出input_ids ， attention_mask
        
    input_ids = batch_enc["input_ids"]
    attention_mask = batch_enc["attention_mask"]
    
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # HF 内部会移位这个移位取消
    # labels = torch.cat([   
    #     labels[:, 1:],
    #     torch.full((labels.size(0), 1), -100, dtype=torch.long)],
    #     dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
from torch.utils.data import Dataset, DataLoader

class SingleFileTextDataset(Dataset):
    def __init__(self, file_path, chunk_size=1000):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # 切分成一个个 chunk 列表
        self.chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
        # 过滤掉太短的
        # self.chunks = [c for c in self.chunks if len(c.strip()) > 10]
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return (self.chunks[idx],)

#######################################




###############################################
# 1. 实例化整个 Dataset
full_dataset = SingleFileTextDataset("/kaggle/input/datasets/mangzhang/allchineseliteralcleaned/pretrain_body_clean.txt")
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
# tiny = SFTDataset(cleaned[:1])
# tiny_loader = DataLoader(tiny, batch_size=1, shuffle=True, collate_fn=collate_fn)

# for n, p in model.named_parameters():
#     p.requires_grad_("lora" in n.lower())
# opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

# model.to(device); model.train()
# for i in range(200):
#     for bb in tiny_loader:
#         opt.zero_grad()
#         l = model(input_ids=bb["input_ids"].to(device),
#                   attention_mask=bb["attention_mask"].to(device),
#                   labels=bb["labels"].to(device))["loss"]
#         l.backward(); opt.step()
#     if i % 20 == 0:
#         print(i, round(l.item(), 4))

# # 立刻背诵测试
# model.eval()
# s0 = cleaned[0]
# print(generate_text(model, build_prompt(s0["system"], s0["instruction"]),
#                     max_new_tokens=300, greedy=True))
# print("---标准答案---")
# print(s0["output"][:300])

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
from torch.optim.lr_scheduler import CosineAnnealingLR



def train_dataTxt():
    # folder_path ="/kaggle/input/datasets/mangzhang/shardstxt3/"
    # folder_path = "/kaggle/input/datasets/mangzhang/qwennewtrainclean/"
    # train_dataset = (
    #     wds.WebDataset(tar_path, shardshuffle=1000) # pyright: ignore[reportAttributeAccessIssue]
    #     .map(decode_txt)      # 👈 用 .map() 来执行自定义的解码逻辑
    #     .to_tuple("text") 
    # )
    # with open("/kaggle/input/datasets/mangzhang/qwennewtrainclean/pretrain_body_clean.txt","r",encoding="utf-8") as f:
        # text_data = f.read()
    # max_length =1024
    # stride = 128
    # dataset = GPTDatasetV1(text_data, tokenizer, max_length, stride)

    train_dataloader = DataLoader(
       train_dataset, 
       batch_size=2, 
       shuffle=True, 
       collate_fn=collate_fn,
       # max_length=GPT_CONFIG_124M["context_length"],
       # stride=GPT_CONFIG_124M["context_length"]
    
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1 )
    # scheduler = CosineAnnealingLR(optimizer, T_max=2500, eta_min=1e-6) #eta_min 是**最小学习率**，必须 ≤ 初始 lr
    model.train()
    start_time = dtcn(datetime.now())
    warmup_steps = 200 #前200步warmup
    cosine_steps = 2500 - warmup_steps
    warmup_sched = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps]
    )
    print("Traing start....\n")
    for epoch in range(2):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # loss = calc_loss_batch( 
            #     step, batch, model, device 
            # ) 
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #梯度下降裁剪
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # plot_chart(loss.item(), epoch)
            if step % 10 == 0:
                print(f"Time now is ---- , {dtcn(datetime.now())} , Train Loss: {loss.item():.4f}")
                # print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
                #Grad——Norm：{model.out_head.weight.grad.norm().item()}")
                # if loss.item() < 0.4 :
                #     stop_train = True
                #     save_model()
                #     break
                print(f"========== 第 Step:{step} 轮训练完成！ TimeSpend {dtcn(datetime.now()) - start_time}==========")
    print(f"**** 第 epoch: {epoch} 轮训练彻底完成！ TimeSpend {dtcn(datetime.now()) - start_time} ****") # type: ignore

# %% [code] {"jupyter":{"outputs_hidden":false}}
# def train_data():
#     # folder_path ="/kaggle/input/datasets/mangzhang/shardstxt3/"
#     folder_path = "/kaggle/input/datasets/mangzhang/allchineseliteral-clean/"
#     # train_dataset = (
#     #     wds.WebDataset(tar_path, shardshuffle=1000) # pyright: ignore[reportAttributeAccessIssue]
#     #     .map(decode_txt)      # 👈 用 .map() 来执行自定义的解码逻辑
#     #     .to_tuple("text") 
#     # )
#     train_dataset = KaggleTextDataset(folder_path)

#     train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size=2, 
#         num_workers=0,
#         collate_fn = collate_fn
#     )

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#     model.train()
#     stop_train = False
#     for epoch in range(1):
#         if stop_train == True:
#             # save_model()
#             break
#         for step, batch in enumerate(train_dataloader):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#             optimizer.zero_grad()
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs["loss"]
#             loss.backward()
#             optimizer.step()
#             if step % 10 == 0:
#                 print("Time now is ---- ",datetime.now())
#                 print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
#                 plot_chart(loss.item(), epoch)
#                 #Grad——Norm：{model.out_head.weight.grad.norm().item()}")
#                 if loss.item() < 0.4 :
#                     stop_train = True
#                     save_model()
#                     break

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
# plotlosses = PlotLosses()

# def plot_chart(train_loss, epoch):
#     """
#     绘制损失和精度曲线。
#     :param loss: 训练损失列表
#     :param val_loss: 验证损失列表
#     :param train_acc: 训练精度列表
#     :param val_acc: 验证精度列表
#     :param epoch: 当前训练轮次
#     """
#     # 绘制损失曲线
#     plotlosses.update({
#         'train_loss': train_loss, 
#         # 'val_loss': val_loss, 
#         # 'train_acc': train_acc, 
#         # 'val_acc': val_acc,
#         'epoch': epoch
#         })
#     # 绘制精度曲线
#     plotlosses.send()
# plotlosses = PlotLosses()

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

# %% [code] {"execution":{"iopub.status.busy":"2026-11-25T01:45:57.358630Z","iopub.execute_input":"2026-11-25T01:45:57.358998Z"}}
# import os
# os.environ["HF_ENDPOINT"] = "https://huggingface.co"
# from transformers import AutoTokenizer

# from huggingface_hub import hf_hub_url

# url = hf_hub_url(
#     repo_id="Qwen/Qwen2.5-0.5B-Instruct",
#     filename="config.json"
# )
# print("实际请求URL：", url)


# tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
# tok.save_pretrained("/kaggle/working/qwen05b-tokenizer")


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
# optimizer  = {}

# 训练轮次初始化

# 训练分类器的简单训练函数

# 打印训练损失，验证损失，训练精度，验证精度， example_seen， global_step， examples_seen

############################SAVE MODEL############################
import re
def save_model():
    
 # 1. 获取当前时间戳（比如 20261113_153022）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # # 2. 把你配置里的一些关键参数也带在文件名里（可选）
    # model_size = NEW_CONFIG.get("n_layers", 12)
    # emb_dim = NEW_CONFIG.get("emb_dim", 768)
    
    # 3. 组装文件名：包含时间戳，绝对不会覆盖
    # file_name = f"allChineseLiteral_L{model_size}_E{emb_dim}_{timestamp}.pth"
    file_name = f"allChineseLiteralCN.pth"
    # 或者如果你想把原来的 CHOOSE_MODEL 恢复：
    # file_name = f"{re.sub(r'[ ()]', '', model_name)}_sft.pth"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"旧的 {file_name} 已删除。")
    else:
        print(f"{file_name} 不存在，无需删除。")
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")

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
# model=GPT_MODEL(NEW_CONFIG)
# model.to(device)
# model.load_state_dict(torch.load("allChineseLiteralCN.pth", map_location=torch.device('cpu')))
# model.eval()

# %% [code] {"jupyter":{"outputs_hidden":false}}
def generate_text_simple9(model, idx,max_new_tokens, context_size): 
    for _ in range(max_new_tokens): 
        idx_cond = idx[:, -context_size:] 
        with torch.no_grad(): 
            logits = model(idx_cond) 
        logits = logits[:, -1, :] 
        probas = torch.softmax(logits, dim=-1) 
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        idx = torch.cat((idx, idx_next), dim=1) 
    return idx

# %% [code] {"jupyter":{"outputs_hidden":false}}
# def text_to_token_ids(text, tokenizer): 
#     encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'}) 
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
#     return encoded_tensor 
# def token_ids_to_text(token_ids, tokenizer): 
#     flat = token_ids.squeeze(0) 
#     return tokenizer.decode(flat.tolist()) 
# token_ids = generate_text_simple9( 
#     model=model, 
#     idx=text_to_token_ids("孙悟空", tokenizer), 
#     max_new_tokens=25, 
#     context_size=GPT_CONFIG_124M["context_length"] 
# ) 
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %% [code] {"jupyter":{"outputs_hidden":false}}
def generate_text1(prompt, max_new_tokens=100):
    # model.eval()
    input_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>", "<|persona|>", "<|user|>", "<|bot|>"}
)
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
def generate_text2(model, prompt, max_new_tokens=150):
    model.eval()
    # 用你训练时用的 Qwen 分词器编码
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)

     # Qwen 通常有两个结束符，一个是句子结束，一个是对话轮次结束
    eos_token_id = tokenizer.eos_token_id
    # 如果你的 tokenizer 有 im_end 这个特殊 token，把它也加进来
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tokenizer.get_vocab() else None
    stop_ids = [eos_token_id]
    if im_end_id is not None:
        stop_ids.append(im_end_id)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            idx_cond = input_tensor[:, -1024:]
            outputs = model(input_ids=idx_cond)
            logits = outputs["logits"][:, -1, :] / 0.1  # 👈 温度降到 0.1，极其保守！

            # 不要随机抽样，直接拿概率最大的那个词（贪心搜索）
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

            input_tensor = torch.cat([input_tensor, next_token], dim=-1)

            # 遇到 <|im_end|> 或 <|endoftext|> 就停
              # ✅ 动态判断是否撞到了停止符
            if next_token.item() in stop_ids:
                break

    return tokenizer.decode(input_tensor[0].tolist())

# 用你训练数据中的标准 Prompt 格式（必须带 Qwen 的特殊标记）
# prompt = "<|im_start|>user\n话说唐僧<|im_end|>\n<|im_start|>assistant\n"
# print(generate_text(model, prompt))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# def generate_text_simple(model, idx, 
#                          max_new_tokens, context_size): 
#     for _ in range(max_new_tokens): 
#         idx_cond = idx[:, -context_size:] 
#         with torch.no_grad(): 
#             logits = model(idx_cond) 
#         logits = logits[:, -1, :] 
#         probas = torch.softmax(logits, dim=-1) 
#         idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
#         idx = torch.cat((idx, idx_next), dim=1) 
#     return idx

# start_context = "Hello, I am" 
# encoded = tokenizer.encode(start_context) 
# print("encoded:", encoded) 
# encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
# print("encoded_tensor.shape:", encoded_tensor.shape) 


# out = generate_text_simple( 
#     model=model, 
#     idx=encoded_tensor, 
#     max_new_tokens=6, 
#     context_size=GPT_CONFIG_124M["context_length"] 
# ) 
# print("Output:", out) 
# print("Output length:", len(out[0]))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# import torch

# def generate_safe(prompt, max_new_tokens=30, temperature=0.1, top_k=50):
#     model.eval()
#     input_ids = tokenizer.encode(prompt)
#     input_tensor = torch.tensor([input_ids]).to(device)
    
#     # 限制输入长度，防止越界
#     if input_tensor.shape[1] > 512:
#         input_tensor = input_tensor[:, -512:]
        
#     for _ in range(max_new_tokens):
#         with torch.no_grad():
#             outputs = model(input_ids=input_tensor)
#             logits = outputs["logits"]
            
#             # 1. 取最后一个词，并除以温度
#             next_token_logits = logits[:, -1, :] / temperature
            
#             # 2. Top-K 过滤
#             if top_k is not None:
#                 top_logits, _ = torch.topk(next_token_logits, top_k)
#                 next_token_logits[next_token_logits < top_logits[:, -1]] = float('-inf')
            
#             # 3. Softmax 转概率
#             probas = torch.softmax(next_token_logits, dim=-1)
            
#             # 4. 随机采样（打破死循环）
#             next_token_id = torch.multinomial(probas, num_samples=1)
            
#             input_tensor = torch.cat([input_tensor, next_token_id], dim=-1)
            
#     # 过滤掉特殊 token 再解码

#     output_ids = input_tensor[0].tolist()
#     print("Output ids: ", output_ids)
#     output_text = tokenizer.decode(output_ids)
#     return output_text

# # 测试
# print("测试1:", generate_safe("如来佛祖"))
# print("测试2:", generate_safe("金角大王"))

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
# # 在 train_lora 函数开头测试
# checkpoint = torch.load("allChineseLiteralCN.pth", map_location="cpu")
# print("权重文件的键名:", list(checkpoint.keys())[:10])
# # 在 train_lora 函数开头测试
# checkpoint = torch.load("lora_sft_weights.pth", map_location="cpu")
# print("权重文件的键名:", list(checkpoint.keys())[:10])

# %% [code] {"jupyter":{"outputs_hidden":false}}
# 确保你的环境里已经定义了 GPT_MODEL、tiktoken 的 tokenizer
# train_lora(
#     json_path="/kaggle/input/datasets/mangzhang/sf-data/sft_data.json", 
#     base_weight_path="allChineseLiteralCN.pth", 
#     tokenizer=tokenizer
# )

# %% [code] {"jupyter":{"outputs_hidden":false}}
##Train sft lora OK

import torch
import torch.nn as nn
import math
import json
import os
from torch.utils.data import Dataset, DataLoader




# ===================== 2. LoRA 核心类 =====================
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        # A 矩阵用正态分布初始化，必须非常小
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.001)
        # B 矩阵初始化为 0
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        # 冻结原始权重
        for param in self.linear.parameters():
            param.requires_grad = False
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# 递归替换模型中所有的 Linear 层
def apply_lora_to_model(model, rank=8, alpha=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        elif isinstance(module, (nn.ModuleList, nn.Sequential)):
            apply_lora_to_model(module, rank, alpha)
        else:
            apply_lora_to_model(module, rank, alpha)
    return model

# ===================== 3. Dataset 与 Collate =====================
class SFTDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        # self.allowed_special = {"<|persona|>", "<|user|>", "<|bot|>", "<|endoftext|>"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sys_msg = item.get("system", "你是一个历史专家")
        user_msg = item.get("instruction", "")
        bot_msg = item.get("output", "")

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": bot_msg},

        ]
        
        # text = f"<|persona|>{sys_msg}<|user|>{user_msg}<|bot|>{bot_msg}<|endoftext|>"
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            all_generation_prompt=False
        )
        ids = self.tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long)

def collate_fn1(batch):
    max_len = max(len(ids) for ids in batch)
    max_len = min(max_len, 512)  # 微调序列缩短到 512 防显存爆炸
    pad_id = tokenizer.pad_token_id # qwen should be 151643
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start>|")    
    padded_ids, masks, labels = [], [], []
    
    for ids in batch:
        ids = ids[:max_len]
        seq_len = len(ids)
        pad_len = max_len - seq_len
        
        padded = ids.tolist() + [pad_id] * pad_len
        mask = [1] * seq_len + [0] * pad_len
        
        label = [-100] * max_len
        
        # bot_idx = (ids == bot_token_id).nonzero()
        last_im_start = -1
        for i, token_id in enumerate(ids):
            if token_id == im_start_id:
                last_im_start = i
        if last_im_start != -1:
            for i in range(last_im_start, seq_len):
                label[i] = ids[i].item()
                
        padded_ids.append(padded)
        masks.append(mask)
        labels.append(label)
        # labels = torch.cat([labels[:, 1:],
        #             torch.full((labels.size(0), 1), -100, dtype=torch.long)], dim=1)
        
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

# ===================== 4. 训练主函数 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_lora(json_path, base_weight_path, tokenizer):
    # 1. 加载基础模型（你的 GPT_MODEL）
    model = GPT_MODEL(NEW_CONFIG)
    # 加载预训练的 50260 词表权重
    model.load_state_dict(torch.load(base_weight_path, map_location="cpu"))
    print("✅ 基础模型加载完成")

    # 2. 全量挂载 LoRA（把每一层的 Linear 都套上外挂）
    apply_lora_to_model(model, rank=8)
    model.to(device)
    model.train()

    # 3. 检查可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ LoRA 挂载完成，可训练参数量: {trainable_params:,}")

    # 4. 准备数据和优化器
    dataset = SFTDataset(json_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 极其重要：学习率必须降到 1e-5
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    num_epochs = 1
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
          
            # 防呆检查：如果全是 -100，跳过
            # if (labels != -100).sum().item() == 0:
            #     continue

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            # 强行拦截 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Step {step} Loss 变 NaN，已拦截！")
                # optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                
        print(f"========== Epoch {epoch+1} 完成，平均 Loss: {total_loss / len(dataloader):.4f} ==========")

    # 6. 保存 LoRA 权重
    lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(lora_state, "lora_sft_weights.pth")
    print("🎉 LoRA 微调完成，已保存 lora_sft_weights.pth")

# %% [code] {"jupyter":{"outputs_hidden":false}}
json_path = "/kaggle/input/datasets/mangzhang/sftconversation/sft_data.json"
# base_weight_path = "allChineseLiteralCN.pth"
base_weight_path = "/kaggle/input/datasets/mangzhang/allchineseliteralcn/allChineseLiteralCN.pth"
# train_lora(json_path,base_weight_path,tokenizer)

# %% [code] {"jupyter":{"outputs_hidden":false}}

import torch

def load_model_with_lora1(base_model_path, lora_path, config):
    # 1. 先初始化原版模型骨架
    model = GPT_MODEL(NEW_CONFIG)
    
    # 2. 加载基础大模型权重（西游记预训练权重）
    model.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    print("✅ 基础模型加载完成")
    
    # 3. 重新把模型里的层替换成带 LoRA 的层（必须和训练时结构一样）
    model = apply_lora_to_model(model, rank=8)
    
    # 4. 把微调好的 LoRA 权重“挂载”上去
    lora_weights = torch.load(lora_path, map_location="cpu")
    model.load_state_dict(lora_weights, strict=False) # strict=False 允许忽略基础权重
    print("✅ LoRA 微调权重挂载完成")
    
    # 5. 移到设备并开启测试模式
    model.to(device)
    model.eval()
    return model
# # 使用：
# config = GPT_CONFIG_124M
model = load_model_with_lora1("/kaggle/input/datasets/mangzhang/allchineseliteralcn/allChineseLiteralCN.pth", "lora_sft_weights.pth", NEW_CONFIG)

#######################################################################################
def generate_text3(model, prompt, max_new_tokens=50):
    # 编码时记得带上特殊 Token
    input_ids = tokenizer.encode(prompt, allowed_special={"<|persona|>", "<|user|>", "<|bot|>"})
    input_tensor = torch.tensor([input_ids]).to(device)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            idx_cond = input_tensor[:, -1024:]
            outputs = model(input_ids=idx_cond)
            logits = outputs["logits"][:, -1, :] / 0.7
            
            # Top-K 过滤
            top_logits, _ = torch.topk(logits, 50)
            logits[logits < top_logits[:, -1]] = float('-inf')
            
            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)
            
            if next_token.item() in (eos_endoftext_id, eos_im_end_id): # 遇到 <|endoftext|> 停止
                break
                
    return tokenizer.decode(input_tensor[0].tolist())

# 测试

# %% [code] {"jupyter":{"outputs_hidden":false}}
# prompt = "刘备是谁？"
# print(generate_text(model, prompt))

# %% [code] {"jupyter":{"outputs_hidden":false}}
import json
import torch

#######################################################################################
# 构造 messages
#######################################################################################
# 直接定义 s，一条样本
s = {
    "system": "你是熟读三国演义的助理，回答简洁准确",
    "instruction": "刘备是谁？",
    "output": "刘备，字玄德，蜀汉开国皇帝。"
}

if isinstance(s, str):
    s = json.loads(s)
if isinstance(s, list):
    s = s[0]

messages = [
    {"role": "system", "content": s["system"]},
    {"role": "user",   "content": s["instruction"]},
    {"role": "assistant", "content": s["output"]},
]

#######################################################################################

def load_model_with_lora(base_model_path, lora_path, config):
    # 1. 先初始化原版模型骨架
    model = GPT_MODEL(NEW_CONFIG)

    # 2. 加载基础大模型权重（西游记预训练权重）
    model.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    print("✅ 基础模型加载完成")

    # 3. 重新把模型里的层替换成带 LoRA 的层（必须和训练时结构一样）
    model = apply_lora_to_model(model, rank=8)

    # 4. 把微调好的 LoRA 权重"挂载"上去
    lora_weights = torch.load(lora_path, map_location="cpu")
    model.load_state_dict(lora_weights, strict=False)
    print("✅ LoRA 微调权重挂载完成")

    # 5. 移到设备并开启测试模式
    model.to(device)
    # model.eval()   # ← 你原来有这行
    return model

model = load_model_with_lora("/kaggle/input/datasets/mangzhang/allchineseliteralcn/allChineseLiteralCN.pth", "lora_sft_weights.pth", NEW_CONFIG)

#######################################################################################

def generate_text(model, prompt, max_new_tokens=500):
    model.eval()  # ✅ 关键修复：确保 dropout 关闭（防止 training=True 导致乱码）

    # 编码时记得带上特殊 Token
    input_ids = tokenizer.encode(prompt, allowed_special={"<|persona|>", "<|user|>", "<|bot|>"})
    input_tensor = torch.tensor([input_ids]).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            idx_cond = input_tensor[:, -1024:]
            outputs = model(input_ids=idx_cond)
            logits = outputs["logits"][:, -1, :] / 0.3

            # Top-K 过滤
            top_logits, _ = torch.topk(logits, 20)
            logits[logits < top_logits[:, -1]] = float('-inf')

            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas, num_samples=1)

            input_tensor = torch.cat([input_tensor, next_token], dim=-1)

            if next_token.item() == 50256:
                break

    return tokenizer.decode(input_tensor[0].tolist())

# 测试
# prompt = "<|persona|>" + s["system"] + "<|user|>" + s["instruction"] + "<|bot|>"
# prompt = "<|persona|>你熟读红楼梦,西游记,水浒传,三国演义,聊斋志异,喻世明言,警世通言,醒世恒言,初刻拍案惊奇，二刻拍案惊奇，回答尽量贴合原著，语言偏古风简洁，依据书中内容作答。<|user|>请简述孙悟空大闹天宫的经过？<|bot|>"
# result = generate_text(model, prompt, max_new_tokens=500)
# print(result)

# %% [code] {"jupyter":{"outputs_hidden":false}}
