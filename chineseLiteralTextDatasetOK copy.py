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
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

                         
#=======================================================
!pip install webdataset livelossplot
import webdataset as wds
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
from datasets import load_dataset
import logging
from rich.logging import RichHandler
from livelossplot import PlotLosses
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

# from mainScore import format_input, check_if_running, generate_model_scores, query_model, console
from rich.console import Console
console = Console()
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "12355"
# dist.init_process_group("nccl", rank=rank, world_size=world_size)
# device = torch.device(f"cuda:{rank}")
# torch.cuda.set_device(rank)
############################ MODEL CONFIG ##########################


# 2. 加载 Qwen2.5 的分词器（以 0.5B 版本为例，词表大小约 15 万）
# trust_remote_code=True 对于 Qwen 系列通常是必须的
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", 
    trust_remote_code=True
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

# 3. 调整模型嵌入层大小，容纳新 token
# NEW_CONFIG = GPT_CONFIG_124M

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(MODEL_CONFIGS[model_name])
############################ Tokenizer ##########################
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


####################################3
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
    
#################################################
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
    
#########################################
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
#########################################

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

model = GPT_MODEL(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

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
    
#############################
import torch
import csv
from torch.utils.data import Dataset, DataLoader

class SingleFileTextDataset(Dataset):
    
    def __init__(self, file_path, chunk_size=1000):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        self.chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return (self.chunks[idx],)

#######################################




###############################################
# 1. 实例化整个 Dataset
full_dataset = SingleFileTextDataset("/kaggle/working/pretrain_guji.tsv")
# full_dataset = load_dataset("flowerone/chinese-classical-corpus", "corpus", split="train")
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
################################333
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

def setup():
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return local_rank, device
    else:
        if torch.cuda.is_available():
            local_rank = 0
            device = torch.device("cuda:0")
        else:
            local_rank = 0
            device = torch.device("cpu")
        return local_rank, device

def cleanup():
    dist.destroy_process_group()

def train_dataTxt():
    IS_DDP = "RANK" in os.environ

    local_rank, device = setup()
    # device = torch.device("cuda", local_rank)

    model = GPT_MODEL(NEW_CONFIG).to(device)
    if IS_DDP:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            train_sampler = DistributedSampler(train_dataset)
            print("✅ 双卡模式运行，启用 DDP")

    else:
        train_sampler = None
        print("⚠️ 单卡模式运行，未启用 DDP")
    # train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=(train_sampler is None), # 单卡时开启打乱，多卡时由sampler管
        collate_fn=collate_fn,
        sampler=train_sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

    warmup_steps = 200
    cosine_steps = 2500 - warmup_steps
    warmup_sched = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    scaler = GradScaler()
    start_time = dtcn(datetime.now())
    print("Training start....\n")

    for epoch in range(3):
        # train_sampler.set_epoch(epoch)
        if IS_DDP:
            train_sampler.set_epoch(epoch) # 👈 只有 DDP 才需要 set_epoch
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] # 👈 从字典里抽出张量

                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), labels.flatten(), ignore_index=-100
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 10 == 0:
                print(f"Time now is ---- , {dtcn(datetime.now())} , Train Loss: {loss.item():.4f}")
                print(f"========== 第 Step:{step} 轮训练完成！ TimeSpend {dtcn(datetime.now()) - start_time}==========")

    print(f"【【【【【【【【【 第 epoch: {epoch} 轮训练彻底完成！ TimeSpend {dtcn(datetime.now()) - start_time} 】】】】】】】】】】")

    # 保存模型时的处理
    if local_rank == 0:
        state_dict = model.module.state_dict() if IS_DDP else model.state_dict()
        torch.save(state_dict, "/kaggle/working/hong_lou_meng.pth")
        print("============= Saved pth to hong_lou_meng.pth ===============")

     # 清理
    if IS_DDP:
        cleanup()
        
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
########################################33
model=GPT_MODEL(NEW_CONFIG)
model.to(device)
model.load_state_dict(torch.load("/kaggle/working/allChineseLiteralCN.pth", map_location=torch.device('cpu')))
model.eval()


import torch
import torch.nn as nn
import math
import json
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# ===================== 2. LoRA 核心类 =====================
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.001)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# 【修复】不会重复嵌套LoRA的递归替换
def apply_lora_to_model(model, rank=8, alpha=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        elif isinstance(module, (nn.ModuleList, nn.Sequential)):
            apply_lora_to_model(module, rank, alpha)
        # 普通block，遍历子模块，但是不无脑递归所有模块，防止重复套LoRA
        elif isinstance(module, nn.Module):
            apply_lora_to_model(module, rank, alpha)
    return model


# ===================== 3. Dataset 与 Collate =====================
class SFTDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

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
        # 【修复参数名】
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        ids = self.tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    max_len = max(len(ids) for ids in batch)
    max_len = min(max_len, 512)
    pad_id = tokenizer.pad_token_id
    im_start_id = 151644
    im_end_id = 151645
    newline_id = 198

    padded_ids, masks, labels = [], [], []

    for ids in batch:
        ids = ids[:max_len]
        seq_len = len(ids)
        pad_len = max_len - seq_len

        # padded = ids.tolist() + [pad_id] * pad_len  # Text 文本数据dataset用这个
        padded = ids.tolist() + [pad_id] * pad_len   #tsv dataset

        mask = [1] * seq_len + [0] * pad_len
        label = [-100] * max_len

        i = 0
        while i < seq_len:
            if ids[i] == im_start_id:
                j = i + 1
                while j < seq_len and ids[j] != im_end_id:
                    j += 1
                if j >= seq_len:
                    i += 1
                    continue
                block_ids = ids[i+1:j]
                split_pos = -1
                for idx, tid in enumerate(block_ids):
                    if tid == newline_id:
                        split_pos = idx
                        break
                if split_pos == -1:
                    i = j + 1
                    continue
                role_part = block_ids[:split_pos]
                role_str = tokenizer.decode(role_part).strip()
                if role_str == "assistant":
                    content_start = i + 1 + split_pos + 1
                    for k in range(content_start, j):
                        label[k] = ids[k].item()
                i = j + 1
            else:
                i += 1

        has_valid = any(x != -100 for x in label)
        if not has_valid:
            print("⚠️ 本条样本 label 全部 -100")
            print(tokenizer.decode(ids))

        padded_ids.append(padded)
        masks.append(mask)
        labels.append(label)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


#-------------------------------------------------------------------
# ===================== 4. 训练主函数 =====================

def train_lora(json_path, base_weight_path, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT_MODEL(NEW_CONFIG)
    model.load_state_dict(torch.load(base_weight_path, map_location="cpu"))
    print("✅ 基础模型加载完成")

    apply_lora_to_model(model, rank=8)
    model.to(device)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ LoRA 挂载完成，可训练参数量: {trainable_params:,}")

    dataset = SFTDataset(json_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # ========== 测试阶段建议：注释scheduler，固定lr，排除调度干扰 ==========
    # scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-7)
    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 自回归移位 + 防空
            if input_ids.size(1) <= 1:
                continue
            # input_ids = input_ids[:, :-1]
            # labels = labels[:, 1:]
            # attention_mask = attention_mask[:, :-1]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Step {step} Loss 变 NaN，已拦截！")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler.step()

            total_loss += loss.item()

            # 【修复】单步梯度norm，不是累加！
            # grad_norm = 0.0
            # for p in model.parameters():
            #     if p.requires_grad and p.grad is not None:
            #         grad_norm += p.grad.norm() ** 2
            # grad_norm = grad_norm.sqrt().item()
            # print(f"grad norm: {grad_norm:.6f}")

            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"========== Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f} ==========")

    lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(lora_state, "lora_sft_weights.pth")
    print("🎉 LoRA 微调完成，已保存 lora_sft_weights.pth")

import json
import torch

#######################################################################################
# 构造 messages
#######################################################################################
# 直接定义 s，一条样本
s = {
    "system": "你是熟读红楼梦的助理，回答简洁准确",
    "instruction": "贾宝玉是谁？",
    "output": "红楼梦主角。"
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
    return model


#######################################################################################

def generate_text(model, prompt, max_new_tokens=500):
    model = load_model_with_lora("/kaggle/working/hongloumeng.pth/hong_lou_meng.pth", "lora_sft_weights.pth", NEW_CONFIG)
    model.eval()

    # Qwen ChatML 编码，不再用旧的 <|persona|> <|user|> <|bot|>
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    eos_id = 151645

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

            if next_token.item() == eos_id:
                break

    return tokenizer.decode(input_tensor[0].tolist())

#========== 重点：改成 Qwen ChatML 模板 ==========
prompt = "<|im_start|>system\n你熟读红楼梦,西游记,水浒传,三国演义,聊斋志异,喻世明言,警世通言,醒世恒言,初刻拍案惊奇，二刻拍案惊奇，回答尽量贴合原著，语言偏古风简洁，依据书中内容作答。<|im_end|>\n<|im_start|>user\n请简述孙悟空大闹天宫的经过？<|im_end|>\n<|im_start|>assistant\n"
prompt = "<|im_start|>system\n你熟读红楼梦，据书中内容作答。<|im_end|>\n<|im_start|>user\贾宝玉是什么人？<|im_end|>\n<|im_start|>assistant\n"

# result = generate_text(model, prompt, max_new_tokens=200)
# print(result)

import torch
import torch.nn.functional as F

def test_generate(model, tokenizer, device):
    model.eval()
    prompt_text = """<|im_start|>system
你熟读史记，据书中内容作答。
<|im_end|>
<|im_start|>user
司马迁是什么人？
<|im_end|>
<|im_start|>assistant
"""
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids]).to(device)
    eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    max_new = 128
    temperature = 0.4
    top_k = 10

    with torch.no_grad():
        for _ in range(max_new):
            outputs = model(input_tensor)
            logits = outputs["logits"]
            next_logit = logits[:, -1, :] / temperature

            # top-k
            values, indices = torch.topk(next_logit, top_k)
            mask = torch.full_like(next_logit, -float("inf"))
            mask.scatter_(1, indices, values)
            probs = F.softmax(mask, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == eos_id:
                break
            input_tensor = torch.cat([input_tensor, next_id], dim=-1)
    out_text = tokenizer.decode(input_tensor[0].tolist())
    print("\n===== 生成结果 =====")
    print(out_text)
    return out_text

# result = test_generate(model, tokenizer, device)
# print("Result: ", result)

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
    file_name = f"allChineseLiteralCNgj.pth"
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
    # logger.info("Starting the prepare data...")
    # prepare_data()
    console.print("Training data...", style='blue')

    train_dataTxt()
    logger.info("Starting the save model...")
    save_model()
    json_path = "/kaggle/input/datasets/mangzhang/sftconversation2/sft_data.json"
    base_weight_path = "allChineseLiteralCN.pth"
    # base_weight_path = "/kaggle/input/datasets/mangzhang/allchineseliteralcn/allChineseLiteralCN.pth"
    # train_lora(json_path,base_weight_path,tokenizer)

    # train_lora(json_path,base_weight_path,tokenizer)
    # train_lora_sft = train_lora_sft()
    # evaluate_lora()


    # generate_model_scores(test_data, "model_response")