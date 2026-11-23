%%writefile saveSessionDDP.py
# !pip list webdataset > /dev/null 2>&1 || pip install webdataset
# !pip list livelossplot > /dev/null 2>&1 || pip install livelossplot
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import kagglehub
import torch, gc
gc.collect()
torch.cuda.empty_cache()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# import webdataset as wds
import json
import re
import torch.nn as nn
import pandas as pd
import torch
import glob
from datetime import datetime
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from datasets import load_dataset
import logging
from rich.logging import RichHandler
# from livelossplot import PlotLosses
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from accelerate import Accelerator
accelerator = Accelerator()

from rich.console import Console
console = Console()

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True
)
eos_endoftext_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
eos_im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
qwen_vocab_size = len(tokenizer)

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
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(MODEL_CONFIGS[model_name])

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
CN_TZ = ZoneInfo("Asia/Shanghai")
def dtcn(dt):
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt, tz=timezone.utc)
    elif isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(CN_TZ)
print(dtcn(datetime.now()))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0)
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
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
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
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, attention_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

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

model = GPT_MODEL(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def collate_fn(batch):
    texts = [item[0].strip() for item in batch]
    batch_enc = tokenizer(
        texts,
        truncation=True,
        truncation_side="left",
        max_length=1024,
        padding=True,
        return_tensors="pt"
    )
    input_ids = batch_enc["input_ids"]
    attention_mask = batch_enc["attention_mask"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

class SingleFileTextDataset(Dataset):
    def __init__(self, file_path, chunk_size=1000):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        self.chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        return (self.chunks[idx],)

full_dataset = SingleFileTextDataset("/kaggle/input/datasets/mangzhang/pretrain/pretrain_body_clean.txt")
total_size = len(full_dataset)
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)
train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(full_dataset, range(train_size + val_size, total_size))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

def setup():
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # dist.init_process_group(backend="nccl")
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
    if dist.is_initialized():
        dist.destroy_process_group()

def train_dataTxt():
    IS_DDP = "RANK" in os.environ
    local_rank, device = setup()
    model = GPT_MODEL(NEW_CONFIG).to(device)
    if IS_DDP:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            train_sampler = DistributedSampler(train_dataset)
            print("✅ 双卡模式运行，启用 DDP")
    else:
        train_sampler = None
        print("⚠️ 单卡模式运行，未启用 DDP")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=(train_sampler is None),
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
    for epoch in range(4):
        if IS_DDP:
            train_sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
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
    if local_rank == 0:
        state_dict = model.module.state_dict() if IS_DDP else model.state_dict()
        torch.save(state_dict, "/kaggle/working/allChineseLiteralCNNormal.pth")
        print("============= Saved pth to hong_lou_meng.pth ===============")
    if IS_DDP:
        cleanup()

logger = logging.getLogger("mainFineTune")
logger.setLevel(logging.INFO)
console_handler = RichHandler(show_time=True, rich_tracebacks=True)
console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def save_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"allChineseLiteralCNAcc.pth"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"旧的 {file_name} 已删除。")
    else:
        print(f"{file_name} 不存在，无需删除。")

    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), file_name)


    # torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")
    logger.info(f"Model saved to {file_name}")

if __name__ == '__main__':
    console.print("Base Modal Training start...", style='blue')
    train_dataTxt()
    logger.info("Base Modal Starting the save model...")
    save_model()
