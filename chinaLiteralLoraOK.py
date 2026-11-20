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
import torch, gc
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
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import logging
from rich.logging import RichHandler
from livelossplot import PlotLosses
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from rich.console import Console
console = Console()
json_path = "/kaggle/input/datasets/mangzhang/sf-data/sft_data.json"
base_weight_path = "allChineseLiteralCN.pth"
############################ MODEL CONFIG ##########################
GPT_CONFIG_124M = {
    "model": "gpt2",
    "context_length": 1024,
    "n_layers": 12,
    "n_heads": 12,
    "emb_dim": 768,
    "drop_rate": 0.1,
    "temperature": 0.7,
    "vocab_size": 50260,
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
base_enc = tiktoken.get_encoding("gpt2")
new_special_tokens = {
    "<|persona|>": 50257,
    "<|user|>": 50258,
    "<|bot|>": 50259
}
tokenizer = tiktoken.Encoding(
    name="gpt2-custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={
        **base_enc._special_tokens,
        **new_special_tokens
    }
)

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(MODEL_CONFIGS[model_name])
############################ Tokenizer ##########################
class SimpleTokenizer():
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = "".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text)
        return text
############################ Tokenizer Gpt2##########################
print(tokenizer.encode("<|bot|>", allowed_special={"<|bot|>"})) 

############################ FeedForward ############################
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
                    "labels 全部为 -100；请检查 <|bot|> token ID 和 collate_fn"
                )
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

############################ DATASET ##########################
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

with open("/kaggle/input/datasets/mangzhang/the-verdict/the-verdict.txt") as f:
    txt = f.read()
dataloader = create_dataloader(
    txt, batch_size=2, max_length=1024, stride=1024, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)

############################ WebDataset ############################
def create_text_webdataset(file_path, output_dir, chunk_size = 2000,max_shards_size=10000):
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "literal-%06d.tar")
    sample_idx =0
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
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
        
    with wds.ShardWriter(output_pattern, maxcount=max_shards_size) as sink:
        for block in pieces:
            if len(block) < 50:
                continue
            sample = {
                "__key__": f"chunk_{sample_idx:08d}",
                "text": block.encode("utf-8")
            }
            sink.write(sample)
            sample_idx += 1
    print(f"Total samples blocks: {sample_idx}")

############################Prepare Data############################
def decode_txt(sample):
    if "text" in sample:
        sample["text"] = sample["text"].decode("utf-8")
    return sample
def prepare_data():
    file_path = "D:\\Project\\AI\\ZeroToTransformer\\allchineseliteral_clean.txt"
    create_text_webdataset(file_path,  "shardsTxt", max_shards_size=10000)

############################ MODEL#################################
model = GPT_MODEL(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

############################Train data ############################
class KaggleTextDataset(Dataset):
    def __init__(self, folder_path):
        folder_path = "/kaggle/input/datasets/mangzhang/the-verdict/"
        self.file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
        if not self.file_paths:
            raise FileNotFoundError(f"在 {folder_path} 下没有找到任何 .text 文件，请检查路径！")
        print(f"✅ 成功找到 {len(self.file_paths)} 个文本文件")
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            text = f.read()
        return (text,)

def collate_fn(batch):
    texts = [item[0].strip() for item in batch]
    encoded_texts = [tokenizer.encode(text, allowed_special={"<|persona|>", "<|user|>", "<|bot|>"})[:1024] for text in texts]
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
    labels = torch.cat([labels[:, 1:],
                    torch.full((labels.size(0), 1), -100, dtype=torch.long)], dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

import torch
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

class SingleFileTextDataset2(Dataset):
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

full_dataset = SingleFileTextDataset("/kaggle/input/datasets/mangzhang/allchineseliteral-clean/allChineseLiteral_clean.txt")
total_size = len(full_dataset)

train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(full_dataset, range(train_size + val_size, total_size))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

from torch.optim.lr_scheduler import CosineAnnealingLR
def train_dataTxt():
    folder_path = "/kaggle/input/datasets/mangzhang/allchineseliteral-clean/"
    with open("/kaggle/input/datasets/mangzhang/allchineseliteral-clean/allChineseLiteral_clean.txt","r",encoding="utf-8") as f:
        text_data = f.read()

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    model.train()
    stop_train = False
    for epoch in range(2):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 10 == 0:
                print("Time now is ---- ",datetime.now())
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
                print(f"========== 第 Step:{step} batch: {batch} 轮训练完成！ ==========")
    print(f"========== 第 epoch: {epoch} 轮训练彻底完成！ ==========")


############################Log status############################
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

#########################Simple Plot chart#########################
plotlosses = PlotLosses()

def plot_chart(train_loss, epoch):
    plotlosses.update({
        'train_loss': train_loss, 
        'epoch': epoch
        })
    plotlosses.send()
plotlosses = PlotLosses()

############################计算损失批次############################
def calc_loss_batch(input_batch, target_batch, model, device):
    pass

############################计算损失加载器############################
def calc_loss_loader(data_loader, model, device, num_batches=None):
    pass

############################训练分类检验############################
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    pass

############################评估模式############################
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    pass

############################优化器设计############################
optimizer  = {}

############################SAVE MODEL############################
import re
def save_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_size = NEW_CONFIG.get("n_layers", 12)
    emb_dim = NEW_CONFIG.get("emb_dim", 768)
    file_name = f"allChineseLiteralCN.pth"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"旧的 {file_name} 已删除。")
    else:
        print(f"{file_name} 不存在，无需删除。")
    torch.save(model.state_dict(), file_name)
    logger.info(f"Model saved to {file_name}")

##########################MAIN############################
if __name__ == '__main__':
    status = "Training"
    logger.info("Starting the train data...")
    console.print("Training data...", style='blue')
    logger.info("Starting the save model...")
    
    if status == "Training":
        pass
    else:   
        pass

def generate_text(prompt, max_new_tokens=100):
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

system_msg = "<|persona|>你是孙悟空。"
user_msg = "<|user|>话说唐僧"
usertext = f"{system_msg}{user_msg}<|bot|><|endoftext|>"
print(generate_text(usertext))

import torch
import torch.nn as nn
import math
import json
import os
from torch.utils.data import Dataset, DataLoader

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

def apply_lora_to_model(model, rank=8, alpha=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        elif isinstance(module, (nn.ModuleList, nn.Sequential)):
            apply_lora_to_model(module, rank, alpha)
        else:
            apply_lora_to_model(module, rank, alpha)

class SFTDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.allowed_special = {"<|persona|>", "<|user|>", "<|bot|>", "<|endoftext|>"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sys_msg = item.get("system", "你是一个智能助手")
        user_msg = item.get("instruction", "")
        bot_msg = item.get("output", "")
        
        text = f"<|persona|>{sys_msg}<|user|>{user_msg}<|bot|>{bot_msg}<|endoftext|>"
        ids = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    max_len = max(len(ids) for ids in batch)
    max_len = min(max_len, 512)
    pad_id = 50256
    bot_token_id = 50259
    
    padded_ids, masks, labels = [], [], []
    
    for ids in batch:
        ids = ids[:max_len]
        seq_len = len(ids)
        pad_len = max_len - seq_len
        
        padded = ids.tolist() + [pad_id] * pad_len
        mask = [1] * seq_len + [0] * pad_len
        
        label = [-100] * max_len
        bot_idx = (ids == bot_token_id).nonzero()
        if len(bot_idx) > 0:
            start_idx = bot_idx[0].item()
            for i in range(start_idx, seq_len):
                label[i] = ids[i].item()
                
        padded_ids.append(padded)
        masks.append(mask)
        labels.append(label)
        
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

def train_lora(json_path, base_weight_path, tokenizer):
    model = GPT_MODEL(GPT_CONFIG_124M)
    model.load_state_dict(torch.load(base_weight_path, map_location="cpu"))
    print("✅ 基础模型加载完成")

    apply_lora_to_model(model, rank=8)
    model.to(device)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ LoRA 挂载完成，可训练参数量: {trainable_params:,}")

    dataset = SFTDataset(json_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Step {step} Loss 变 NaN，已拦截！")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                
        print(f"========== Epoch {epoch+1} 完成，平均 Loss: {total_loss / len(dataloader):.4f} ==========")

    lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(lora_state, "lora_sft_weights.pth")
    print("🎉 LoRA 微调完成，已保存 lora_sft_weights.pth")
    
if __name__ == "__main__":
    train_lora(json_path, base_weight_path, tokenizer)
