
import numpy as np
import pandas as pd
import os
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
from torch.optim.lr_scheduler import CosineAnnealingLR

console = Console()
TF_ENABLE_ONEDNN_OPTS=0

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
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # pyright: ignore[reportCallIssue]
        
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
    def forward(self,  input_ids, attention_mask=None, labels=None):
        batch_size, seq_len= input_ids.shape
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=input_ids.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, attention_mask)
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

# def create_dataloader(txt, batch_size=2, max_length=256,
#                     stride=128, shuffle=True, drop_last=True,
#                     num_workers=0):
#     tokenizer = tiktoken.get_encoding("gpt2")
#     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,
#         num_workers=num_workers
#     )
#     return dataloader

# with open("/kaggle/input/datasets/mangzhang/the-verdict/the-verdict.txt") as f:
#     txt = f.read()
# dataloader = create_dataloader(
#     txt, batch_size=2, max_length=1024, stride=1024, shuffle=False
# )
# data_iter = iter(dataloader)
# first_batch = next(data_iter)

def decode_txt(sample):
    if "text" in sample:
        sample["text"] = sample["text"].decode("utf-8")
    return sample
model = GPT_MODEL(NEW_CONFIG)
model.eval()
model.to(device)


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
    labels = torch.cat([labels[:, 1:],
                    torch.full((labels.size(0), 1), -100, dtype=torch.long)], dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
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

full_dataset = SingleFileTextDataset("D:/Project/AI/ZeroToTransformer/pretrain_body_clean.txt")
total_size = len(full_dataset)

train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(full_dataset, range(train_size + val_size, total_size))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

def train_dataTxt():
    # folder_path = "/kaggle/input/datasets/mangzhang/allchineseliteral-clean/"
    # with open("/kaggle/input/datasets/mangzhang/allchineseliteral-clean/allChineseLiteral_clean.txt","r",encoding="utf-8") as f:
    #     text_data = f.read()

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
    print(f"========== 第 epoch: {epoch} 轮训练彻底完成！ ==========") # type: ignore

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

# plotlosses = PlotLosses()
# def plot_chart(train_loss, epoch):
#     plotlosses.update({
#         'train_loss': train_loss,
#         'epoch': epoch
#         })
#     plotlosses.send()

def calc_loss_batch(input_batch, target_batch, model, device):
    pass

def calc_loss_loader(data_loader, model, device, num_batches=None):
    pass

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    pass

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    pass

optimizer  = {}

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

if __name__ == '__main__':
    status = "Training"
    logger.info("Starting the train data...")
    console.print("Training data...", style='blue')
    train_dataTxt()
    logger.info("Starting the save model...")
    save_model()    
    if status == "Training":
        pass
    else:
        pass


############################## TEST #############################################
model=GPT_MODEL(NEW_CONFIG)
model.to(device)
model.load_state_dict(torch.load("allChineseLiteralCN.pth"))
model.eval()

def generate_text(prompt, max_new_tokens=100):
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

print(generate_text("东土大唐"))

def generate_safe(prompt, max_new_tokens=30, temperature=0.1, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    if input_tensor.shape[1] > 512:
        input_tensor = input_tensor[:, -512:]
        
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            logits = outputs["logits"]
            
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                top_logits, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_logits[:, -1]] = float('-inf')
            
            probas = torch.softmax(next_token_logits, dim=-1)
            
            next_token_id = torch.multinomial(probas, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token_id], dim=-1)
            
    output_ids = input_tensor[0].tolist()
    print("Output ids: ", output_ids)
    output_text = tokenizer.decode(output_ids)
    return output_text

print("测试1:", generate_safe("如来佛祖"))
print("测试2:", generate_safe("金角大王"))

def test_generate(prompt, max_tokens=30):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            idx_cond = input_tensor[:, -512:]
            outputs = model(input_ids=idx_cond)
            logits = outputs["logits"][:, -1, :] / 0.7
            
            top_logits, _ = torch.topk(logits, 50)
            min_val = top_logits[:, -1]
            logits[logits < min_val] = float('-inf')
            
            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)
            
    output_ids = input_tensor[0].tolist()
    clean_ids = [i for i in output_ids if i < 50256]
    return tokenizer.decode(clean_ids)

print("输出:", test_generate("孙悟空和猪八戒的关系是： "))
