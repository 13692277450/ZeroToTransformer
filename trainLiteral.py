import os, re, math, json
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

#######################################################################################
# 路径 & 配置
#######################################################################################
ROOT = r"D:\Project\AI\ZeroToTransformer"
CLEAN_PATH = os.path.join(ROOT, "pretrain_body_clean.txt")
OUT_DIR = os.path.join(ROOT, "ckpt"); os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0),
        f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

CTX     = min(1024, NEW_CONFIG["context_length"])
STRIDE  = 512        # 50% overlap → 样本翻倍
BS      = 4
ACCUM   = 8          # 等效 batch 32
EPOCHS  = 12
LR      = 3e-4       # ★ 关键:从 1e-5 提到 3e-4
WARMUP  = 400
WD      = 0.1
SAVE_EVERY = 500
USE_AMP = (device.type == "cuda")

#######################################################################################
# 读语料 + 最后清理
#######################################################################################
text = open(CLEAN_PATH, encoding="utf-8").read()
text = re.sub(r"[，,]?[^。\n]{0,15}著介绍[：:]", "", text)
text = re.sub(r"^[，。：:、；]+", "", text).strip()

n_tok_all = len(tokenizer.encode(text))
print(f"字数 {len(text):,} | token {n_tok_all:,} | 比值 {n_tok_all/len(text):.2f}")

#######################################################################################
# Dataset
#######################################################################################
class PretrainDataset(Dataset):
    def __init__(self, text, ctx=CTX, stride=STRIDE):
        ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.chunks = [torch.tensor(ids[i:i+ctx+1], dtype=torch.long)
                       for i in range(0, len(ids)-ctx-1, stride)]
        print(f"样本数 {len(self.chunks):,} (ctx={ctx}, stride={stride})")
    def __len__(self): return len(self.chunks)
    def __getitem__(self, i):
        c = self.chunks[i]
        return {"input_ids": c[:-1], "labels": c[1:],
                "attention_mask": torch.ones_like(c[:-1])}

def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

ds = PretrainDataset(text)
dl = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collate,
                drop_last=True, num_workers=0)   # Windows 建议 0

steps_per_epoch = len(dl) // ACCUM
total_steps = EPOCHS * steps_per_epoch
print(f"每 epoch opt-step {steps_per_epoch} | 总 opt-step {total_steps}")
print(f"预计见 token 总量 ≈ {total_steps*BS*ACCUM*CTX/1e6:.0f} M")

#######################################################################################
# scheduler / checkpoint
#######################################################################################
def sched_fn(step):
    if step < WARMUP: return (step+1)/WARMUP
    p = min(1.0, (step-WARMUP)/max(1, total_steps-WARMUP))
    return 0.1 + 0.9*0.5*(1+math.cos(math.pi*p))

def save_ckpt(model, opt, sch, ep, gs, hist, tag="last"):
    torch.save({"epoch":ep, "global_step":gs,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":opt.state_dict(),
                "scheduler_state_dict":sch.state_dict(),
                "config":NEW_CONFIG, "history":hist},
               os.path.join(OUT_DIR, f"pretrain_ckpt_{tag}.pth"))
    torch.save(model.state_dict(),
               os.path.join(OUT_DIR, f"allChineseLiteralCN_{tag}.pth"))
    print(f"💾 {tag} @ step {gs}")

@torch.no_grad()
def probe(model, n=100):
    model.eval()
    for p in ["话说天下大势，", "滚滚长江东逝水，", "冤屈难伸，只见"]:
        print(f"  {p} → {generate_text(model, p, max_new_tokens=n, greedy=True)}")
    model.train()

#######################################################################################
# 训练
#######################################################################################
def train(resume=None, load_old_base=True):
    model = GPT_MODEL(NEW_CONFIG)
    if load_old_base:
        p = os.path.join(ROOT, "allChineseLiteralCN.pth")
        model.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
        print("✅ 载入旧 base 继续训")

    # 数据量有限 → 关 dropout,利于 memorize
    model.drop_emb.p = 0.0
    for blk in model.trf_blocks:
        for m in blk.modules():
            if isinstance(m, torch.nn.Dropout): m.p = 0.0
    print("✅ dropout 已置 0")

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9,0.95))
    sch = LambdaLR(opt, sched_fn)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    ep0, gs, hist = 0, 0, []
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sch.load_state_dict(ck["scheduler_state_dict"])
        ep0, gs, hist = ck["epoch"], ck["global_step"], ck.get("history", [])
        print(f"▶️ resume ep{ep0} step{gs}")

    t0 = datetime.now()
    for ep in range(ep0, EPOCHS):
        model.train(); tot, cnt = 0., 0
        opt.zero_grad(set_to_none=True)

        for i, b in enumerate(dl):
            with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
                out = model(input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                            labels=b["labels"].to(device))
                loss = out["loss"] / ACCUM

            scaler.scale(loss).backward()

            if (i+1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                sch.step(); opt.zero_grad(set_to_none=True)
                gs += 1

                l = loss.item()*ACCUM
                hist.append(l)
                if gs % 20 == 0:
                    el = (datetime.now()-t0).total_seconds()/60
                    print(f"[{el:6.1f}min] ep{ep} gs{gs}/{total_steps} "
                          f"loss {l:.4f} lr {sch.get_last_lr()[0]:.2e}")
                if gs % SAVE_EVERY == 0:
                    save_ckpt(model,opt,sch,ep,gs,hist,"last")
                    probe(model)

            tot += loss.item()*ACCUM; cnt += 1

        print(f"===== epoch {ep} avg loss {tot/cnt:.4f} =====")
        save_ckpt(model,opt,sch,ep+1,gs,hist,f"ep{ep}")
        probe(model, n=150)

    save_ckpt(model,opt,sch,EPOCHS,gs,hist,"final")
    return model, hist

base_model, hist = train(load_old_base=True)
