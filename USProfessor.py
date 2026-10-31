# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:39.865050Z","iopub.execute_input":"2026-10-31T07:11:39.865612Z","iopub.status.idle":"2026-10-31T07:11:39.876962Z","shell.execute_reply.started":"2026-10-31T07:11:39.865575Z","shell.execute_reply":"2026-10-31T07:11:39.875863Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:39.891226Z","iopub.execute_input":"2026-10-31T07:11:39.891690Z","iopub.status.idle":"2026-10-31T07:11:40.237578Z","shell.execute_reply.started":"2026-10-31T07:11:39.891660Z","shell.execute_reply":"2026-10-31T07:11:40.236146Z"},"jupyter":{"outputs_hidden":false}}
import urllib.request 
url = ("https://raw.githubusercontent.com/rasbt/" 
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/" 
       "the-verdict.txt") 
file_path = "the-verdict.txt"
resulturl = urllib.request.urlretrieve(url, file_path)
print(file_path[:50])
print(resulturl)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:40.239949Z","iopub.execute_input":"2026-10-31T07:11:40.240416Z","iopub.status.idle":"2026-10-31T07:11:41.403105Z","shell.execute_reply.started":"2026-10-31T07:11:40.240375Z","shell.execute_reply":"2026-10-31T07:11:41.401676Z"},"jupyter":{"outputs_hidden":false}}
!ls -l

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.404993Z","iopub.execute_input":"2026-10-31T07:11:41.405404Z","iopub.status.idle":"2026-10-31T07:11:41.414929Z","shell.execute_reply.started":"2026-10-31T07:11:41.405370Z","shell.execute_reply":"2026-10-31T07:11:41.413646Z"},"jupyter":{"outputs_hidden":false}}
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.418037Z","iopub.execute_input":"2026-10-31T07:11:41.419651Z","iopub.status.idle":"2026-10-31T07:11:41.444179Z","shell.execute_reply.started":"2026-10-31T07:11:41.419608Z","shell.execute_reply":"2026-10-31T07:11:41.442998Z"},"jupyter":{"outputs_hidden":false}}
import re
text = "Hello, world, This, is a text."
result = re.split(r'(\s)', text)
print(result)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.445912Z","iopub.execute_input":"2026-10-31T07:11:41.446265Z","iopub.status.idle":"2026-10-31T07:11:41.468809Z","shell.execute_reply.started":"2026-10-31T07:11:41.446234Z","shell.execute_reply":"2026-10-31T07:11:41.467542Z"},"jupyter":{"outputs_hidden":false}}
result= re.split(r'([,.]|\s)',text)
print(result)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.470323Z","iopub.execute_input":"2026-10-31T07:11:41.470689Z","iopub.status.idle":"2026-10-31T07:11:41.497013Z","shell.execute_reply.started":"2026-10-31T07:11:41.470654Z","shell.execute_reply":"2026-10-31T07:11:41.495993Z"},"jupyter":{"outputs_hidden":false}}
result = [item for item in result if item.strip()]
print(result)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.498389Z","iopub.execute_input":"2026-10-31T07:11:41.498708Z","iopub.status.idle":"2026-10-31T07:11:41.521959Z","shell.execute_reply.started":"2026-10-31T07:11:41.498674Z","shell.execute_reply":"2026-10-31T07:11:41.520562Z"},"jupyter":{"outputs_hidden":false}}
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)',text)
result = [item.strip() for item in result if item.strip()]
print(result)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.523822Z","iopub.execute_input":"2026-10-31T07:11:41.524358Z","iopub.status.idle":"2026-10-31T07:11:41.553112Z","shell.execute_reply.started":"2026-10-31T07:11:41.524315Z","shell.execute_reply":"2026-10-31T07:11:41.551967Z"},"jupyter":{"outputs_hidden":false}}
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.554788Z","iopub.execute_input":"2026-10-31T07:11:41.555275Z","iopub.status.idle":"2026-10-31T07:11:41.575577Z","shell.execute_reply.started":"2026-10-31T07:11:41.555234Z","shell.execute_reply":"2026-10-31T07:11:41.574500Z"},"jupyter":{"outputs_hidden":false}}
print(preprocessed[:30])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.582187Z","iopub.execute_input":"2026-10-31T07:11:41.582791Z","iopub.status.idle":"2026-10-31T07:11:41.600755Z","shell.execute_reply.started":"2026-10-31T07:11:41.582754Z","shell.execute_reply":"2026-10-31T07:11:41.599425Z"},"jupyter":{"outputs_hidden":false}}
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.602151Z","iopub.execute_input":"2026-10-31T07:11:41.602737Z","iopub.status.idle":"2026-10-31T07:11:41.627012Z","shell.execute_reply.started":"2026-10-31T07:11:41.602706Z","shell.execute_reply":"2026-10-31T07:11:41.625926Z"},"jupyter":{"outputs_hidden":false}}
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()): #解包
    print(item)
    if i >= 50:
        break

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.628485Z","iopub.execute_input":"2026-10-31T07:11:41.628909Z","iopub.status.idle":"2026-10-31T07:11:41.652586Z","shell.execute_reply.started":"2026-10-31T07:11:41.628874Z","shell.execute_reply":"2026-10-31T07:11:41.651280Z"},"jupyter":{"outputs_hidden":false}}
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)',text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        #实场景需要 <|unk|> token 或 self.str_to_int.get(s, unk_id)。
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i]for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.654119Z","iopub.execute_input":"2026-10-31T07:11:41.654594Z","iopub.status.idle":"2026-10-31T07:11:41.680180Z","shell.execute_reply.started":"2026-10-31T07:11:41.654554Z","shell.execute_reply":"2026-10-31T07:11:41.679103Z"},"jupyter":{"outputs_hidden":false}}
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.681548Z","iopub.execute_input":"2026-10-31T07:11:41.681887Z","iopub.status.idle":"2026-10-31T07:11:41.709663Z","shell.execute_reply.started":"2026-10-31T07:11:41.681859Z","shell.execute_reply":"2026-10-31T07:11:41.708473Z"},"jupyter":{"outputs_hidden":false}}
print(tokenizer.decode(ids))
print(repr(tokenizer.decode(ids)))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.711405Z","iopub.execute_input":"2026-10-31T07:11:41.711804Z","iopub.status.idle":"2026-10-31T07:11:41.738437Z","shell.execute_reply.started":"2026-10-31T07:11:41.711774Z","shell.execute_reply":"2026-10-31T07:11:41.737294Z"},"jupyter":{"outputs_hidden":false}}
text = "Hello, do you like tea"
# print(tokenizer.encode(text))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.739740Z","iopub.execute_input":"2026-10-31T07:11:41.740097Z","iopub.status.idle":"2026-10-31T07:11:41.766486Z","shell.execute_reply.started":"2026-10-31T07:11:41.740071Z","shell.execute_reply":"2026-10-31T07:11:41.765222Z"},"jupyter":{"outputs_hidden":false}}
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.767996Z","iopub.execute_input":"2026-10-31T07:11:41.768396Z","iopub.status.idle":"2026-10-31T07:11:41.794045Z","shell.execute_reply.started":"2026-10-31T07:11:41.768355Z","shell.execute_reply":"2026-10-31T07:11:41.792950Z"},"jupyter":{"outputs_hidden":false}}
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.795668Z","iopub.execute_input":"2026-10-31T07:11:41.796023Z","iopub.status.idle":"2026-10-31T07:11:41.819211Z","shell.execute_reply.started":"2026-10-31T07:11:41.795993Z","shell.execute_reply":"2026-10-31T07:11:41.817994Z"},"jupyter":{"outputs_hidden":false}}
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)',text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int 
                       else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'([,.?_!"()\']|--|\s)',r'\1',text)
        return text

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.820669Z","iopub.execute_input":"2026-10-31T07:11:41.821085Z","iopub.status.idle":"2026-10-31T07:11:41.854077Z","shell.execute_reply.started":"2026-10-31T07:11:41.821042Z","shell.execute_reply":"2026-10-31T07:11:41.853004Z"},"jupyter":{"outputs_hidden":false}}
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1,text2))
print(text)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.855756Z","iopub.execute_input":"2026-10-31T07:11:41.856742Z","iopub.status.idle":"2026-10-31T07:11:41.871174Z","shell.execute_reply.started":"2026-10-31T07:11:41.856708Z","shell.execute_reply":"2026-10-31T07:11:41.869885Z"},"jupyter":{"outputs_hidden":false}}
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode((text)))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.872515Z","iopub.execute_input":"2026-10-31T07:11:41.872913Z","iopub.status.idle":"2026-10-31T07:11:41.898082Z","shell.execute_reply.started":"2026-10-31T07:11:41.872875Z","shell.execute_reply":"2026-10-31T07:11:41.897094Z"},"jupyter":{"outputs_hidden":false}}
print(tokenizer.decode(tokenizer.encode(text)))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.899631Z","iopub.execute_input":"2026-10-31T07:11:41.900051Z","iopub.status.idle":"2026-10-31T07:11:41.921826Z","shell.execute_reply.started":"2026-10-31T07:11:41.900012Z","shell.execute_reply":"2026-10-31T07:11:41.920366Z"},"jupyter":{"outputs_hidden":false}}
# !pip uninstall tiktoken -y

# !pip install tiktoken==0.8.0

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.923828Z","iopub.execute_input":"2026-10-31T07:11:41.924345Z","iopub.status.idle":"2026-10-31T07:11:41.957680Z","shell.execute_reply.started":"2026-10-31T07:11:41.924301Z","shell.execute_reply":"2026-10-31T07:11:41.956557Z"},"jupyter":{"outputs_hidden":false}}
from importlib.metadata import version
import tiktoken
print("tiktoken version: ", version("tiktoken"))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.958827Z","iopub.execute_input":"2026-10-31T07:11:41.959345Z","iopub.status.idle":"2026-10-31T07:11:41.973377Z","shell.execute_reply.started":"2026-10-31T07:11:41.959315Z","shell.execute_reply":"2026-10-31T07:11:41.972058Z"},"jupyter":{"outputs_hidden":false}}
tokenizer = tiktoken.get_encoding("gpt2")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:41.975252Z","iopub.execute_input":"2026-10-31T07:11:41.975872Z","iopub.status.idle":"2026-10-31T07:11:42.003411Z","shell.execute_reply.started":"2026-10-31T07:11:41.975829Z","shell.execute_reply":"2026-10-31T07:11:42.001909Z"},"jupyter":{"outputs_hidden":false}}
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terrraces"
    "of someunknowPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.005180Z","iopub.execute_input":"2026-10-31T07:11:42.005624Z","iopub.status.idle":"2026-10-31T07:11:42.030400Z","shell.execute_reply.started":"2026-10-31T07:11:42.005582Z","shell.execute_reply":"2026-10-31T07:11:42.028906Z"},"jupyter":{"outputs_hidden":false}}
strings = tokenizer.decode(integers)
print(strings)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.032062Z","iopub.execute_input":"2026-10-31T07:11:42.034261Z","iopub.status.idle":"2026-10-31T07:11:42.072016Z","shell.execute_reply.started":"2026-10-31T07:11:42.034180Z","shell.execute_reply":"2026-10-31T07:11:42.071097Z"},"jupyter":{"outputs_hidden":false}}
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.076709Z","iopub.execute_input":"2026-10-31T07:11:42.077080Z","iopub.status.idle":"2026-10-31T07:11:42.085550Z","shell.execute_reply.started":"2026-10-31T07:11:42.077053Z","shell.execute_reply":"2026-10-31T07:11:42.084405Z"},"jupyter":{"outputs_hidden":false}}
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.087091Z","iopub.execute_input":"2026-10-31T07:11:42.087533Z","iopub.status.idle":"2026-10-31T07:11:42.115574Z","shell.execute_reply.started":"2026-10-31T07:11:42.087502Z","shell.execute_reply":"2026-10-31T07:11:42.114092Z"},"jupyter":{"outputs_hidden":false}}
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "--->", desired)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.117187Z","iopub.execute_input":"2026-10-31T07:11:42.117937Z","iopub.status.idle":"2026-10-31T07:11:42.144144Z","shell.execute_reply.started":"2026-10-31T07:11:42.117896Z","shell.execute_reply":"2026-10-31T07:11:42.142952Z"},"jupyter":{"outputs_hidden":false}}
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.145593Z","iopub.execute_input":"2026-10-31T07:11:42.145945Z","iopub.status.idle":"2026-10-31T07:11:42.193747Z","shell.execute_reply.started":"2026-10-31T07:11:42.145916Z","shell.execute_reply":"2026-10-31T07:11:42.192341Z"},"jupyter":{"outputs_hidden":false}}
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.196457Z","iopub.execute_input":"2026-10-31T07:11:42.196891Z","iopub.status.idle":"2026-10-31T07:11:42.213384Z","shell.execute_reply.started":"2026-10-31T07:11:42.196853Z","shell.execute_reply":"2026-10-31T07:11:42.212244Z"},"jupyter":{"outputs_hidden":false}}
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.216023Z","iopub.execute_input":"2026-10-31T07:11:42.216685Z","iopub.status.idle":"2026-10-31T07:11:42.343280Z","shell.execute_reply.started":"2026-10-31T07:11:42.216644Z","shell.execute_reply":"2026-10-31T07:11:42.341879Z"},"jupyter":{"outputs_hidden":false}}
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text= f.read()
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=8, stride=2, shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.344675Z","iopub.execute_input":"2026-10-31T07:11:42.345089Z","iopub.status.idle":"2026-10-31T07:11:42.353244Z","shell.execute_reply.started":"2026-10-31T07:11:42.345053Z","shell.execute_reply":"2026-10-31T07:11:42.352223Z"},"jupyter":{"outputs_hidden":false}}
second_batch = next(data_iter)
print(second_batch)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.354750Z","iopub.execute_input":"2026-10-31T07:11:42.355139Z","iopub.status.idle":"2026-10-31T07:11:42.407875Z","shell.execute_reply.started":"2026-10-31T07:11:42.355102Z","shell.execute_reply":"2026-10-31T07:11:42.406679Z"},"jupyter":{"outputs_hidden":false}}
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs: \n", inputs)
print("Targets: \n", targets)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.409242Z","iopub.execute_input":"2026-10-31T07:11:42.409673Z","iopub.status.idle":"2026-10-31T07:11:42.432899Z","shell.execute_reply.started":"2026-10-31T07:11:42.409633Z","shell.execute_reply":"2026-10-31T07:11:42.431635Z"},"jupyter":{"outputs_hidden":false}}
input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.436580Z","iopub.execute_input":"2026-10-31T07:11:42.437018Z","iopub.status.idle":"2026-10-31T07:11:42.446431Z","shell.execute_reply.started":"2026-10-31T07:11:42.436978Z","shell.execute_reply":"2026-10-31T07:11:42.445287Z"},"jupyter":{"outputs_hidden":false}}
print(embedding_layer(torch.tensor([2])))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:42.627034Z","iopub.execute_input":"2026-10-31T07:11:42.627941Z","iopub.status.idle":"2026-10-31T07:11:42.766883Z","shell.execute_reply.started":"2026-10-31T07:11:42.627890Z","shell.execute_reply":"2026-10-31T07:11:42.765807Z"},"jupyter":{"outputs_hidden":false}}
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.170677Z","iopub.execute_input":"2026-10-31T07:11:43.171313Z","iopub.status.idle":"2026-10-31T07:11:43.207665Z","shell.execute_reply.started":"2026-10-31T07:11:43.171263Z","shell.execute_reply":"2026-10-31T07:11:43.206491Z"},"jupyter":{"outputs_hidden":false}}
max_length =4
dataloader = create_dataloader_v1(
    raw_text, batch_size =8, max_length = max_length,
    stride=max_length, shuffle=False
)
data_iter= iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs: \n", inputs)
print("\nInputs shape:\n", inputs.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.444269Z","iopub.execute_input":"2026-10-31T07:11:43.444665Z","iopub.status.idle":"2026-10-31T07:11:43.451492Z","shell.execute_reply.started":"2026-10-31T07:11:43.444637Z","shell.execute_reply":"2026-10-31T07:11:43.450472Z"},"jupyter":{"outputs_hidden":false}}
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.465493Z","iopub.execute_input":"2026-10-31T07:11:43.465940Z","iopub.status.idle":"2026-10-31T07:11:43.474640Z","shell.execute_reply.started":"2026-10-31T07:11:43.465905Z","shell.execute_reply":"2026-10-31T07:11:43.473465Z"},"jupyter":{"outputs_hidden":false}}
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.497744Z","iopub.execute_input":"2026-10-31T07:11:43.498133Z","iopub.status.idle":"2026-10-31T07:11:43.510794Z","shell.execute_reply.started":"2026-10-31T07:11:43.498102Z","shell.execute_reply":"2026-10-31T07:11:43.509833Z"},"jupyter":{"outputs_hidden":false}}
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.514555Z","iopub.execute_input":"2026-10-31T07:11:43.514901Z","iopub.status.idle":"2026-10-31T07:11:43.533811Z","shell.execute_reply.started":"2026-10-31T07:11:43.514872Z","shell.execute_reply":"2026-10-31T07:11:43.532695Z"},"jupyter":{"outputs_hidden":false}}
import torch 
inputs = torch.tensor( 
  [[0.43, 0.15, 0.89], # Your     (x^1) 
   [0.55, 0.87, 0.66], # journey  (x^2) 
   [0.57, 0.85, 0.64], # starts   (x^3) 
   [0.22, 0.58, 0.33], # with     (x^4) 
   [0.77, 0.25, 0.10], # one      (x^5) 
   [0.05, 0.80, 0.55]] # step     (x^6) 
)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.554181Z","iopub.execute_input":"2026-10-31T07:11:43.554599Z","iopub.status.idle":"2026-10-31T07:11:43.564338Z","shell.execute_reply.started":"2026-10-31T07:11:43.554567Z","shell.execute_reply":"2026-10-31T07:11:43.563315Z"},"jupyter":{"outputs_hidden":false}}
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.580552Z","iopub.execute_input":"2026-10-31T07:11:43.580938Z","iopub.status.idle":"2026-10-31T07:11:43.590898Z","shell.execute_reply.started":"2026-10-31T07:11:43.580910Z","shell.execute_reply":"2026-10-31T07:11:43.589583Z"},"jupyter":{"outputs_hidden":false}}
res = 0
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0], query))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.622137Z","iopub.execute_input":"2026-10-31T07:11:43.622514Z","iopub.status.idle":"2026-10-31T07:11:43.631775Z","shell.execute_reply.started":"2026-10-31T07:11:43.622488Z","shell.execute_reply":"2026-10-31T07:11:43.630643Z"},"jupyter":{"outputs_hidden":false}}
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.655593Z","iopub.execute_input":"2026-10-31T07:11:43.655983Z","iopub.status.idle":"2026-10-31T07:11:43.665628Z","shell.execute_reply.started":"2026-10-31T07:11:43.655952Z","shell.execute_reply":"2026-10-31T07:11:43.664608Z"},"jupyter":{"outputs_hidden":false}}
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum: ", attn_weights_2_naive.sum())

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.688128Z","iopub.execute_input":"2026-10-31T07:11:43.688546Z","iopub.status.idle":"2026-10-31T07:11:43.697823Z","shell.execute_reply.started":"2026-10-31T07:11:43.688515Z","shell.execute_reply":"2026-10-31T07:11:43.696690Z"},"jupyter":{"outputs_hidden":false}}
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention wights: ", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.726015Z","iopub.execute_input":"2026-10-31T07:11:43.726461Z","iopub.status.idle":"2026-10-31T07:11:43.735571Z","shell.execute_reply.started":"2026-10-31T07:11:43.726429Z","shell.execute_reply":"2026-10-31T07:11:43.734444Z"},"jupyter":{"outputs_hidden":false}}
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.761177Z","iopub.execute_input":"2026-10-31T07:11:43.761590Z","iopub.status.idle":"2026-10-31T07:11:43.774024Z","shell.execute_reply.started":"2026-10-31T07:11:43.761558Z","shell.execute_reply":"2026-10-31T07:11:43.772629Z"},"jupyter":{"outputs_hidden":false}}
import time
start_time = time.time()  # 记录开始时的系统时间（时间戳）
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
end_time = time.time()    # 记录结束时的系统时间
elapsed_time = end_time - start_time
print(f"这段代码运行了 {elapsed_time:.4f} 秒")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.787300Z","iopub.execute_input":"2026-10-31T07:11:43.788433Z","iopub.status.idle":"2026-10-31T07:11:43.797891Z","shell.execute_reply.started":"2026-10-31T07:11:43.788389Z","shell.execute_reply":"2026-10-31T07:11:43.796696Z"},"jupyter":{"outputs_hidden":false}}
start_time = time.time()  # 记录开始时的系统时间（时间戳）
attn_scores = inputs @ inputs.T
print(attn_scores)
end_time = time.time()    # 记录结束时的系统时间
elapsed_time = end_time - start_time
print(f"这段代码运行了 {elapsed_time:.4f} 秒")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.821875Z","iopub.execute_input":"2026-10-31T07:11:43.822306Z","iopub.status.idle":"2026-10-31T07:11:43.827881Z","shell.execute_reply.started":"2026-10-31T07:11:43.822271Z","shell.execute_reply":"2026-10-31T07:11:43.826750Z"},"jupyter":{"outputs_hidden":false}}
# import torch
# import time

# # 造一个 1000 个词，每个词 768 维的矩阵（接近真实规模）
# big_inputs = torch.randn(1000, 768)
# query = big_inputs[1]

# # 1. 测试 for 循环
# start = time.time()
# for _ in range(1000): # 跑 10 次
#     scores = torch.empty(big_inputs.shape[0])
#     for i, x_i in enumerate(big_inputs):
#         scores[i] = torch.dot(x_i, query)
# end = time.time()
# print(f"for 循环耗时: {end - start:.4f} 秒")

# # 2. 测试矩阵乘法
# start = time.time()
# for _ in range(1000):
#     scores_matmul = big_inputs @ query
# end = time.time()
# print(f"矩阵乘法耗时: {end - start:.4f} 秒")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.863847Z","iopub.execute_input":"2026-10-31T07:11:43.865070Z","iopub.status.idle":"2026-10-31T07:11:43.873047Z","shell.execute_reply.started":"2026-10-31T07:11:43.865023Z","shell.execute_reply":"2026-10-31T07:11:43.871896Z"},"jupyter":{"outputs_hidden":false}}
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.884560Z","iopub.execute_input":"2026-10-31T07:11:43.884985Z","iopub.status.idle":"2026-10-31T07:11:43.901541Z","shell.execute_reply.started":"2026-10-31T07:11:43.884952Z","shell.execute_reply":"2026-10-31T07:11:43.900409Z"},"jupyter":{"outputs_hidden":false}}
row_2_sum = sum([0.1385,0.2379,0.2333,0.1240,0.1082,0.1581])
row_3_sum = sum([0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452])
row_4_sum = sum([0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565])

print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
print("Row 3 sum:", row_3_sum)
print("Row 4 sum:", row_4_sum)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.922443Z","iopub.execute_input":"2026-10-31T07:11:43.922839Z","iopub.status.idle":"2026-10-31T07:11:43.936441Z","shell.execute_reply.started":"2026-10-31T07:11:43.922809Z","shell.execute_reply":"2026-10-31T07:11:43.935172Z"},"jupyter":{"outputs_hidden":false}}
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.956539Z","iopub.execute_input":"2026-10-31T07:11:43.957319Z","iopub.status.idle":"2026-10-31T07:11:43.967919Z","shell.execute_reply.started":"2026-10-31T07:11:43.957272Z","shell.execute_reply":"2026-10-31T07:11:43.966636Z"},"jupyter":{"outputs_hidden":false}}
print("Previous 2nd context vector:", context_vec_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:43.969777Z","iopub.execute_input":"2026-10-31T07:11:43.970280Z","iopub.status.idle":"2026-10-31T07:11:43.994434Z","shell.execute_reply.started":"2026-10-31T07:11:43.970131Z","shell.execute_reply":"2026-10-31T07:11:43.993306Z"},"jupyter":{"outputs_hidden":false}}
x_2 = inputs[1]
d_in = inputs.shape[1] #实际使用：d_in = inputs.shape[-1]   # 取最后一个维度，不管是 2D 还是 3D，永远是特征维度 768

d_out = 2

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.006690Z","iopub.execute_input":"2026-10-31T07:11:44.007107Z","iopub.status.idle":"2026-10-31T07:11:44.020727Z","shell.execute_reply.started":"2026-10-31T07:11:44.007074Z","shell.execute_reply":"2026-10-31T07:11:44.019111Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_key = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))
W_value  = torch.nn.Parameter(torch.rand(d_in, d_out, requires_grad=False))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.022533Z","iopub.execute_input":"2026-10-31T07:11:44.022938Z","iopub.status.idle":"2026-10-31T07:11:44.045934Z","shell.execute_reply.started":"2026-10-31T07:11:44.022897Z","shell.execute_reply":"2026-10-31T07:11:44.044733Z"},"jupyter":{"outputs_hidden":false}}
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.048248Z","iopub.execute_input":"2026-10-31T07:11:44.048654Z","iopub.status.idle":"2026-10-31T07:11:44.076101Z","shell.execute_reply.started":"2026-10-31T07:11:44.048623Z","shell.execute_reply":"2026-10-31T07:11:44.074772Z"},"jupyter":{"outputs_hidden":false}}
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.085408Z","iopub.execute_input":"2026-10-31T07:11:44.085859Z","iopub.status.idle":"2026-10-31T07:11:44.104143Z","shell.execute_reply.started":"2026-10-31T07:11:44.085829Z","shell.execute_reply":"2026-10-31T07:11:44.103071Z"},"jupyter":{"outputs_hidden":false}}
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.114849Z","iopub.execute_input":"2026-10-31T07:11:44.115308Z","iopub.status.idle":"2026-10-31T07:11:44.132093Z","shell.execute_reply.started":"2026-10-31T07:11:44.115270Z","shell.execute_reply":"2026-10-31T07:11:44.130826Z"},"jupyter":{"outputs_hidden":false}}
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.153673Z","iopub.execute_input":"2026-10-31T07:11:44.154086Z","iopub.status.idle":"2026-10-31T07:11:44.162846Z","shell.execute_reply.started":"2026-10-31T07:11:44.154053Z","shell.execute_reply":"2026-10-31T07:11:44.161583Z"},"jupyter":{"outputs_hidden":false}}
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 /d_k**0.5, dim=-1)
print(attn_weights_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.189722Z","iopub.execute_input":"2026-10-31T07:11:44.190245Z","iopub.status.idle":"2026-10-31T07:11:44.198333Z","shell.execute_reply.started":"2026-10-31T07:11:44.190210Z","shell.execute_reply":"2026-10-31T07:11:44.197121Z"},"jupyter":{"outputs_hidden":false}}
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.217321Z","iopub.execute_input":"2026-10-31T07:11:44.217696Z","iopub.status.idle":"2026-10-31T07:11:44.229685Z","shell.execute_reply.started":"2026-10-31T07:11:44.217668Z","shell.execute_reply":"2026-10-31T07:11:44.228475Z"},"jupyter":{"outputs_hidden":false}}
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        # 用随机数初始化，实际应用中，避免随机数导致梯度暴，用Linear，它内部自带 Kaiming 均匀分布初始化：self.W_query = nn.Linear(d_in, d_out, bias=False)

    def forward(self,x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega 注意力分数它里面装的是点积（内积）。点积在数学上衡量的是两个向量的相似度或相关性。
#如果分数很大：说明第 i 个词在找的信息，刚好第 j 个词身上有，它们关系很紧密。（例如：“苹果”的 Query 遇到了“红色”的 Key）。
#如果分数很小（甚至为负数）：说明这俩词八竿子打不着，方向不一致。
        attn_weights = torch.softmax(
            attn_scores /keys.shape[-1]**0.5, dim=-1
        ) #注意力权重
        context_vec = attn_weights @ values
        return context_vec

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.243230Z","iopub.execute_input":"2026-10-31T07:11:44.244504Z","iopub.status.idle":"2026-10-31T07:11:44.265671Z","shell.execute_reply.started":"2026-10-31T07:11:44.244456Z","shell.execute_reply":"2026-10-31T07:11:44.264521Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.271335Z","iopub.execute_input":"2026-10-31T07:11:44.271842Z","iopub.status.idle":"2026-10-31T07:11:44.286334Z","shell.execute_reply.started":"2026-10-31T07:11:44.271795Z","shell.execute_reply":"2026-10-31T07:11:44.285218Z"},"jupyter":{"outputs_hidden":false}}
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value=nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        queries = x @ W_query
        keys = x @ W_key
        values = x @ W_value
        attn_socres = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.316096Z","iopub.execute_input":"2026-10-31T07:11:44.317661Z","iopub.status.idle":"2026-10-31T07:11:44.330810Z","shell.execute_reply.started":"2026-10-31T07:11:44.317606Z","shell.execute_reply":"2026-10-31T07:11:44.329514Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.333184Z","iopub.execute_input":"2026-10-31T07:11:44.333764Z","iopub.status.idle":"2026-10-31T07:11:44.358639Z","shell.execute_reply.started":"2026-10-31T07:11:44.333730Z","shell.execute_reply":"2026-10-31T07:11:44.357369Z"},"jupyter":{"outputs_hidden":false}}
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5, dim=-1
)
print(attn_weights)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.377566Z","iopub.execute_input":"2026-10-31T07:11:44.377957Z","iopub.status.idle":"2026-10-31T07:11:44.386849Z","shell.execute_reply.started":"2026-10-31T07:11:44.377927Z","shell.execute_reply":"2026-10-31T07:11:44.385668Z"},"jupyter":{"outputs_hidden":false}}
context_length = attn_scores.shape[0]
masked_simple = torch.tril(torch.ones(context_length, context_length))
print(masked_simple)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.391691Z","iopub.execute_input":"2026-10-31T07:11:44.392671Z","iopub.status.idle":"2026-10-31T07:11:44.413815Z","shell.execute_reply.started":"2026-10-31T07:11:44.392624Z","shell.execute_reply":"2026-10-31T07:11:44.412525Z"},"jupyter":{"outputs_hidden":false}}
maksed_simple = attn_weights * masked_simple
print(masked_simple)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.420253Z","iopub.execute_input":"2026-10-31T07:11:44.420814Z","iopub.status.idle":"2026-10-31T07:11:44.439317Z","shell.execute_reply.started":"2026-10-31T07:11:44.420780Z","shell.execute_reply":"2026-10-31T07:11:44.437976Z"},"jupyter":{"outputs_hidden":false}}
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.458154Z","iopub.execute_input":"2026-10-31T07:11:44.458608Z","iopub.status.idle":"2026-10-31T07:11:44.469263Z","shell.execute_reply.started":"2026-10-31T07:11:44.458571Z","shell.execute_reply":"2026-10-31T07:11:44.468128Z"},"jupyter":{"outputs_hidden":false}}
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.485007Z","iopub.execute_input":"2026-10-31T07:11:44.485419Z","iopub.status.idle":"2026-10-31T07:11:44.494374Z","shell.execute_reply.started":"2026-10-31T07:11:44.485388Z","shell.execute_reply":"2026-10-31T07:11:44.493033Z"},"jupyter":{"outputs_hidden":false}}
attn_weights = torch.softmax(masked /keys.shape[-1],dim=-1)
print(attn_weights)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.513992Z","iopub.execute_input":"2026-10-31T07:11:44.514742Z","iopub.status.idle":"2026-10-31T07:11:44.528262Z","shell.execute_reply.started":"2026-10-31T07:11:44.514707Z","shell.execute_reply":"2026-10-31T07:11:44.526860Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.550684Z","iopub.execute_input":"2026-10-31T07:11:44.551850Z","iopub.status.idle":"2026-10-31T07:11:44.562613Z","shell.execute_reply.started":"2026-10-31T07:11:44.551810Z","shell.execute_reply":"2026-10-31T07:11:44.561431Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
print(dropout(attn_weights))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.576945Z","iopub.execute_input":"2026-10-31T07:11:44.577457Z","iopub.status.idle":"2026-10-31T07:11:44.585038Z","shell.execute_reply.started":"2026-10-31T07:11:44.577422Z","shell.execute_reply":"2026-10-31T07:11:44.583616Z"},"jupyter":{"outputs_hidden":false}}
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.587570Z","iopub.execute_input":"2026-10-31T07:11:44.588058Z","iopub.status.idle":"2026-10-31T07:11:44.613348Z","shell.execute_reply.started":"2026-10-31T07:11:44.588028Z","shell.execute_reply":"2026-10-31T07:11:44.611921Z"},"jupyter":{"outputs_hidden":false}}
class CausalAttention(nn.Module): #因果注意力，在预测当前词时，模型只能看到它和它之前的词（过去），绝对不能看到它之后的词（未来）
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) #model.eval() 会让 Dropout 失效
        self.register_buffer( #自动跟随设备，自动保存，不计算梯度
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_( #用了原地操作，会直接修改 attn_scores 的值以节省显存。
            self.mask.bool() [:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
        #BERT，它就不是因果注意力。因为 BERT 的任务是“完形填空”（比如遮住中间的词，根据前后文猜）

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.617173Z","iopub.execute_input":"2026-10-31T07:11:44.617601Z","iopub.status.idle":"2026-10-31T07:11:44.648333Z","shell.execute_reply.started":"2026-10-31T07:11:44.617572Z","shell.execute_reply":"2026-10-31T07:11:44.647265Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = ca(batch)
print("context_vec.shape: ", context_vec.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.650183Z","iopub.execute_input":"2026-10-31T07:11:44.651544Z","iopub.status.idle":"2026-10-31T07:11:44.671076Z","shell.execute_reply.started":"2026-10-31T07:11:44.651503Z","shell.execute_reply":"2026-10-31T07:11:44.669972Z"},"jupyter":{"outputs_hidden":false}}
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        #后面有 LayerNorm，Bias 是多余的,节约参数和显存,去掉 bias 可以稍微减轻过拟合的风险
        #在 GPT 的前馈网络（FFN）层里，Linear 通常还是保留 bias 的。
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.672839Z","iopub.execute_input":"2026-10-31T07:11:44.673273Z","iopub.status.idle":"2026-10-31T07:11:44.714320Z","shell.execute_reply.started":"2026-10-31T07:11:44.673238Z","shell.execute_reply":"2026-10-31T07:11:44.713084Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads = 2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape: ", context_vecs.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.717322Z","iopub.execute_input":"2026-10-31T07:11:44.717699Z","iopub.status.idle":"2026-10-31T07:11:44.733837Z","shell.execute_reply.started":"2026-10-31T07:11:44.717669Z","shell.execute_reply":"2026-10-31T07:11:44.732792Z"},"jupyter":{"outputs_hidden":false}}
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0 ), \
            "d_out must be divisilble by num_heads"
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
            torch.triu(torch.ones(context_length, context_length),diagonal=1)
        ) #对角线上移取上三角
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        #keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim) better
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)

        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #生成一个布尔掩码矩阵，用来防止模型在预测时“偷看”未来的词
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1,2)#-2 和 -1 指的是倒数第二维和倒数第一维（也就是最后两维）。
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        #out_proj 提供了一个线性混合的机会。它的权重矩阵 
        #会同时看所有头的输出，把它们按比例融合。在训练中，模型会自己学会：
        #“当头 1（语法）发现这是个问句，且头 3（情感）发现语气强烈时，
        #我应该把这两个特征结合起来，输出一个强烈的疑问语义。”
        return context_vec

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# 你用的是 torch.triu（取上三角）。如果不加 diagonal 参数，它会把主对角线也变成 1。
# 但是，因果注意力的规则是：“每个词只能看自己和之前的词”。 也就是说，主对角线（自己看自己）是允许的，必须保留为 0（不屏蔽）；而它右上方（看未来的词）必须被屏蔽。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.735464Z","iopub.execute_input":"2026-10-31T07:11:44.736044Z","iopub.status.idle":"2026-10-31T07:11:44.771442Z","shell.execute_reply.started":"2026-10-31T07:11:44.736003Z","shell.execute_reply":"2026-10-31T07:11:44.770172Z"},"jupyter":{"outputs_hidden":false}}
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], 
                    [0.8993, 0.0390, 0.9268, 0.7388], 
                    [0.7179, 0.7058, 0.9156, 0.4340]], 
                   [[0.0772, 0.3565, 0.1479, 0.5331], 
                    [0.4066, 0.2318, 0.4545, 0.9737], 
                    [0.4606, 0.5159, 0.4220, 0.5786]]]]) 
print(a.transpose(2,3))
print(a @ a.transpose(2,3))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.774075Z","iopub.execute_input":"2026-10-31T07:11:44.774573Z","iopub.status.idle":"2026-10-31T07:11:44.793982Z","shell.execute_reply.started":"2026-10-31T07:11:44.774541Z","shell.execute_reply":"2026-10-31T07:11:44.792827Z"},"jupyter":{"outputs_hidden":false}}
first_head = a[0,0,:,:]
first_res = first_head @ first_head.T
print("First head: \n", first_res)
second_head = a[0,1,:,:]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# 向量：通常代表一个具体的事物的表示。比如一个词的特征，或者一个点。它是一维的，方向性很明确。
# 
# 张量：通常代表批量的事物或复杂的数据结构。比如一个 batch 的句子，它包含了多个向量，
# 是更高维度的容器。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.795533Z","iopub.execute_input":"2026-10-31T07:11:44.796411Z","iopub.status.idle":"2026-10-31T07:11:44.822867Z","shell.execute_reply.started":"2026-10-31T07:11:44.796375Z","shell.execute_reply":"2026-10-31T07:11:44.821674Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.825047Z","iopub.execute_input":"2026-10-31T07:11:44.825577Z","iopub.status.idle":"2026-10-31T07:11:44.847561Z","shell.execute_reply.started":"2026-10-31T07:11:44.825541Z","shell.execute_reply":"2026-10-31T07:11:44.846500Z"},"jupyter":{"outputs_hidden":false}}
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.848844Z","iopub.execute_input":"2026-10-31T07:11:44.849218Z","iopub.status.idle":"2026-10-31T07:11:44.878002Z","shell.execute_reply.started":"2026-10-31T07:11:44.849162Z","shell.execute_reply":"2026-10-31T07:11:44.876848Z"},"jupyter":{"outputs_hidden":false}}
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 词嵌入层：将输入的 token ID 映射为稠密向量，形状 (vocab_size, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入层：为序列中的每个位置学习一个向量，形状 (context_length, emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Dropout 层：在训练时随机丢弃部分神经元，防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 堆叠 n_layers 个 Transformer 块（这里用 Dummy 占位，直接透传）
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
             for _ in range(cfg["n_layers"])]
        )
        # 最终层归一化（这里用 Dummy 占位，直接透传）
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 输出投影层：将 emb_dim 维特征映射回词表大小，输出每个词的预测分数 (logits)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        # 获取输入的批次大小和序列长度
        batch_size, seq_len = in_idx.shape
        # 查表得到词向量：形状 (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        # 生成位置索引并查表得到位置向量：形状 (seq_len, emb_dim)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        # 词向量 + 位置向量（广播机制），注入位置信息,向量向量向量向量向量向量
        x = tok_embeds + pos_embeds
        # 对嵌入向量应用 Dropout
        x = self.drop_emb(x)
        # 依次通过所有 Transformer 块（此处占位，形状不变）
        x = self.trf_blocks(x)
        # 最终层归一化（此处占位，形状不变）
        x = self.final_norm(x)
        # 映射到词表空间，得到每个位置的下一个词预测分数
        logits = self.out_head(x)
        # 返回 logits，形状 (batch_size, seq_len, vocab_size)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 占位块：目前不包含任何子模块，用于搭建骨架
    def forward(self, x):
        # 直接返回输入，不改变数据，等后续替换为真正的注意力+前馈网络
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # 占位归一化层：不包含可训练参数
    def forward(self, x):
        # 直接返回输入，不改变数据，等后续替换为真正的 LayerNorm
        return x

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.880451Z","iopub.execute_input":"2026-10-31T07:11:44.881599Z","iopub.status.idle":"2026-10-31T07:11:44.908037Z","shell.execute_reply.started":"2026-10-31T07:11:44.881563Z","shell.execute_reply":"2026-10-31T07:11:44.906707Z"},"jupyter":{"outputs_hidden":false}}
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0) #stack 的作用是把多个形状相同的张量，
#按照指定的维度（dim）拼在一起，并产生一个新的维度。用 stack 把它们合成一个 (32, seq_len) 
#的大矩阵一次性喂进去，GPU 就能把所有句子同时放到几千个核心上并行计算，时间瞬间缩短。
#这就是矩阵乘法（向量化）的威力。
print(batch)
print(tokenizer)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:44.909420Z","iopub.execute_input":"2026-10-31T07:11:44.909805Z","iopub.status.idle":"2026-10-31T07:11:45.961068Z","shell.execute_reply.started":"2026-10-31T07:11:44.909775Z","shell.execute_reply":"2026-10-31T07:11:45.960063Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
#底层函数：torch.argmax() 或者 torch.topk() 配合 GPU 并行扫描，能瞬间找到最高分。
#实际应用：为了生成自然、不僵硬的文本，模型不会死板地选“唯一最高分”，而是把分数转成概率，
#在高分词的候选池（Top-K 或 Top-P）里“抽签”决定下一个词。
#这就是大模型有“创造力”和“随机性”的来源。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:45.963232Z","iopub.execute_input":"2026-10-31T07:11:45.963616Z","iopub.status.idle":"2026-10-31T07:11:45.978533Z","shell.execute_reply.started":"2026-10-31T07:11:45.963588Z","shell.execute_reply":"2026-10-31T07:11:45.977230Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
batch_example = torch.randn(2,5)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU()) #ReLU:：它只是简单地将负输
#入值设为0，从而确保层的输出值都是正值，
out = layer(batch_example)
print(out)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:45.980115Z","iopub.execute_input":"2026-10-31T07:11:45.980592Z","iopub.status.idle":"2026-10-31T07:11:45.998274Z","shell.execute_reply.started":"2026-10-31T07:11:45.980551Z","shell.execute_reply":"2026-10-31T07:11:45.997170Z"},"jupyter":{"outputs_hidden":false}}
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
#如果没有keepdim=True，那么返回
#的均值张量将是一个二维向量[0.1324, 0.2170]，而不是 2×1 维的矩阵[[0.1324], 
#[0.2170]]。 使用keepdim=True 可以确保输出张量与输入张量具有相同
#的维度，尽管这类运算是沿指定的维度dim 减少张量的。
#在数学和编程里，像 mean（求均值）、sum（求和）、max（求最大值）、var（求方差）这类操作，统称为“归约操作”（Reduction Operations）。
#它们共同的特征就是：吃进去一堆数，吐出来一个数。 这会直接导致原来存放这堆数的那个“维度”消失，所以张量的总维数（Rank）减少了

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:45.999681Z","iopub.execute_input":"2026-10-31T07:11:46.000080Z","iopub.status.idle":"2026-10-31T07:11:46.025845Z","shell.execute_reply.started":"2026-10-31T07:11:46.000048Z","shell.execute_reply":"2026-10-31T07:11:46.024673Z"},"jupyter":{"outputs_hidden":false}}
out_norm =(out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Variance:\n", var)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.027749Z","iopub.execute_input":"2026-10-31T07:11:46.028231Z","iopub.status.idle":"2026-10-31T07:11:46.049114Z","shell.execute_reply.started":"2026-10-31T07:11:46.028163Z","shell.execute_reply":"2026-10-31T07:11:46.047928Z"},"jupyter":{"outputs_hidden":false}}
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.050741Z","iopub.execute_input":"2026-10-31T07:11:46.051378Z","iopub.status.idle":"2026-10-31T07:11:46.084063Z","shell.execute_reply.started":"2026-10-31T07:11:46.051321Z","shell.execute_reply":"2026-10-31T07:11:46.082534Z"},"jupyter":{"outputs_hidden":false}}
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 #变量eps 是一个小常数（epsilon），在归一化过程中会被加到方差上以防止除零错误。
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        #。scale 和shift是两个可训练的参数（与输入维度相同）,模型会自动进行调整。
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        #设置 unbiased=False。这意味着在方差计算中，我们会使用样本数量 n 作为方差公式的除数
        norm_x = (x - mean) /torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.089420Z","iopub.execute_input":"2026-10-31T07:11:46.090644Z","iopub.status.idle":"2026-10-31T07:11:46.118742Z","shell.execute_reply.started":"2026-10-31T07:11:46.090590Z","shell.execute_reply":"2026-10-31T07:11:46.117258Z"},"jupyter":{"outputs_hidden":false}}
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# GELU和SwiGLU是更为复杂且平滑的激活函数，分别结合了高斯分布和sigmoid门控线性
# 单元  。与较为简单的ReLU激活函数相比，它们能够提升深度学习模型的性能。 *

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.120136Z","iopub.execute_input":"2026-10-31T07:11:46.120599Z","iopub.status.idle":"2026-10-31T07:11:46.144071Z","shell.execute_reply.started":"2026-10-31T07:11:46.120565Z","shell.execute_reply":"2026-10-31T07:11:46.142603Z"},"jupyter":{"outputs_hidden":false}}
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5*x*(1+ torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.145702Z","iopub.execute_input":"2026-10-31T07:11:46.146559Z","iopub.status.idle":"2026-10-31T07:11:46.602554Z","shell.execute_reply.started":"2026-10-31T07:11:46.146508Z","shell.execute_reply":"2026-10-31T07:11:46.601045Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8,3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.603929Z","iopub.execute_input":"2026-10-31T07:11:46.604287Z","iopub.status.idle":"2026-10-31T07:11:46.611365Z","shell.execute_reply.started":"2026-10-31T07:11:46.604259Z","shell.execute_reply":"2026-10-31T07:11:46.610305Z"},"jupyter":{"outputs_hidden":false}}
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.612839Z","iopub.execute_input":"2026-10-31T07:11:46.613720Z","iopub.status.idle":"2026-10-31T07:11:46.692661Z","shell.execute_reply.started":"2026-10-31T07:11:46.613688Z","shell.execute_reply":"2026-10-31T07:11:46.691481Z"},"jupyter":{"outputs_hidden":false}}
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.694015Z","iopub.execute_input":"2026-10-31T07:11:46.694886Z","iopub.status.idle":"2026-10-31T07:11:46.703771Z","shell.execute_reply.started":"2026-10-31T07:11:46.694839Z","shell.execute_reply":"2026-10-31T07:11:46.702559Z"},"jupyter":{"outputs_hidden":false}}
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1], GELU())),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2], GELU())),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3], GELU())),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4], GELU())),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5], GELU())),
        ])
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.705188Z","iopub.execute_input":"2026-10-31T07:11:46.705539Z","iopub.status.idle":"2026-10-31T07:11:46.733719Z","shell.execute_reply.started":"2026-10-31T07:11:46.705508Z","shell.execute_reply":"2026-10-31T07:11:46.732463Z"},"jupyter":{"outputs_hidden":false}}
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.734993Z","iopub.execute_input":"2026-10-31T07:11:46.736280Z","iopub.status.idle":"2026-10-31T07:11:46.742411Z","shell.execute_reply.started":"2026-10-31T07:11:46.736241Z","shell.execute_reply":"2026-10-31T07:11:46.741317Z"},"jupyter":{"outputs_hidden":false}}
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.744008Z","iopub.execute_input":"2026-10-31T07:11:46.745375Z","iopub.status.idle":"2026-10-31T07:11:46.769554Z","shell.execute_reply.started":"2026-10-31T07:11:46.745328Z","shell.execute_reply":"2026-10-31T07:11:46.768420Z"},"jupyter":{"outputs_hidden":false}}
print_gradients(model_without_shortcut, sample_input)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.771095Z","iopub.execute_input":"2026-10-31T07:11:46.771532Z","iopub.status.idle":"2026-10-31T07:11:46.789803Z","shell.execute_reply.started":"2026-10-31T07:11:46.771493Z","shell.execute_reply":"2026-10-31T07:11:46.788733Z"},"jupyter":{"outputs_hidden":false}}
#from chapter03 import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"]) #目的 1：缓解“深层网络”的过拟合
        #目的 2：减少深层网络的“冗余”,目的 3：变相实现模型集成（Ensemble）
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.791314Z","iopub.execute_input":"2026-10-31T07:11:46.791692Z","iopub.status.idle":"2026-10-31T07:11:46.928854Z","shell.execute_reply.started":"2026-10-31T07:11:46.791653Z","shell.execute_reply":"2026-10-31T07:11:46.927687Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.930625Z","iopub.execute_input":"2026-10-31T07:11:46.931120Z","iopub.status.idle":"2026-10-31T07:11:46.941670Z","shell.execute_reply.started":"2026-10-31T07:11:46.931072Z","shell.execute_reply":"2026-10-31T07:11:46.940471Z"},"jupyter":{"outputs_hidden":false}}
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len,  device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:46.943084Z","iopub.execute_input":"2026-10-31T07:11:46.943915Z","iopub.status.idle":"2026-10-31T07:11:49.547622Z","shell.execute_reply.started":"2026-10-31T07:11:46.943873Z","shell.execute_reply":"2026-10-31T07:11:49.546393Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:49.548850Z","iopub.execute_input":"2026-10-31T07:11:49.549720Z","iopub.status.idle":"2026-10-31T07:11:49.556591Z","shell.execute_reply.started":"2026-10-31T07:11:49.549688Z","shell.execute_reply":"2026-10-31T07:11:49.555290Z"},"jupyter":{"outputs_hidden":false}}
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:49.557836Z","iopub.execute_input":"2026-10-31T07:11:49.558220Z","iopub.status.idle":"2026-10-31T07:11:49.581045Z","shell.execute_reply.started":"2026-10-31T07:11:49.558161Z","shell.execute_reply":"2026-10-31T07:11:49.580012Z"},"jupyter":{"outputs_hidden":false}}
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:49.587614Z","iopub.execute_input":"2026-10-31T07:11:49.588556Z","iopub.status.idle":"2026-10-31T07:11:49.605138Z","shell.execute_reply.started":"2026-10-31T07:11:49.588517Z","shell.execute_reply":"2026-10-31T07:11:49.604032Z"},"jupyter":{"outputs_hidden":false}}
total_params_gpt2=(
    total_params - sum(p.numel()
                       for p in model.out_head.parameters())
)
print(f"Number of trainable parameters"
     f"considering weight tying: {total_params_gpt2:,}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:49.606456Z","iopub.execute_input":"2026-10-31T07:11:49.607557Z","iopub.status.idle":"2026-10-31T07:11:49.628951Z","shell.execute_reply.started":"2026-10-31T07:11:49.607507Z","shell.execute_reply":"2026-10-31T07:11:49.627825Z"},"jupyter":{"outputs_hidden":false}}
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:49.630462Z","iopub.execute_input":"2026-10-31T07:11:49.630948Z","iopub.status.idle":"2026-10-31T07:11:49.656506Z","shell.execute_reply.started":"2026-10-31T07:11:49.630905Z","shell.execute_reply":"2026-10-31T07:11:49.655045Z"},"jupyter":{"outputs_hidden":false}}
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            #转换代码来展示将logits转换为概率的完整过程，这有助于
            #更好地理解模型如何生成最有可能的下一个词元，这一过程被称为贪心解码。
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:49.657905Z","iopub.execute_input":"2026-10-31T07:11:49.658274Z","iopub.status.idle":"2026-10-31T07:11:49.693350Z","shell.execute_reply.started":"2026-10-31T07:11:49.658232Z","shell.execute_reply":"2026-10-31T07:11:49.692248Z"},"jupyter":{"outputs_hidden":false}}
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:49.694866Z","iopub.execute_input":"2026-10-31T07:11:49.695387Z","iopub.status.idle":"2026-10-31T07:11:50.200813Z","shell.execute_reply.started":"2026-10-31T07:11:49.695348Z","shell.execute_reply":"2026-10-31T07:11:50.199920Z"},"jupyter":{"outputs_hidden":false}}

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output: ", out)
print("Output length: ", len(out[0]))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:50.202064Z","iopub.execute_input":"2026-10-31T07:11:50.202898Z","iopub.status.idle":"2026-10-31T07:11:50.209006Z","shell.execute_reply.started":"2026-10-31T07:11:50.202851Z","shell.execute_reply":"2026-10-31T07:11:50.208054Z"},"jupyter":{"outputs_hidden":false}}
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:50.210391Z","iopub.execute_input":"2026-10-31T07:11:50.210814Z","iopub.status.idle":"2026-10-31T07:11:52.637542Z","shell.execute_reply.started":"2026-10-31T07:11:50.210773Z","shell.execute_reply":"2026-10-31T07:11:52.636269Z"},"jupyter":{"outputs_hidden":false}}
import torch
GPT_CONFIG_124M={
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:52.639130Z","iopub.execute_input":"2026-10-31T07:11:52.639533Z","iopub.status.idle":"2026-10-31T07:11:53.452668Z","shell.execute_reply.started":"2026-10-31T07:11:52.639488Z","shell.execute_reply":"2026-10-31T07:11:53.451609Z"},"jupyter":{"outputs_hidden":false}}
import tiktoken

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
start_context = "Every effor moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"]
)
print("Output text: \n", token_ids_to_text(token_ids, tokenizer))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.454104Z","iopub.execute_input":"2026-10-31T07:11:53.454589Z","iopub.status.idle":"2026-10-31T07:11:53.460919Z","shell.execute_reply.started":"2026-10-31T07:11:53.454546Z","shell.execute_reply":"2026-10-31T07:11:53.459580Z"},"jupyter":{"outputs_hidden":false}}
inputs = torch.tensor([[16833, 3626, 6100],
                      [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345 ],
                      [1107, 588, 11311]])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.462467Z","iopub.execute_input":"2026-10-31T07:11:53.462782Z","iopub.status.idle":"2026-10-31T07:11:53.574984Z","shell.execute_reply.started":"2026-10-31T07:11:53.462748Z","shell.execute_reply":"2026-10-31T07:11:53.573976Z"},"jupyter":{"outputs_hidden":false}}
with torch.no_grad(): #关闭自动求导（Autograd）机制,因为这里只是查看模型输出（推理/评估），不需要更新参数，所以关闭计算图以节省显存和加快速度
    logits = model(inputs) #把输入数据 inputs 喂给模型，执行完整的前向传播
    print("logits: \n",logits)
probas = torch.softmax(logits, dim=-1) #对 logits 的最后一个维度（即 vocab_size 词表维度）应用 Softmax 函数。
#数学意义：把原始分数变成概率分布，确保每个位置上的 5 万多个词的概率加起来等于 1，且每个数都在 0 到 1 之间。
print(probas.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.576380Z","iopub.execute_input":"2026-10-31T07:11:53.576772Z","iopub.status.idle":"2026-10-31T07:11:53.586397Z","shell.execute_reply.started":"2026-10-31T07:11:53.576731Z","shell.execute_reply":"2026-10-31T07:11:53.584964Z"},"jupyter":{"outputs_hidden":false}}
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
#它把刚刚算出来的“概率分布（probas）”，瞬间变成了具体的词（Token ID）。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.588134Z","iopub.execute_input":"2026-10-31T07:11:53.588651Z","iopub.status.idle":"2026-10-31T07:11:53.610992Z","shell.execute_reply.started":"2026-10-31T07:11:53.588618Z","shell.execute_reply":"2026-10-31T07:11:53.609883Z"},"jupyter":{"outputs_hidden":false}}
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
     f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.612597Z","iopub.execute_input":"2026-10-31T07:11:53.613035Z","iopub.status.idle":"2026-10-31T07:11:53.637582Z","shell.execute_reply.started":"2026-10-31T07:11:53.612992Z","shell.execute_reply":"2026-10-31T07:11:53.636414Z"},"jupyter":{"outputs_hidden":false}}
torch.set_printoptions(sci_mode=True) #避免四舍五入精度
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1: ", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2: ", target_probas_2)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.639445Z","iopub.execute_input":"2026-10-31T07:11:53.640347Z","iopub.status.idle":"2026-10-31T07:11:53.668947Z","shell.execute_reply.started":"2026-10-31T07:11:53.640295Z","shell.execute_reply":"2026-10-31T07:11:53.667902Z"},"jupyter":{"outputs_hidden":false}}
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
#计算交叉熵损失（Cross-Entropy Loss）的前置步骤
#它做了两件事：把之前抽出来的“目标词概率”拼接起来，然后取对数（log）。
#在大模型中，我们通常使用负对数似然（Negative Log-Likelihood, NLL）来作为损失函数。因为：
#防止数值下溢：概率值极小（比如 1e-45），在计算机里相乘很容易直接变成 0。变成对数后，乘法变成了加法（
#log(a⋅b)=loga+logb），数值更稳定。
#优化目标：模型的目标是最大化正确词的概率，等价于最小化它的负对数（即让负数越负越好，或者取相反数后越小越好）。
#梯度特性：当概率接近 1 时，log(1)=0，损失为 0；当概率接近 0 时，对数会变成负无穷大，模型会受到极大的惩罚（梯度爆炸），从而逼着它快速修正错误。
#一句话总结：这段代码把抽出来的目标词概率拼接，并求对数。因为概率极小，所以 log_probas 会是一堆负数。这些负数越“负”，说明模型预测得越差，后续训练的目标就是把这些负数“拉回”到 0 附近。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.670434Z","iopub.execute_input":"2026-10-31T07:11:53.670820Z","iopub.status.idle":"2026-10-31T07:11:53.690763Z","shell.execute_reply.started":"2026-10-31T07:11:53.670790Z","shell.execute_reply":"2026-10-31T07:11:53.689629Z"},"jupyter":{"outputs_hidden":false}}
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas) #-1.0794e+01 就是科学计数法，等于 -10.794。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.692162Z","iopub.execute_input":"2026-10-31T07:11:53.692598Z","iopub.status.idle":"2026-10-31T07:11:53.716683Z","shell.execute_reply.started":"2026-10-31T07:11:53.692557Z","shell.execute_reply":"2026-10-31T07:11:53.715676Z"},"jupyter":{"outputs_hidden":false}}
ng_avg_log_probas = avg_log_probas * -1
print(ng_avg_log_probas)
#在深度学习中，将-10.7940这个负值转换为 10.7940的术语称为交叉熵损失。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.718034Z","iopub.execute_input":"2026-10-31T07:11:53.718519Z","iopub.status.idle":"2026-10-31T07:11:53.740421Z","shell.execute_reply.started":"2026-10-31T07:11:53.718488Z","shell.execute_reply":"2026-10-31T07:11:53.739231Z"},"jupyter":{"outputs_hidden":false}}
print("Logits share: ", logits.shape)
print("Targets shape: ", targets.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.741711Z","iopub.execute_input":"2026-10-31T07:11:53.742059Z","iopub.status.idle":"2026-10-31T07:11:53.769566Z","shell.execute_reply.started":"2026-10-31T07:11:53.742031Z","shell.execute_reply":"2026-10-31T07:11:53.768517Z"},"jupyter":{"outputs_hidden":false}}
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits: ", logits_flat.shape)
print("Flattened targets: ", targets_flat.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.771029Z","iopub.execute_input":"2026-10-31T07:11:53.771625Z","iopub.status.idle":"2026-10-31T07:11:53.796528Z","shell.execute_reply.started":"2026-10-31T07:11:53.771580Z","shell.execute_reply":"2026-10-31T07:11:53.795290Z"},"jupyter":{"outputs_hidden":false}}
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# 困惑度通常与交叉熵损失一起用来评估模型在诸如语言建模等任务中的性能。它可以提
# 供一种更易解释的方式来理解模型在预测序列中的下一个词元时的不确定性。

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.797943Z","iopub.execute_input":"2026-10-31T07:11:53.798672Z","iopub.status.idle":"2026-10-31T07:11:53.821209Z","shell.execute_reply.started":"2026-10-31T07:11:53.798628Z","shell.execute_reply":"2026-10-31T07:11:53.820231Z"},"jupyter":{"outputs_hidden":false}}
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.822644Z","iopub.execute_input":"2026-10-31T07:11:53.823162Z","iopub.status.idle":"2026-10-31T07:11:53.841071Z","shell.execute_reply.started":"2026-10-31T07:11:53.823120Z","shell.execute_reply":"2026-10-31T07:11:53.839888Z"},"jupyter":{"outputs_hidden":false}}
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
print(split_idx)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.842554Z","iopub.execute_input":"2026-10-31T07:11:53.843551Z","iopub.status.idle":"2026-10-31T07:11:53.876747Z","shell.execute_reply.started":"2026-10-31T07:11:53.843514Z","shell.execute_reply":"2026-10-31T07:11:53.875612Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)


# def create_dataloader_v11(txt, batch_size=4, max_length=256,
#                         stride=256, shuffle=True, drop_last=True,
#                         num_workers=0):
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

train_loader = create_dataloader_v1(
    train_data,
    batch_size =2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
    
)
val_loader= create_dataloader_v1(
    val_data,
    batch_size=1,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
# 训练集 train_loader：
# drop_last=True：如果最后剩下的数据不够凑齐一个完整的 batch_size（比如最后只剩 1 条，但你要求每批 2 条），就直接丢弃。
# 原因：在训练时，保持固定的张量形状能让 GPU 计算更稳定。如果保留只有 1 条数据的批次，会导致矩阵形状突变，容易引发报错。
# shuffle=True：每个 Epoch 开始前，打乱所有数据的顺序。
# 原因：防止模型记住数据的先后顺序，强迫它真正学习语言规律，这是防过拟合的绝对铁律。
# 验证集 val_loader：
# drop_last=False：最后如果不够一个 batch，也保留。
# 原因：验证集不用反向传播，形状变化不会有影响。而且我们要的是绝对公平的评估，不能漏掉任何一条验证数据。
# shuffle=False：绝对不打乱顺序。
# 原因：为了让每次评估的结果可复现、可对比。
print(len(val_loader))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:11:53.877953Z","iopub.execute_input":"2026-10-31T07:11:53.878360Z","iopub.status.idle":"2026-10-31T07:11:53.906139Z","shell.execute_reply.started":"2026-10-31T07:11:53.878321Z","shell.execute_reply":"2026-10-31T07:11:53.904960Z"},"jupyter":{"outputs_hidden":false}}
print('Train loader:')
for x, y in train_loader:
    print(x.shape, y.shape)
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
# print(len(val_loader))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-31T07:11:53.908048Z","iopub.execute_input":"2026-10-31T07:11:53.908465Z","iopub.status.idle":"2026-10-31T07:11:53.929419Z","shell.execute_reply.started":"2026-10-31T07:11:53.908429Z","shell.execute_reply":"2026-10-31T07:11:53.928298Z"}}
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-31T07:11:53.930838Z","iopub.execute_input":"2026-10-31T07:11:53.931298Z","iopub.status.idle":"2026-10-31T07:11:53.954527Z","shell.execute_reply.started":"2026-10-31T07:11:53.931257Z","shell.execute_reply":"2026-10-31T07:11:53.953328Z"}}
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-31T07:11:53.956115Z","iopub.execute_input":"2026-10-31T07:11:53.956678Z","iopub.status.idle":"2026-10-31T07:12:09.426007Z","shell.execute_reply.started":"2026-10-31T07:11:53.956637Z","shell.execute_reply":"2026-10-31T07:12:09.424675Z"}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss: ", train_loss)
print("Validation loss: ", val_loss)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:12:09.427511Z","iopub.execute_input":"2026-10-31T07:12:09.427915Z","iopub.status.idle":"2026-10-31T07:12:09.438556Z","shell.execute_reply.started":"2026-10-31T07:12:09.427886Z","shell.execute_reply":"2026-10-31T07:12:09.437475Z"},"jupyter":{"outputs_hidden":false}}
def train_model_simple(model, train_loader, val_loader,
                      optimizer, device, num_epochs, 
                      eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                     f"Train loss {train_loss:.3f},"
                     f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:12:09.439883Z","iopub.execute_input":"2026-10-31T07:12:09.440269Z","iopub.status.idle":"2026-10-31T07:12:09.466282Z","shell.execute_reply.started":"2026-10-31T07:12:09.440230Z","shell.execute_reply":"2026-10-31T07:12:09.465152Z"},"jupyter":{"outputs_hidden":false}}
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:12:09.467625Z","iopub.execute_input":"2026-10-31T07:12:09.467986Z","iopub.status.idle":"2026-10-31T07:12:09.490530Z","shell.execute_reply.started":"2026-10-31T07:12:09.467948Z","shell.execute_reply":"2026-10-31T07:12:09.489423Z"},"jupyter":{"outputs_hidden":false}}
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:12:09.491726Z","iopub.execute_input":"2026-10-31T07:12:09.492119Z","iopub.status.idle":"2026-10-31T07:13:35.400420Z","shell.execute_reply.started":"2026-10-31T07:12:09.492079Z","shell.execute_reply":"2026-10-31T07:13:35.399044Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
num_epochs =1
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs= num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effor moves you", tokenizer=tokenizer
)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:35.402129Z","iopub.execute_input":"2026-10-31T07:13:35.402628Z","iopub.status.idle":"2026-10-31T07:13:35.618487Z","shell.execute_reply.started":"2026-10-31T07:13:35.402584Z","shell.execute_reply":"2026-10-31T07:13:35.617279Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, axl = plt.subplots(figsize=(5, 3))
    axl.plot(epochs_seen, train_losses, label="Training loss")
    axl.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
        
    )
    axl.set_xlabel("Epochs")
    axl.set_ylabel("Loss")
    axl.legend(loc="upper right")
    axl.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = axl.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Toens seen")
    fig.tight_layout()
    plt.show()
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:35.619745Z","iopub.execute_input":"2026-10-31T07:13:35.620311Z","iopub.status.idle":"2026-10-31T07:13:35.636495Z","shell.execute_reply.started":"2026-10-31T07:13:35.620280Z","shell.execute_reply":"2026-10-31T07:13:35.635298Z"},"jupyter":{"outputs_hidden":false}}
model.to("cpu")
model.eval()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:35.638392Z","iopub.execute_input":"2026-10-31T07:13:35.638784Z","iopub.status.idle":"2026-10-31T07:13:38.071795Z","shell.execute_reply.started":"2026-10-31T07:13:35.638754Z","shell.execute_reply":"2026-10-31T07:13:38.070554Z"},"jupyter":{"outputs_hidden":false}}
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.073360Z","iopub.execute_input":"2026-10-31T07:13:38.074071Z","iopub.status.idle":"2026-10-31T07:13:38.080571Z","shell.execute_reply.started":"2026-10-31T07:13:38.074037Z","shell.execute_reply":"2026-10-31T07:13:38.079363Z"},"jupyter":{"outputs_hidden":false}}
vocab = { 
    "closer": 0, 
    "every": 1, 
    "effort": 2, 
    "forward": 3, 
    "inches": 4, 
    "moves": 5, 
    "pizza": 6, 
    "toward": 7, 
    "you": 8, 
} 
inverse_vocab = {v: k for k, v in vocab.items()}

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.082350Z","iopub.execute_input":"2026-10-31T07:13:38.082898Z","iopub.status.idle":"2026-10-31T07:13:38.105511Z","shell.execute_reply.started":"2026-10-31T07:13:38.082854Z","shell.execute_reply":"2026-10-31T07:13:38.104145Z"},"jupyter":{"outputs_hidden":false}}

next_token_logits = torch.tensor( 
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79] 
)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.107140Z","iopub.execute_input":"2026-10-31T07:13:38.107646Z","iopub.status.idle":"2026-10-31T07:13:38.133469Z","shell.execute_reply.started":"2026-10-31T07:13:38.107605Z","shell.execute_reply":"2026-10-31T07:13:38.132300Z"},"jupyter":{"outputs_hidden":false}}
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.134835Z","iopub.execute_input":"2026-10-31T07:13:38.135322Z","iopub.status.idle":"2026-10-31T07:13:38.168471Z","shell.execute_reply.started":"2026-10-31T07:13:38.135279Z","shell.execute_reply":"2026-10-31T07:13:38.167067Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.169844Z","iopub.execute_input":"2026-10-31T07:13:38.170158Z","iopub.status.idle":"2026-10-31T07:13:38.221790Z","shell.execute_reply.started":"2026-10-31T07:13:38.170124Z","shell.execute_reply":"2026-10-31T07:13:38.220457Z"},"jupyter":{"outputs_hidden":false}}
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
             for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
print_sampled_tokens(probas)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-31T07:13:38.223108Z","iopub.execute_input":"2026-10-31T07:13:38.223593Z","iopub.status.idle":"2026-10-31T07:13:38.229084Z","shell.execute_reply.started":"2026-10-31T07:13:38.223563Z","shell.execute_reply":"2026-10-31T07:13:38.228170Z"}}
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.230378Z","iopub.execute_input":"2026-10-31T07:13:38.230738Z","iopub.status.idle":"2026-10-31T07:13:38.481645Z","shell.execute_reply.started":"2026-10-31T07:13:38.230709Z","shell.execute_reply":"2026-10-31T07:13:38.480383Z"},"jupyter":{"outputs_hidden":false}}
temperature = [1, 0.1,5]
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                for T in temperature]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperature):
    rects = ax.bar(x + i * bar_width, scaled_probas[i],
                  bar_width, label=f'Temperature={T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.483143Z","iopub.execute_input":"2026-10-31T07:13:38.483645Z","iopub.status.idle":"2026-10-31T07:13:38.494692Z","shell.execute_reply.started":"2026-10-31T07:13:38.483606Z","shell.execute_reply":"2026-10-31T07:13:38.493587Z"},"jupyter":{"outputs_hidden":false}}
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits: ", top_logits)
print("Top positions: ", top_pos)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.496113Z","iopub.execute_input":"2026-10-31T07:13:38.497570Z","iopub.status.idle":"2026-10-31T07:13:38.523804Z","shell.execute_reply.started":"2026-10-31T07:13:38.497528Z","shell.execute_reply":"2026-10-31T07:13:38.522775Z"},"jupyter":{"outputs_hidden":false}}
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float('-inf')),
    other=next_token_logits
)
print(new_logits)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.525087Z","iopub.execute_input":"2026-10-31T07:13:38.525584Z","iopub.status.idle":"2026-10-31T07:13:38.547614Z","shell.execute_reply.started":"2026-10-31T07:13:38.525547Z","shell.execute_reply":"2026-10-31T07:13:38.546423Z"},"jupyter":{"outputs_hidden":false}}
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.548932Z","iopub.execute_input":"2026-10-31T07:13:38.549335Z","iopub.status.idle":"2026-10-31T07:13:38.570714Z","shell.execute_reply.started":"2026-10-31T07:13:38.549296Z","shell.execute_reply":"2026-10-31T07:13:38.569543Z"},"jupyter":{"outputs_hidden":false}}
def generate(model, idx, max_new_tokens, context_size,
            temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
            
        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:38.572168Z","iopub.execute_input":"2026-10-31T07:13:38.572656Z","iopub.status.idle":"2026-10-31T07:13:39.825626Z","shell.execute_reply.started":"2026-10-31T07:13:38.572624Z","shell.execute_reply":"2026-10-31T07:13:39.824471Z"},"jupyter":{"outputs_hidden":false}}
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:39.834585Z","iopub.execute_input":"2026-10-31T07:13:39.834964Z","iopub.status.idle":"2026-10-31T07:13:40.826587Z","shell.execute_reply.started":"2026-10-31T07:13:39.834938Z","shell.execute_reply":"2026-10-31T07:13:40.825485Z"},"jupyter":{"outputs_hidden":false}}
torch.save(model.state_dict(), "model.pth")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:40.827811Z","iopub.execute_input":"2026-10-31T07:13:40.828272Z","iopub.status.idle":"2026-10-31T07:13:43.813373Z","shell.execute_reply.started":"2026-10-31T07:13:40.828231Z","shell.execute_reply":"2026-10-31T07:13:43.812131Z"},"jupyter":{"outputs_hidden":false}}
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:43.814961Z","iopub.execute_input":"2026-10-31T07:13:43.815397Z","iopub.status.idle":"2026-10-31T07:13:44.438853Z","shell.execute_reply.started":"2026-10-31T07:13:43.815358Z","shell.execute_reply":"2026-10-31T07:13:44.437511Z"},"jupyter":{"outputs_hidden":false}}
!ls -l

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:44.441238Z","iopub.execute_input":"2026-10-31T07:13:44.442177Z","iopub.status.idle":"2026-10-31T07:13:50.330109Z","shell.execute_reply.started":"2026-10-31T07:13:44.442127Z","shell.execute_reply":"2026-10-31T07:13:50.328776Z"},"jupyter":{"outputs_hidden":false}}
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:50.331646Z","iopub.execute_input":"2026-10-31T07:13:50.332578Z","iopub.status.idle":"2026-10-31T07:13:57.440534Z","shell.execute_reply.started":"2026-10-31T07:13:50.332541Z","shell.execute_reply":"2026-10-31T07:13:57.439004Z"},"jupyter":{"outputs_hidden":false}}
# !pip install tensorflow tqdm

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:57.442350Z","iopub.execute_input":"2026-10-31T07:13:57.442751Z","iopub.status.idle":"2026-10-31T07:13:57.771831Z","shell.execute_reply.started":"2026-10-31T07:13:57.442713Z","shell.execute_reply":"2026-10-31T07:13:57.770546Z"},"jupyter":{"outputs_hidden":false}}
import urllib.request
url = (
     "https://raw.githubusercontent.com/rasbt/" 
    "LLMs-from-scratch/main/ch05/" 
    "01_main-chapter-code/gpt_download.py" 
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:57.773063Z","iopub.execute_input":"2026-10-31T07:13:57.773581Z","iopub.status.idle":"2026-10-31T07:13:58.594756Z","shell.execute_reply.started":"2026-10-31T07:13:57.773542Z","shell.execute_reply":"2026-10-31T07:13:58.593314Z"},"jupyter":{"outputs_hidden":false}}
!ls -l

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:13:58.596928Z","iopub.execute_input":"2026-10-31T07:13:58.597493Z","iopub.status.idle":"2026-10-31T07:14:05.073548Z","shell.execute_reply.started":"2026-10-31T07:13:58.597433Z","shell.execute_reply":"2026-10-31T07:14:05.071769Z"},"jupyter":{"outputs_hidden":false}}
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:05.075051Z","iopub.execute_input":"2026-10-31T07:14:05.075523Z","iopub.status.idle":"2026-10-31T07:14:05.082658Z","shell.execute_reply.started":"2026-10-31T07:14:05.075483Z","shell.execute_reply":"2026-10-31T07:14:05.081462Z"},"jupyter":{"outputs_hidden":false}}
print("Settings: ", settings)
print("Parameter dictionary keys:", params.keys())

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:05.084006Z","iopub.execute_input":"2026-10-31T07:14:05.084430Z","iopub.status.idle":"2026-10-31T07:14:05.115440Z","shell.execute_reply.started":"2026-10-31T07:14:05.084377Z","shell.execute_reply":"2026-10-31T07:14:05.114400Z"},"jupyter":{"outputs_hidden":false}}
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:05.116966Z","iopub.execute_input":"2026-10-31T07:14:05.117372Z","iopub.status.idle":"2026-10-31T07:14:07.196280Z","shell.execute_reply.started":"2026-10-31T07:14:05.117331Z","shell.execute_reply":"2026-10-31T07:14:07.195079Z"},"jupyter":{"outputs_hidden":false}}
model_configs = { 
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}, 
}
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:07.197634Z","iopub.execute_input":"2026-10-31T07:14:07.197977Z","iopub.status.idle":"2026-10-31T07:14:07.205012Z","shell.execute_reply.started":"2026-10-31T07:14:07.197950Z","shell.execute_reply":"2026-10-31T07:14:07.203870Z"},"jupyter":{"outputs_hidden":false}}
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                        "Right: {right.shape}"
                        )
    return torch.nn.Parameter(torch.tensor(right))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:07.206617Z","iopub.execute_input":"2026-10-31T07:14:07.206996Z","iopub.status.idle":"2026-10-31T07:14:07.426058Z","shell.execute_reply.started":"2026-10-31T07:14:07.206966Z","shell.execute_reply":"2026-10-31T07:14:07.425055Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])): 
        q_w, k_w, v_w = np.split( 
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1) 
        gpt.trf_blocks[b].att.W_query.weight = assign( 
            gpt.trf_blocks[b].att.W_query.weight, q_w.T) 
        gpt.trf_blocks[b].att.W_key.weight = assign( 
            gpt.trf_blocks[b].att.W_key.weight, k_w.T) 
        gpt.trf_blocks[b].att.W_value.weight = assign( 
            gpt.trf_blocks[b].att.W_value.weight, v_w.T) 
        q_b, k_b, v_b = np.split( 
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1) 
        gpt.trf_blocks[b].att.W_query.bias = assign( 
            gpt.trf_blocks[b].att.W_query.bias, q_b) 
        gpt.trf_blocks[b].att.W_key.bias = assign( 
            gpt.trf_blocks[b].att.W_key.bias, k_b) 
        gpt.trf_blocks[b].att.W_value.bias = assign( 
            gpt.trf_blocks[b].att.W_value.bias, v_b) 
        gpt.trf_blocks[b].att.out_proj.weight = assign( 
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T) 
        gpt.trf_blocks[b].att.out_proj.bias = assign( 
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"]) 
        gpt.trf_blocks[b].ff.layers[0].weight = assign( 
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T) 
        gpt.trf_blocks[b].ff.layers[0].bias = assign( 
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"]) 
        gpt.trf_blocks[b].ff.layers[2].weight = assign( 
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T) 
        gpt.trf_blocks[b].ff.layers[2].bias = assign( 
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"]) 
        gpt.trf_blocks[b].norm1.scale = assign( 
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"]) 
        gpt.trf_blocks[b].norm1.shift = assign( 
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"]) 
        gpt.trf_blocks[b].norm2.scale = assign( 
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"]) 
        gpt.trf_blocks[b].norm2.shift = assign( 
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"]) 
gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:07.427564Z","iopub.execute_input":"2026-10-31T07:14:07.428003Z","iopub.status.idle":"2026-10-31T07:14:10.385182Z","shell.execute_reply.started":"2026-10-31T07:14:07.427970Z","shell.execute_reply":"2026-10-31T07:14:10.384030Z"},"jupyter":{"outputs_hidden":false}}
load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Who are you?", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
    
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %% [markdown]
# 微调语言模型最常见的方法是指令微调和分类微**

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:10.386778Z","iopub.execute_input":"2026-10-31T07:14:10.387251Z","iopub.status.idle":"2026-10-31T07:14:10.399872Z","shell.execute_reply.started":"2026-10-31T07:14:10.387208Z","shell.execute_reply":"2026-10-31T07:14:10.398381Z"}}
import urllib.request
import zipfile
import os
from pathlib import Path

url =  "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
    url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists, Skipping download"
             "and extraction."
             )
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")
    

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
#为什么下载文件必须用 "wb"？因为 urllib.request.urlopen(url).read() 抓取回来的是一个 .zip 压缩包。压缩包是由二进制字节组成的（里面包含了图片、代码、特殊的压缩头等），它不是文本。
#如果你用普通的 "w" 模式（文本写入模式）去写它：
#Python 会尝试把字节解码为字符串，或者自动把 \n 转换成 \r\n。
#这会导致压缩包内部的字节流被破坏，下载下来的文件会损坏，解压时会报错（如 BadZipFile）。


# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:10.401283Z","iopub.execute_input":"2026-10-31T07:14:10.401720Z","iopub.status.idle":"2026-10-31T07:14:10.993167Z","shell.execute_reply.started":"2026-10-31T07:14:10.401681Z","shell.execute_reply":"2026-10-31T07:14:10.991835Z"}}
!ls -l


# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:10.995214Z","iopub.execute_input":"2026-10-31T07:14:10.995819Z","iopub.status.idle":"2026-10-31T07:14:11.058653Z","shell.execute_reply.started":"2026-10-31T07:14:10.995764Z","shell.execute_reply":"2026-10-31T07:14:11.057407Z"}}
import pandas as pd
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
) #每行的 ham 或 spam 后面，跟着一个真实的制表符（即按一下键盘上的 Tab 键，在代码里写作 \t），然后再跟着具体的短信内容。
df
print(df["Label"].value_counts())
#其他常见的分隔符：sep=","：默认，CSV 文件（Comma-Separated Values）。
#sep=";"：有些欧洲的 CSV 会用分号（因为他们的逗号是小数点）。
#sep=r"\s+"：用一个或多个空格作为分隔符（常用于对齐的 txt 文本）。


# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.060180Z","iopub.execute_input":"2026-10-31T07:14:11.061410Z","iopub.status.idle":"2026-10-31T07:14:11.084589Z","shell.execute_reply.started":"2026-10-31T07:14:11.061359Z","shell.execute_reply":"2026-10-31T07:14:11.083396Z"}}
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())


# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.085992Z","iopub.execute_input":"2026-10-31T07:14:11.086463Z","iopub.status.idle":"2026-10-31T07:14:11.102411Z","shell.execute_reply.started":"2026-10-31T07:14:11.086422Z","shell.execute_reply":"2026-10-31T07:14:11.100583Z"}}
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.104306Z","iopub.execute_input":"2026-10-31T07:14:11.104883Z","iopub.status.idle":"2026-10-31T07:14:11.146260Z","shell.execute_reply.started":"2026-10-31T07:14:11.104838Z","shell.execute_reply":"2026-10-31T07:14:11.144977Z"}}
def random_split(df, train_frac, validation_frac):
    df = df.sample( #随机取样
        frac=1, random_state=123 #抽取的比例frac
    ).reset_index(drop=True) #重建索引编号
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df
train_df, validation_df, test_df = random_split(
    balanced_df, 0.7, 0.1)


train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
    

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.147880Z","iopub.execute_input":"2026-10-31T07:14:11.149102Z","iopub.status.idle":"2026-10-31T07:14:11.686560Z","shell.execute_reply.started":"2026-10-31T07:14:11.149058Z","shell.execute_reply":"2026-10-31T07:14:11.685256Z"}}
!ls -l

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.688714Z","iopub.execute_input":"2026-10-31T07:14:11.689265Z","iopub.status.idle":"2026-10-31T07:14:11.697009Z","shell.execute_reply.started":"2026-10-31T07:14:11.689212Z","shell.execute_reply":"2026-10-31T07:14:11.695783Z"}}
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
#50256 is endoftext

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.698729Z","iopub.execute_input":"2026-10-31T07:14:11.699141Z","iopub.status.idle":"2026-10-31T07:14:11.723988Z","shell.execute_reply.started":"2026-10-31T07:14:11.699109Z","shell.execute_reply":"2026-10-31T07:14:11.722757Z"}}
import torch
from torch.utils.data import Dataset
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
            #如果不传 max_length：调用底部的 _longest_encoded_length()，找出最长的那条。
        else:
            self.max_length = max_length
        self.encoded_texts = [
            encoded_text[:self.max_length]#因为大模型最多只能看固定长度的词，或者为了节省显存，强制截断。
            for encoded_text in self.encoded_texts
        ]
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
            #DataLoader 要求同一个 batch 内的数据形状必须一致。所以短于 max_length 的句子必须被填充
        ]
    def __getitem__(self, index):
                encoded = self.encoded_texts[index]
                label= self.data.iloc[index]["Label"]
                return (
                    torch.tensor(encoded, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long)
                )
    def __len__(self):
        return len(self.data)
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.725637Z","iopub.execute_input":"2026-10-31T07:14:11.726100Z","iopub.status.idle":"2026-10-31T07:14:11.799708Z","shell.execute_reply.started":"2026-10-31T07:14:11.726060Z","shell.execute_reply":"2026-10-31T07:14:11.798602Z"}}
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
print(train_dataset.max_length)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.800962Z","iopub.execute_input":"2026-10-31T07:14:11.801322Z","iopub.status.idle":"2026-10-31T07:14:11.829917Z","shell.execute_reply.started":"2026-10-31T07:14:11.801294Z","shell.execute_reply":"2026-10-31T07:14:11.828721Z"}}
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset= SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
print("val_dataset len:", len(val_dataset))
print("test_dataset len:", len(test_dataset))
print("val_dataset max length:", val_dataset.max_length)
print("test_dataset max length:", test_dataset.max_length)

# %% [markdown]
# 

# %% [code]


# %% [code]


# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.831501Z","iopub.execute_input":"2026-10-31T07:14:11.832006Z","iopub.status.idle":"2026-10-31T07:14:11.858821Z","shell.execute_reply.started":"2026-10-31T07:14:11.831963Z","shell.execute_reply":"2026-10-31T07:14:11.857785Z"}}
from torch.utils.data import DataLoader

num_workers=0 #控制“多进程读取数据”的核心开关,实际大于0
batch_size=8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    #shuffle=False
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    #shuffle=False
    drop_last=False
)

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:11.860047Z","iopub.execute_input":"2026-10-31T07:14:11.860912Z","iopub.status.idle":"2026-10-31T07:14:12.002536Z","shell.execute_reply.started":"2026-10-31T07:14:11.860867Z","shell.execute_reply":"2026-10-31T07:14:12.001160Z"}}
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions:", target_batch.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:12.004170Z","iopub.execute_input":"2026-10-31T07:14:12.004656Z","iopub.status.idle":"2026-10-31T07:14:12.011399Z","shell.execute_reply.started":"2026-10-31T07:14:12.004614Z","shell.execute_reply":"2026-10-31T07:14:12.009945Z"}}
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:12.012696Z","iopub.execute_input":"2026-10-31T07:14:12.013173Z","iopub.status.idle":"2026-10-31T07:14:12.036439Z","shell.execute_reply.started":"2026-10-31T07:14:12.013140Z","shell.execute_reply":"2026-10-31T07:14:12.035146Z"}}
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effor moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},

}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:12.037951Z","iopub.execute_input":"2026-10-31T07:14:12.038409Z","iopub.status.idle":"2026-10-31T07:14:20.229905Z","shell.execute_reply.started":"2026-10-31T07:14:12.038365Z","shell.execute_reply":"2026-10-31T07:14:20.228848Z"}}
from gpt_download import download_and_load_gpt2
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
setings, params = download_and_load_gpt2(
    model_size = model_size, models_dir = "gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:14:20.231553Z","iopub.execute_input":"2026-10-31T07:14:20.232033Z","iopub.status.idle":"2026-10-31T07:14:21.562942Z","shell.execute_reply.started":"2026-10-31T07:14:20.231992Z","shell.execute_reply":"2026-10-31T07:14:21.561749Z"}}
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"],
    ##############
    #top_k=50,
    #temperature=1.0
)
print(token_ids_to_text(token_ids, tokenizer))

# %% [code] {"execution":{"iopub.status.busy":"2026-10-31T07:19:04.860855Z","iopub.execute_input":"2026-10-31T07:19:04.861277Z","iopub.status.idle":"2026-10-31T07:19:09.419963Z","shell.execute_reply.started":"2026-10-31T07:19:04.861246Z","shell.execute_reply":"2026-10-31T07:19:09.418877Z"}}
text_2 = ( 
    "Is the following text 'spam'? Answer with 'yes' or 'no':" 
    " 'You are a winner you have been specially" 
    " selected to receive $1000 cash or a $2000 award.'" 
) 
token_ids = generate( 
    model=model, 
    idx=text_to_token_ids(text_2, tokenizer), 
    max_new_tokens=23, 
    context_size=BASE_CONFIG["context_length"],
    # top_k=50,
    # temperature=1.0
) 
print(token_ids_to_text(token_ids, tokenizer))