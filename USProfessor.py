# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.622423Z","iopub.execute_input":"2026-11-04T15:38:58.623194Z","iopub.status.idle":"2026-11-04T15:38:58.629539Z","shell.execute_reply.started":"2026-11-04T15:38:58.623163Z","shell.execute_reply":"2026-11-04T15:38:58.628450Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.640726Z","iopub.execute_input":"2026-11-04T15:38:58.641327Z","iopub.status.idle":"2026-11-04T15:38:58.659154Z","shell.execute_reply.started":"2026-11-04T15:38:58.641298Z","shell.execute_reply":"2026-11-04T15:38:58.658138Z"}}
import os
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/" 
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/" 
       "the-verdict.txt")
if not os.path.exists("the-verdict.txt"):
    file_path = "the-verdict.txt"
    raw_text = urllib.request.urlretrieve(url, file_path)
print("the-verdict.txt downloaded...")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.664941Z","iopub.execute_input":"2026-11-04T15:38:58.665346Z","iopub.status.idle":"2026-11-04T15:38:58.681998Z","shell.execute_reply.started":"2026-11-04T15:38:58.665318Z","shell.execute_reply":"2026-11-04T15:38:58.680858Z"}}
with open("the-verdict.txt", "r",encoding="utf-8") as f:
    raw_text = f.read()
len(raw_text)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.700911Z","iopub.execute_input":"2026-11-04T15:38:58.701732Z","iopub.status.idle":"2026-11-04T15:38:58.833600Z","shell.execute_reply.started":"2026-11-04T15:38:58.701695Z","shell.execute_reply":"2026-11-04T15:38:58.832277Z"}}
!ls -l

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.836016Z","iopub.execute_input":"2026-11-04T15:38:58.836375Z","iopub.status.idle":"2026-11-04T15:38:58.843278Z","shell.execute_reply.started":"2026-11-04T15:38:58.836346Z","shell.execute_reply":"2026-11-04T15:38:58.842121Z"}}
import re

text = "Hello, world. This is a test."
result = re.split(r'(\s)', text)
print(result)


# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.844668Z","iopub.execute_input":"2026-11-04T15:38:58.845026Z","iopub.status.idle":"2026-11-04T15:38:58.867654Z","shell.execute_reply.started":"2026-11-04T15:38:58.844968Z","shell.execute_reply":"2026-11-04T15:38:58.866500Z"}}
result = re.split(r'([,.]|\s)', text)
print(result)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.870158Z","iopub.execute_input":"2026-11-04T15:38:58.870545Z","iopub.status.idle":"2026-11-04T15:38:58.890464Z","shell.execute_reply.started":"2026-11-04T15:38:58.870505Z","shell.execute_reply":"2026-11-04T15:38:58.889262Z"}}
result = [item for item in result if item.strip()]
#item.strip() 的作用是去掉字符串两端的空白字符。

print(result)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.891878Z","iopub.execute_input":"2026-11-04T15:38:58.892171Z","iopub.status.idle":"2026-11-04T15:38:58.930385Z","shell.execute_reply.started":"2026-11-04T15:38:58.892135Z","shell.execute_reply":"2026-11-04T15:38:58.929388Z"}}
text = "Hello, world. Is this -- a test?"

result = re. split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
preprocessed = result
len(preprocessed)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.931404Z","iopub.execute_input":"2026-11-04T15:38:58.931733Z","iopub.status.idle":"2026-11-04T15:38:58.953404Z","shell.execute_reply.started":"2026-11-04T15:38:58.931707Z","shell.execute_reply":"2026-11-04T15:38:58.952394Z"}}
preprocessed[:2]

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.954673Z","iopub.execute_input":"2026-11-04T15:38:58.955105Z","iopub.status.idle":"2026-11-04T15:38:58.979143Z","shell.execute_reply.started":"2026-11-04T15:38:58.955068Z","shell.execute_reply":"2026-11-04T15:38:58.977961Z"}}
all_words = sorted(set(preprocessed))
vocab_size=len(all_words)
print(vocab_size)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:58.980361Z","iopub.execute_input":"2026-11-04T15:38:58.980788Z","iopub.status.idle":"2026-11-04T15:38:58.999665Z","shell.execute_reply.started":"2026-11-04T15:38:58.980721Z","shell.execute_reply":"2026-11-04T15:38:58.998878Z"}}
vocab = {token: integer for integer, token in enumerate(all_words)}

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.002271Z","iopub.execute_input":"2026-11-04T15:38:59.002685Z","iopub.status.idle":"2026-11-04T15:38:59.024590Z","shell.execute_reply.started":"2026-11-04T15:38:59.002657Z","shell.execute_reply":"2026-11-04T15:38:59.023461Z"}}
vocab["Jack"]

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.028518Z","iopub.execute_input":"2026-11-04T15:38:59.028827Z","iopub.status.idle":"2026-11-04T15:38:59.046422Z","shell.execute_reply.started":"2026-11-04T15:38:59.028800Z","shell.execute_reply":"2026-11-04T15:38:59.045437Z"}}
int_to_str = {i:s for s,i in vocab.items()}
int_to_str[57]

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.047687Z","iopub.execute_input":"2026-11-04T15:38:59.048454Z","iopub.status.idle":"2026-11-04T15:38:59.074213Z","shell.execute_reply.started":"2026-11-04T15:38:59.048425Z","shell.execute_reply":"2026-11-04T15:38:59.073203Z"}}
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.075433Z","iopub.execute_input":"2026-11-04T15:38:59.075852Z","iopub.status.idle":"2026-11-04T15:38:59.086625Z","shell.execute_reply.started":"2026-11-04T15:38:59.075807Z","shell.execute_reply":"2026-11-04T15:38:59.085931Z"}}
vocab = {token: integer for integer, token in enumerate(all_words)}
#print(vocab)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.087934Z","iopub.execute_input":"2026-11-04T15:38:59.088181Z","iopub.status.idle":"2026-11-04T15:38:59.108232Z","shell.execute_reply.started":"2026-11-04T15:38:59.088159Z","shell.execute_reply":"2026-11-04T15:38:59.107235Z"}}
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text =" ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.109608Z","iopub.execute_input":"2026-11-04T15:38:59.110317Z","iopub.status.idle":"2026-11-04T15:38:59.127966Z","shell.execute_reply.started":"2026-11-04T15:38:59.110277Z","shell.execute_reply":"2026-11-04T15:38:59.126871Z"}}
tokenizer = SimpleTokenizerV1(vocab)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.128983Z","iopub.execute_input":"2026-11-04T15:38:59.129584Z","iopub.status.idle":"2026-11-04T15:38:59.147408Z","shell.execute_reply.started":"2026-11-04T15:38:59.129557Z","shell.execute_reply":"2026-11-04T15:38:59.146350Z"}}
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.148842Z","iopub.execute_input":"2026-11-04T15:38:59.149123Z","iopub.status.idle":"2026-11-04T15:38:59.171813Z","shell.execute_reply.started":"2026-11-04T15:38:59.149096Z","shell.execute_reply":"2026-11-04T15:38:59.171033Z"}}
ids = tokenizer.encode(text)
print(ids)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.173011Z","iopub.execute_input":"2026-11-04T15:38:59.173415Z","iopub.status.idle":"2026-11-04T15:38:59.193258Z","shell.execute_reply.started":"2026-11-04T15:38:59.173389Z","shell.execute_reply":"2026-11-04T15:38:59.192322Z"}}
tokenizer.decode(ids)

# %% [markdown]
# tokenizer.decode(tokenizer.encode(text))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.194523Z","iopub.execute_input":"2026-11-04T15:38:59.194902Z","iopub.status.idle":"2026-11-04T15:38:59.213533Z","shell.execute_reply.started":"2026-11-04T15:38:59.194864Z","shell.execute_reply":"2026-11-04T15:38:59.212681Z"}}
text ="Hello, do you like tea, is this--- a test?"
# tokenizer.encode(text)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.214969Z","iopub.execute_input":"2026-11-04T15:38:59.215327Z","iopub.status.idle":"2026-11-04T15:38:59.225326Z","shell.execute_reply.started":"2026-11-04T15:38:59.215287Z","shell.execute_reply":"2026-11-04T15:38:59.224337Z"}}
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
len(vocab)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.226622Z","iopub.execute_input":"2026-11-04T15:38:59.227057Z","iopub.status.idle":"2026-11-04T15:38:59.249647Z","shell.execute_reply.started":"2026-11-04T15:38:59.227030Z","shell.execute_reply":"2026-11-04T15:38:59.248541Z"}}
len(vocab.items())

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.250965Z","iopub.execute_input":"2026-11-04T15:38:59.251453Z","iopub.status.idle":"2026-11-04T15:38:59.269240Z","shell.execute_reply.started":"2026-11-04T15:38:59.251411Z","shell.execute_reply":"2026-11-04T15:38:59.268079Z"}}
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.270424Z","iopub.execute_input":"2026-11-04T15:38:59.270894Z","iopub.status.idle":"2026-11-04T15:38:59.289466Z","shell.execute_reply.started":"2026-11-04T15:38:59.270855Z","shell.execute_reply":"2026-11-04T15:38:59.288473Z"}}
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text =" ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.291022Z","iopub.execute_input":"2026-11-04T15:38:59.291367Z","iopub.status.idle":"2026-11-04T15:38:59.316467Z","shell.execute_reply.started":"2026-11-04T15:38:59.291334Z","shell.execute_reply":"2026-11-04T15:38:59.315496Z"}}
tokenizer = SimpleTokenizerV2(vocab)
tokenizer.encode(text)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.317649Z","iopub.execute_input":"2026-11-04T15:38:59.318600Z","iopub.status.idle":"2026-11-04T15:38:59.338771Z","shell.execute_reply.started":"2026-11-04T15:38:59.318563Z","shell.execute_reply":"2026-11-04T15:38:59.337648Z"}}
tokenizer.encode(text)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.343310Z","iopub.execute_input":"2026-11-04T15:38:59.343677Z","iopub.status.idle":"2026-11-04T15:38:59.360975Z","shell.execute_reply.started":"2026-11-04T15:38:59.343648Z","shell.execute_reply":"2026-11-04T15:38:59.359867Z"}}
tokenizer.decode(tokenizer.encode(text))

# %% [markdown]
# # 2.5 Byte pair coding

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.362258Z","iopub.execute_input":"2026-11-04T15:38:59.362931Z","iopub.status.idle":"2026-11-04T15:38:59.384631Z","shell.execute_reply.started":"2026-11-04T15:38:59.362906Z","shell.execute_reply":"2026-11-04T15:38:59.383532Z"}}
import tiktoken
tiktoken.__version__

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.385743Z","iopub.execute_input":"2026-11-04T15:38:59.386030Z","iopub.status.idle":"2026-11-04T15:38:59.403358Z","shell.execute_reply.started":"2026-11-04T15:38:59.386006Z","shell.execute_reply":"2026-11-04T15:38:59.402114Z"}}
tokenizer = tiktoken.get_encoding("gpt2")


# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.404828Z","iopub.execute_input":"2026-11-04T15:38:59.405180Z","iopub.status.idle":"2026-11-04T15:38:59.426045Z","shell.execute_reply.started":"2026-11-04T15:38:59.405146Z","shell.execute_reply":"2026-11-04T15:38:59.425154Z"}}
tokenizer.encode("Hello world")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.427274Z","iopub.execute_input":"2026-11-04T15:38:59.427957Z","iopub.status.idle":"2026-11-04T15:38:59.446287Z","shell.execute_reply.started":"2026-11-04T15:38:59.427928Z","shell.execute_reply":"2026-11-04T15:38:59.445287Z"}}
tokenizer.decode(tokenizer.encode("Hello world"))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.447659Z","iopub.execute_input":"2026-11-04T15:38:59.448176Z","iopub.status.idle":"2026-11-04T15:38:59.468721Z","shell.execute_reply.started":"2026-11-04T15:38:59.448141Z","shell.execute_reply":"2026-11-04T15:38:59.467837Z"}}
text =(
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."

)

tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# %% [markdown]
# # 2.6 Data sampling with sliding windows

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.469995Z","iopub.execute_input":"2026-11-04T15:38:59.470303Z","iopub.status.idle":"2026-11-04T15:38:59.497644Z","shell.execute_reply.started":"2026-11-04T15:38:59.470278Z","shell.execute_reply":"2026-11-04T15:38:59.496564Z"}}
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.498828Z","iopub.execute_input":"2026-11-04T15:38:59.499122Z","iopub.status.idle":"2026-11-04T15:38:59.504679Z","shell.execute_reply.started":"2026-11-04T15:38:59.499098Z","shell.execute_reply":"2026-11-04T15:38:59.503634Z"}}
enc_sample = enc_text[50:]

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.505897Z","iopub.execute_input":"2026-11-04T15:38:59.506153Z","iopub.status.idle":"2026-11-04T15:38:59.526912Z","shell.execute_reply.started":"2026-11-04T15:38:59.506131Z","shell.execute_reply":"2026-11-04T15:38:59.525870Z"}}
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size]
print(f"x: {x}")
print(f"y:        {y}")

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.528219Z","iopub.execute_input":"2026-11-04T15:38:59.528565Z","iopub.status.idle":"2026-11-04T15:38:59.547833Z","shell.execute_reply.started":"2026-11-04T15:38:59.528532Z","shell.execute_reply":"2026-11-04T15:38:59.546939Z"}}
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "------>", desired)
    print(tokenizer.decode(context), "------>", tokenizer.decode([desired]))


# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.549563Z","iopub.execute_input":"2026-11-04T15:38:59.550014Z","iopub.status.idle":"2026-11-04T15:38:59.569245Z","shell.execute_reply.started":"2026-11-04T15:38:59.549971Z","shell.execute_reply":"2026-11-04T15:38:59.568453Z"}}
import torch
torch.__version__

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.570513Z","iopub.execute_input":"2026-11-04T15:38:59.570876Z","iopub.status.idle":"2026-11-04T15:38:59.587792Z","shell.execute_reply.started":"2026-11-04T15:38:59.570843Z","shell.execute_reply":"2026-11-04T15:38:59.587000Z"}}
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
     def __init__(self, txt, tokenizer, max_length, stride):
         self.input_ids = []
         self.target_ids = []
         token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
         for i in range(0, len(token_ids) - max_length, stride):
             input_chunk = token_ids[i:i + max_length]
             target_chunk  = token_ids[i + 1: i + max_length + 1]
             self.input_ids.append(torch.tensor(input_chunk))
             self.target_ids.append(torch.tensor(target_chunk))
     def __len__(self):
         return len(self.input_ids)
     def __getitem__(self, idx):
         return self.input_ids[idx], self.target_ids[idx]


# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.589248Z","iopub.execute_input":"2026-11-04T15:38:59.589492Z","iopub.status.idle":"2026-11-04T15:38:59.618291Z","shell.execute_reply.started":"2026-11-04T15:38:59.589470Z","shell.execute_reply":"2026-11-04T15:38:59.617147Z"}}
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True, #如果最后一批是一个数据的话
                         #会导致损失急剧上升训练不稳定
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

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.620203Z","iopub.execute_input":"2026-11-04T15:38:59.620688Z","iopub.status.idle":"2026-11-04T15:38:59.646593Z","shell.execute_reply.started":"2026-11-04T15:38:59.620643Z","shell.execute_reply":"2026-11-04T15:38:59.645376Z"}}
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.647998Z","iopub.execute_input":"2026-11-04T15:38:59.648295Z","iopub.status.idle":"2026-11-04T15:38:59.676739Z","shell.execute_reply.started":"2026-11-04T15:38:59.648271Z","shell.execute_reply":"2026-11-04T15:38:59.675710Z"}}
dataloader = create_dataloader_v1(
    raw_text, batch_size = 1, max_length=4, stride=4, shuffle=False#, drop_last=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.677986Z","iopub.execute_input":"2026-11-04T15:38:59.678580Z","iopub.status.idle":"2026-11-04T15:38:59.684733Z","shell.execute_reply.started":"2026-11-04T15:38:59.678545Z","shell.execute_reply":"2026-11-04T15:38:59.684025Z"}}
second_batch = next(data_iter)
print(second_batch)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.685916Z","iopub.execute_input":"2026-11-04T15:38:59.686472Z","iopub.status.idle":"2026-11-04T15:38:59.728715Z","shell.execute_reply.started":"2026-11-04T15:38:59.686431Z","shell.execute_reply":"2026-11-04T15:38:59.728060Z"}}
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
                                  #, drop_last=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# %% [markdown]
# # 2.7 Create tokenize embeddings.# 

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.729993Z","iopub.execute_input":"2026-11-04T15:38:59.730483Z","iopub.status.idle":"2026-11-04T15:38:59.736245Z","shell.execute_reply.started":"2026-11-04T15:38:59.730458Z","shell.execute_reply":"2026-11-04T15:38:59.735328Z"}}
tokenizer.n_vocab

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.737394Z","iopub.execute_input":"2026-11-04T15:38:59.737777Z","iopub.status.idle":"2026-11-04T15:38:59.757403Z","shell.execute_reply.started":"2026-11-04T15:38:59.737739Z","shell.execute_reply":"2026-11-04T15:38:59.756425Z"}}
dir(tokenizer)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.758671Z","iopub.execute_input":"2026-11-04T15:38:59.759325Z","iopub.status.idle":"2026-11-04T15:38:59.777014Z","shell.execute_reply.started":"2026-11-04T15:38:59.759294Z","shell.execute_reply":"2026-11-04T15:38:59.775992Z"}}
input_ids = torch.tensor([2,3,5,1])

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.781757Z","iopub.execute_input":"2026-11-04T15:38:59.782189Z","iopub.status.idle":"2026-11-04T15:38:59.799717Z","shell.execute_reply.started":"2026-11-04T15:38:59.782157Z","shell.execute_reply":"2026-11-04T15:38:59.798853Z"}}
vocab_size =6
output_dim =3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.801213Z","iopub.execute_input":"2026-11-04T15:38:59.801743Z","iopub.status.idle":"2026-11-04T15:38:59.820495Z","shell.execute_reply.started":"2026-11-04T15:38:59.801718Z","shell.execute_reply":"2026-11-04T15:38:59.819468Z"}}
print(embedding_layer.weight)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.821805Z","iopub.execute_input":"2026-11-04T15:38:59.822143Z","iopub.status.idle":"2026-11-04T15:38:59.842002Z","shell.execute_reply.started":"2026-11-04T15:38:59.822109Z","shell.execute_reply":"2026-11-04T15:38:59.840945Z"}}
embedding_layer(torch.tensor([3]))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.843265Z","iopub.execute_input":"2026-11-04T15:38:59.843858Z","iopub.status.idle":"2026-11-04T15:38:59.860691Z","shell.execute_reply.started":"2026-11-04T15:38:59.843822Z","shell.execute_reply":"2026-11-04T15:38:59.859734Z"}}
embedding_layer(torch.tensor([2]))

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.861996Z","iopub.execute_input":"2026-11-04T15:38:59.862360Z","iopub.status.idle":"2026-11-04T15:38:59.883557Z","shell.execute_reply.started":"2026-11-04T15:38:59.862329Z","shell.execute_reply":"2026-11-04T15:38:59.882447Z"}}
input_ids

# %% [markdown]
# # 2.8 Encoding word positions

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:38:59.885253Z","iopub.execute_input":"2026-11-04T15:38:59.885634Z","iopub.status.idle":"2026-11-04T15:39:00.017100Z","shell.execute_reply.started":"2026-11-04T15:38:59.885609Z","shell.execute_reply":"2026-11-04T15:39:00.016355Z"}}
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:39:00.018059Z","iopub.execute_input":"2026-11-04T15:39:00.018287Z","iopub.status.idle":"2026-11-04T15:39:00.043322Z","shell.execute_reply.started":"2026-11-04T15:39:00.018265Z","shell.execute_reply":"2026-11-04T15:39:00.042424Z"}}
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:39:00.044239Z","iopub.execute_input":"2026-11-04T15:39:00.044490Z","iopub.status.idle":"2026-11-04T15:39:00.051250Z","shell.execute_reply.started":"2026-11-04T15:39:00.044469Z","shell.execute_reply":"2026-11-04T15:39:00.050275Z"}}
print("TokenID:\n", inputs)
print("\nInputes shaep:\n", inputs.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:39:00.052472Z","iopub.execute_input":"2026-11-04T15:39:00.052833Z","iopub.status.idle":"2026-11-04T15:39:00.074937Z","shell.execute_reply.started":"2026-11-04T15:39:00.052795Z","shell.execute_reply":"2026-11-04T15:39:00.073834Z"}}
token_embeddings = token_embedding_layer(inputs)
token_embeddings.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-11-04T15:52:17.968631Z","iopub.execute_input":"2026-11-04T15:52:17.968995Z","iopub.status.idle":"2026-11-04T15:52:17.974624Z","shell.execute_reply.started":"2026-11-04T15:52:17.968969Z","shell.execute_reply":"2026-11-04T15:52:17.973850Z"}}
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)