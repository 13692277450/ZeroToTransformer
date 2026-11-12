import webdataset as wds
import json
import re

import pandas as pd
import tiktoken
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import logging
from rich.logging import RichHandler
from livelossplot import PlotLosses
from torch.utils.tensorboard import SummaryWriter
# from mainScore import format_input, check_if_running, generate_model_scores, query_model, console
from rich.console import Console
console = Console()







############################ MODEL CONFIG ##########################

GPT_CONFIG_124M = {
    "model": "gpt2",
    "context_length": 1024,
    "n_layers": 12,
    "n_heads": 12,
    "n_embd": 768,
    "dropout": 0.0,
    "temperature": 0.7,
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": "768", "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": "1024", "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": "1280", "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": "1600", "n_layers": 48, "n_heads": 25},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
model_name = "gpt2-medium (355M)"
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

############################ GPT MODEL ##########################
class GPT_MODEL(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_dim = int(config["emb_dim"])
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.dropout = config["dropout"]
        self.temperature = config["temperature"]
        self.context_length = config["context_length"]
        
        

    def forward(self, x):
        """
        前向传播
        :param x: 输入序列，形状为 (batch_size, context_length)
        :return: 输出序列，形状为 (batch_size, context_length, n_embd)
        """
        # x = self.embedding(x)
        # x = self.dropout(x)
        # x = self.emb_dim(x)
        return x
        


model = GPT_MODEL(NEW_CONFIG)
model.eval()
model.to(device)
############################ WebDataset ############################
def create_text_webdataset(df, ouput_pattern, max_samples_per_share=10000):
    """
    把 DataFrame 转换成 WebDataset 的 .tar 分片。
    
    :param df: 包含文本数据的 Pandas DataFrame
    :param output_pattern: 输出路径模式，例如 "shards/retail-%06d.tar"
    :param max_samples_per_shard: 每个分片包含的样本数
    """
    # SharedWriter 会自动按maxcount分片
    with wds.ShardWriter(ouput_pattern, maxcount=max_samples_per_share) as sink: # type: ignore
        # 遍历 DataFrame 中的每一行
        for idx, row in df.iterrows():
            text = (
                    f"InvoiceNo: {row['InvoiceNo']}\n"
                    f"Description: {row['Description']}\n"
                    f"Quantity: {row['Quantity']}\n"
                    f"Country: {row['Country']}\n"
                    f"InvoiceDate: {row['InvoiceDate']}\n"
                    f"CustomerID: {row['CustomerID']}\n"
                    f"StockCode: {row['StockCode']}\n"
                    f"UnitPrice: {row['UnitPrice']}\n"
                )
            textBatch = {
                "__key__": f"textBatch_{idx:08d}",
                "text": text.encode("utf-8")
            }
            sink.write(textBatch)

############################Prepare Data############################
class OnlineRetailDataset(Dataset):
    def __init__(self, data):
        # 初始化数据集， 读取数据文件 data
        self.data = pd.read_csv(data)
        """
        准备数据文件。
        :param file_path: 保存的文件路径
        :param url: 数据文件的 URL
        """
        # Prepare 数据dataset文件
    def __len__(self):
        # 返回数据集的样本数量
        return len(self.data)
    def __getitem__(self, idx):
        # 获取数据集的第 idx 个样本
        text = (
            f"InvoiceNo: {self.data.iloc[idx]['InvoiceNo']}\n"
            f"Description: {self.data.iloc[idx]['Description']}\n"
            f"Quantity: {self.data.iloc[idx]['Quantity']}\n"
            f"Country: {self.data.iloc[idx]['Country']}\n"
            f"InvoiceDate: {self.data.iloc[idx]['InvoiceDate']}\n"
            f"CustomerID: {self.data.iloc[idx]['CustomerID']}\n"
            f"StockCode: {self.data.iloc[idx]['StockCode']}\n"
            f"UnitPrice: {self.data.iloc[idx]['UnitPrice']}\n"
        )
        # 编码文本
        ids = tokenizer.encode(text)
        return {"text": text}
# class StreamingRetailDataset(IterableDataset):
#     def __init__(self, file_path, total_size, split="train",chunk_size=10000):
#         self.file_path = file_path
#         self.chunk_size = chunk_size
#         self.total_size = total_size
#         self.split = split
#         self.data = pd.read_csv(file_path, chunksize=chunk_size)
#     def __len__(self):
#         return self.total_size
#     def __iter__(self):
#         for chunk in self.data:
#             for _, row in chunk.iterrows():
#                 text = (
#                         f"InvoiceNo: {row['InvoiceNo']}\n"
#                         f"Description: {row['Description']}\n"
#                         f"Quantity: {row['Quantity']}\n"
#                         f"Country: {row['Country']}\n"
#                         f"InvoiceDate: {row['InvoiceDate']}\n"
#                         f"CustomerID: {row['CustomerID']}\n"
#                         f"StockCode: {row['StockCode']}\n"
#                         f"UnitPrice: {row['UnitPrice']}\n"
#                     )
#                 yield {"text": text}
                
def prepare_data():
    # 检查文件路径是否存在
    # 打开数据文件 

    file_name = "D:\\Project\\AI\\ZeroToTransformer\\OnlineRetail.csv"
    with open(file_name, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)-1
    total_size = total_lines
    
    full_dataset = OnlineRetailDataset(file_name)  
    # full_dataset = StreamingRetailDataset(file_name, total_size)  

    total_size = len(full_dataset) 
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.1)
    test_size = int(total_size - train_size - val_size)
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    # 划分数据集
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 读取分割数据
    train_data= train_dataset[:train_size]
    val_data= val_dataset[train_size:train_size+val_size]
    test_data = test_dataset[train_size+val_size:]
    
    # 返回数据
    console.print("-"*80)
    console.print(f"Total data: {total_size}", style="bold green")
    console.print("-"*80)

    console.print(f"Train data: {train_size}", style="bold yellow")
    console.print("-"*80)

    console.print(f"Val data: {val_size}", style="bold purple")
    console.print("-"*80)
    console.print(f"Test data: {test_size}", style="bold blue")
    return train_data, val_data, test_data

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

#########################Simple Plot chart#########################
plotlosses = PlotLosses()

def plot_chart(train_loss, val_loss, train_acc, val_acc, epoch):
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
        'val_loss': val_loss, 
        'train_acc': train_acc, 
        'val_acc': val_acc,
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

############################训练分类简单############################

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
# file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
file_name = "OnlineRetail-llama3.2-sft.pth"
torch.save(model.state_dict(), file_name)
logger.info(f"Model saved to {file_name}")




##########################MAIN############################

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    status = "Training"
    logger.info("Starting the training process...")
    if status == "Training":
        prepare_data()
    else:   
        #  check_if_running("ollama")
        pass

    # generate_model_scores(test_data, "model_response")

