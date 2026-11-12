# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

GPT_BASIC_CONFIG = {
    "model": "gpt2",
    "max_length": 1024,
    "n_layers": 12,
    "n_heads": 12,
    "n_embd": 768,
    "dropout": 0.0,
    "temperature": 0.7,
}

MODEL_LIST = {
    {"model": "gpt2", "config": GPT_BASIC_CONFIG, "params": 117M },
    {"model": "gpt2-medium", "config": GPT_BASIC_CONFIG, "params": 345M},
    {"model": "gpt2-large", "config": GPT_BASIC_CONFIG, "params": 762M},
    {"model": "gpt2-xl", "config": GPT_BASIC_CONFIG, "params": 1558M},
}



def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
