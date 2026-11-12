import urllib.request
import json
from tqdm import tqdm
import psutil
from rich.console import Console
console = Console()
##########################Check Ollama Running########################

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
        break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
        console.print("❌ Error: Ollama not running. Launch ollama before proceeding.", style="red")
        # exit(1)
        console.print("✅ Ollama running:", check_if_running("ollama") , style="bold green")

##########################Tqdm Status Bar########################
file_path = "instruction-data-with-response.json"

with open(file_path, "r") as file:
    test_data = json.load(file)

def format_input(entry):
    instruction_text = (
    f"Below is an instruction that describes a task."
    f"Write a response that appropriately completes the request."
    f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

##########################Conversation############################
def query_model(
    prompt,
    model="llama3.2",
    url="http://localhost:11434/api/chat"
):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed":123,
            "max_tokens": 256,
            "num_ctx": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url, 
        data=payload, 
        headers={"Content-Type": "application/json"}
        )
    request.add_header("Content-Type", "application/json")
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            print(line)
            response_json = json.loads(line)
            if "content" in response_json:
                response_data += response_json["message"]["content"]
        return response_data
model = "llama3.2"
result = query_model("what do Llams eat?", model=model)
print(result)

##########################Conversation############################


for entry in test_data[:3]:
    prompt = (
    f"Given the input '{format_input(entry)}' "
    f"and correct output {entry['output']}*, "
    f"score the model response '{entry['model_response']} '"
    f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n ---")
    
##########################MODEL SCORE############################
# def generate_model_scores(json_data, json_key, model="llama3.2"):
#     """
#     生成模型评分。
#     :param json_data: 包含数据的 JSON 对象
#     :param json_key: JSON 对象中包含问题的键
#     :param model: 模型名称
#     :param url: API URL
#     :return: 包含评分的 JSON 对象
#     """
    # 可以给不同的模型评分，或者给同一个模型不同的 prompt 评分
    # for entry in json_data:
    #     question = entry[json_key]
    #     prompt = f"Please provide a score for the following question:\n{question}"
    #     score = query_model(prompt, model=model, url=url)
    #     entry["model_score"] = score
    # return json_data
    

def generate_model_scores(json_data, json_key, model="llama3.2"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
        f"Given the input '{format_input(entry)}' "
        f"and correct output '{entry['output']}', "
        f"score the model response '{entry [json_key]}'"
        f" on a scale from 0 to 100, where 100 is the best score. "
        f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int (score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

        return scores

scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}") # pyright: ignore[reportArgumentType]
print(f"Average score: {sum(scores)/len(scores) :. 2f}\n") # pyright: ignore[reportCallIssue, reportArgumentType]    