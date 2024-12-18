from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch
import torch.nn as nn
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


name = ["Llama3.2 1B  ", "Qwen2.5 0.5B"]
mode_path = [
    "E:/pretrained_models/LLM-Research/Llama-3___2-1B-Instruct",
    "E:\pretrained_models\Qwen\Qwen2___5-0___5B-Instruct"]
data_path = "E:/datasets/mgsm"


def count_zh(text: str) -> int:
    zh_pattern = re.findall(r'[\u4e00-\u9fff，。？！：；“”‘’（）《》【】]', text)
    return len(zh_pattern)


def main_work(path):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype="auto", trust_remote_code=True).eval()
    
    NUM_LAYERS = model.config.num_hidden_layers
    
    zh_table = [0] * NUM_LAYERS
    al_table = [0] * NUM_LAYERS
    entropys = [0] * NUM_LAYERS
    hooks = [] # store hooks
    
    # print(model)
    def CustomHook(modules: nn.Module, inp, out, layer_num):
        with torch.no_grad():
            x = out[0][:, :-1, :].detach().clone()
            if x.device != model.device:
                x = x.to(model.device)

            x = model.model.norm(x)
            logits = model.lm_head(x).float()
            sublogprobs = torch.log_softmax(logits, dim=-1)
            entropy = -torch.sum(sublogprobs * torch.exp(sublogprobs), dim=-1).mean()
            input_ids = logits.argmax(dim=-1, keepdim=True)
            entropys[layer_num] += entropy.item()
            
            zh, al, s = 0, 0, ""
            for id in input_ids[0]:
                generate_str = tokenizer.decode(
                    id, skip_special_tokens=True)   
                zh += 1 if count_zh(generate_str) else 0
                al += 1
                s += generate_str
            # print(
            #     repr(f"layer: {layer_num}, generate: {s}, count_zh: {zh}, length: {al}"))
            zh_table[layer_num] += zh
            al_table[layer_num] += al
        return out

    for i in range(NUM_LAYERS):
        # register hook
        hook_fn = lambda modules, inp, out, layer_num=i: CustomHook(
            modules, inp, out, layer_num)
        hook = model.model.layers[i].register_forward_hook(hook_fn)
        hooks.append(hook)
    
    ds = load_dataset(data_path, "zh", trust_remote_code=True, split="test")["question"]
    for x in tqdm(ds):
        y = model(**tokenizer(x, return_tensors="pt", add_special_tokens=False).to(model.device)).logits
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return zh_table, al_table, entropys


if __name__ == "__main__":
    fig, ax = plt.subplots()
    for i in range(len(mode_path)):
        print(f"Processing {name[i]}")
        x, y, en = main_work(mode_path[i])
        x_values = np.arange(len(x)) 
        y_values = [x[i] / y[i] for i in range(len(x))]
        max_en = max(en)
        en_values = [en[i] / max_en for i in range(len(en))]
        # smooth en_values
        en_value_smooth = np.cumsum(en_values) / (np.arange(1, len(en_values) + 1))
        ax.plot(x_values, y_values, label=name[i] + " zh ratio", marker='o', linestyle='-')
        ax.plot(x_values, en_values, label=name[i] + " entropy", marker='o', linestyle='-')
    ax.grid()
    ax.set_xlabel("Layer Number")
    ax.set_ylabel("zh word ratio or entropy")
    ax.legend()
    plt.savefig('./pic/total.png', dpi=500)
    plt.show()