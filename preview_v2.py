from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteriaList, EosTokenCriteria, MaxLengthCriteria
import re
import torch
import torch.nn as nn
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


name = ["Qwen2.5 0.5B"]
mode_path = [
    # "E:/pretrained_models/LLM-Research/Llama-3___2-1B-Instruct"]
    "E:\pretrained_models\Qwen\Qwen2___5-0___5B-Instruct"]
data_path = "E:/datasets/mgsm"

lang = "en" # "zh", "en", "ja"


def count(text: str, lang="zh") -> int:
    if lang == "zh":
        pattern = re.findall(r'[\u4e00-\u9fff，。？！：；“”‘’（）《》【】]', text)
    elif lang == "en":
        pattern = re.findall(r"[a-zA-Z0-9,.;!?\"'(){}\[\]<>/-]", text)
    elif lang == "ja":
        pattern = re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uFF66-\uFF9F々〆〤、。！？「」『』【】]", text)
    else: raise NotImplementedError
    return len(pattern)



def main_work(path, prefix="Qwen2.5 0.5B"):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype="auto", trust_remote_code=True).eval()
    
    NUM_LAYERS = model.config.num_hidden_layers
    
    zh_table = [0] * NUM_LAYERS
    al_table = [0] * NUM_LAYERS
    entropys = [0] * NUM_LAYERS


    # Custom Greedy Decoding
    def CustomDecode(
        input_ids,
        model,
        tokenizer,
        max_length,
        do_sample=False
    ):
        r"""
            Results: List[List[str]] For logits len
        """
        stop_criteria = StoppingCriteriaList()
        stop_criteria.append(MaxLengthCriteria(max_length=max_length))
        stop_criteria.append(EosTokenCriteria(eos_token_id=tokenizer.eos_token_id))
        
        bsz = input_ids.shape[0]
        Results = [[[] for __ in range(bsz)] for _ in range(NUM_LAYERS)]
        
        unfinished_sequences = torch.ones(bsz, dtype=torch.long, device=input_ids.device)
        
        calc = True
        while True:
            torch.cuda.empty_cache()
            outputs = model(
                input_ids=input_ids, output_hidden_states=True, return_dict=True
            )
            
            
            logits = outputs.logits[:, -1, :] # batch_size x vocab_size
            x = torch.stack(outputs.hidden_states[1:-1], dim=0) # num_layers - 1 x batch_size x seq_len x hidden_size
            
            if calc:
                logits_temp = model.lm_head(model.model.norm(x.clone())) # num_layers - 1 x batch_size x seq_len x vocab_size
                logits_temp = torch.cat(
                    [logits_temp, outputs.logits[None, :]], dim=0
                ) # num_layers x batch_size x seq_len x vocab_size
                # print(logits_temp.shape)
                logprobs = torch.log_softmax(logits_temp.float(), dim=-1)
                # print(logprobs.shape)
                entropy = -torch.sum(torch.exp(logprobs) * logprobs, dim=-1).reshape(NUM_LAYERS, -1)
                entropy = entropy.mean(dim=-1)
                assert entropy.shape[0] == NUM_LAYERS and len(entropy.shape) == 1, f"{entropy.shape}"
                for i in range(NUM_LAYERS):
                    entropys[i] += entropy[i].item()
                calc = False
            
            x = model.lm_head(model.model.norm(x[:, :, -1, :])) # num_layers - 1 x batch_size x vocab_size
            
            layers_next_token = [0] * (NUM_LAYERS - 1)
            if do_sample:
                logits_probs = torch.softmax(logits, dim=-1)
                x_probs = torch.softmax(x, dim=-1)
                next_token = torch.multinomial(logits_probs, num_samples=1).squeeze(1)
                for i in range(len(x)):
                    layers_next_token[i] = torch.multinomial(x_probs[i], num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)
                for i in range(len(x)):
                    layers_next_token[i] = torch.argmax(x[i], dim=-1)
            
            assert bsz == len(next_token), f"{bsz} != {len(next_token)}"
            for i in range(len(next_token)): # batch_size
                ch = tokenizer.decode(next_token[i].item())
                Results[-1][i].append(ch)
                if count(ch, lang) > 0:
                    zh_table[-1] += 1
                    al_table[-1] += 1
                assert len(layers_next_token) == NUM_LAYERS - 1
                for j in range(len(layers_next_token)):
                    ch_ = tokenizer.decode(layers_next_token[j][i].item())
                    Results[j][i].append(ch_)
                    assert j != NUM_LAYERS - 1
                    if count(ch, lang) > 0:
                        al_table[j] += 1
                        if count(ch_, lang) > 0:
                            zh_table[j] += 1
            
            next_token = next_token * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stop_criteria(input_ids, None)
            
            if not unfinished_sequences.any():
                return Results
            

    # ds = load_dataset(data_path, lang, trust_remote_code=True, split="test")["question"]
    ds = ["Q: What is the capital of China? A:\n"]
    for x in tqdm(ds):
        with torch.no_grad():
            Results = CustomDecode(
                input_ids=tokenizer(x, return_tensors="pt")["input_ids"].to(model.device),
                model=model,
                tokenizer=tokenizer,
                max_length=20,
                do_sample=True
            )

    
    with open(f"./log/{prefix}_{lang}.txt", "w", encoding="utf-8") as f:
        for j in range(len(Results[0])):
            f.write(f"BATCH {j}\n")
            for i in range(NUM_LAYERS):
                f.write(f"Layer {i + 1}: {Results[i][j]}\n")
    
    
    torch.cuda.empty_cache()
    return zh_table[:-1], al_table[:-1], entropys[:-1]


if __name__ == "__main__":
    fig, ax = plt.subplots()
    for i in range(len(mode_path)):
        print(f"Processing {name[i]}")
        x, y, en = main_work(mode_path[i], prefix=name[i])
        print(
            f"For check, {x[-1], y[-1], en[-1]}")
        x_values = np.arange(len(x)) 
        y_values = [x[i] / y[i] for i in range(len(x))]
        max_en = max(en)
        en_values = [en[i] / max_en for i in range(len(en))]
        # smooth en_values
        en_value_smooth = np.cumsum(en_values) / (np.arange(1, len(en_values) + 1))
        ax.plot(x_values, y_values, label=name[i] + f" {lang} ratio", marker='o', linestyle='-')
        ax.plot(x_values, en_values, label=name[i] + " entropy", marker='o', linestyle='-')
    ax.grid()
    ax.set_xlabel("Layer Number")
    ax.set_ylabel(f"{lang} word ratio or entropy")
    ax.legend()
    plt.savefig(f'./pic/Llama_ {lang}.png', dpi=500)
    plt.show()