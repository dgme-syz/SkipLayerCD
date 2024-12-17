from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
from typing import List, Literal
import random
import numpy as np

MODEL_NAME = ["transformer", "model"]
LAYER_NAME = ["h", "layer"]
NORMF_NAME = ["norm", "ln_f"]

def custom_hook(Module: nn.Module, input, output: tuple[torch.Tensor, ...]):
    # Because we stack data x 2, so we just update amateur output
    # the second part is the amateur output
    output_logits = output[0]
    batch_size = output_logits.shape[0] // 2
    output_logits[batch_size:, :, :] = input[0][batch_size:, :, :]
    return output_logits, *output[1:]



def get_net_from_model(model: PreTrainedModel) -> nn.Module:
    for name in MODEL_NAME: 
        if hasattr(model, name): 
            return getattr(model, name) 
    else: raise NotImplementedError(f"model has no attribute {MODEL_NAME}")



def set_skip_layers(model: PreTrainedModel, skip_layers: list[int]):
    # find net, decoderlayer, and register hook
    net = get_net_from_model(model)
    
    for layer_name in LAYER_NAME:
        if hasattr(net, layer_name):
            layers: nn.ModuleList = getattr(net, layer_name)
            for x in skip_layers:
                if x < len(layers):
                    layers[x].register_forward_hook(custom_hook)
            break
    else: raise NotImplementedError(f"model has no attribute {LAYER_NAME}")
    
    
    
@torch.inference_mode()
def choose_layers(config: PretrainedConfig, strategy: Literal["sl-h", "sl-d"] = "sl-h", **kwargs) -> List[int]:
    num_layers = config.num_hidden_layers
    if num_layers < 8: 
        raise ValueError("Model size is too small, it should have more than 8 layers")

    skip_len = int(round(num_layers / 8))

    if strategy == "sl-h":
        u = random.randrange(4, num_layers // 2)
        return list(range(u, num_layers))

    elif strategy == "sl-d":
        params_list = ["model", "tokenizer", "prefix"]
        model, tokenizer, prefix = [kwargs.get(x, None) for x in params_list]
        if model is None or tokenizer is None or prefix is None:
            raise ValueError(f"Missing some parameters of: {params_list}")

        outputs = model(
            **tokenizer(prefix, return_tensors="pt").to(model.device),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        Y = []

        net = get_net_from_model(model)
        norm = None
        for name in NORMF_NAME:
            if hasattr(net, name):
                norm = getattr(net, name)
                break
        else: 
            raise NotImplementedError(f"Model has no attribute {NORMF_NAME}")

        for x in hidden_states[:-1]:
            x = x[:, :-1, :]
            logits = model.lm_head(norm(x)).cpu().float()
            logprobs = torch.log_softmax(logits, dim=-1)
            entropy = -(logprobs * torch.exp(logprobs)).sum(dim=-1).mean()
            Y.append(entropy.item())

        def select_layer_based_on_entropy(entropy, delta: float = 0.1, k: int = 6) -> List[int]:
            num_layers = len(entropy)
            en = np.cumsum(entropy) / np.arange(1, num_layers + 1)
            en_diff = [en[i] - en[i + 1] for i in range(num_layers - 1)]

            for l in range(max(0, k), num_layers - skip_len):
                r = l + skip_len - 1
                if en_diff[l] > delta and entropy[r] >= max(entropy[r:]):
                    return list(range(l, r + 1))
            else:
                return choose_layers(config, strategy="sl-h")

        return select_layer_based_on_entropy(Y)
    
