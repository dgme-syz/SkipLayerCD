from transformers import PreTrainedModel, PretrainedConfig, LlamaForCausalLM
from transformers.generation import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.streamers import BaseStreamer 
from transformers.generation.utils import GenerateNonBeamOutput, _relative_top_filter, GenerateDecoderOnlyOutput, GenerateOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_torchdynamo_compiling, logging
import torch.distributed as dist
import torch
import torch.nn as nn
from typing import List, Literal, Union, Optional, Callable
import random
import numpy as np
from torch.utils.hooks import RemovableHandle
import inspect
import warnings
from .plot import plot_with_debug

MODEL_NAME = ["transformer", "model"]
LAYER_NAME = ["h", "layers"]
NORMF_NAME = ["norm", "ln_f"]

logger = logging.get_logger(__name__)

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



def set_skip_layers(model: PreTrainedModel, skip_layers: list[int]) -> Union[None, List[RemovableHandle]]:
    # find net, decoderlayer, and register hook
    net = get_net_from_model(model)
    hooks = []
    for layer_name in LAYER_NAME:
        if hasattr(net, layer_name):
            layers: nn.ModuleList = getattr(net, layer_name)
            for x in skip_layers:
                if x < len(layers):
                    hooks.append(
                        layers[x].register_forward_hook(custom_hook))
            return hooks
    else: raise NotImplementedError(f"model has no attribute {LAYER_NAME}")
    
    
    
@torch.inference_mode()
def choose_layers(config: PretrainedConfig, strategy: Literal["sl-h", "sl-d"] = "sl-h", **kwargs) -> List[int]:
    num_layers = config.num_hidden_layers
    if num_layers < 8: 
        raise ValueError("Model size is too small, it should have more than 8 layers")

    skip_len = int(round(num_layers / 8))

    if strategy == "sl-h":
        u = random.randrange(4, num_layers // 2)
        return list(range(u, u + skip_len))

    elif strategy == "sl-d":
        params_list = ["model", "prefix_logits"]
        model, prefix_logits = [kwargs.get(x, None) for x in params_list]
        if model is None or prefix_logits is None:
            raise ValueError(f"Missing some parameters of: {params_list}")

        outputs = model(
            prefix_logits.to(model.device),
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

        for x in hidden_states[:-1]: # the last hidden state has been activated
            x = x[:, :-1, :]
            logits = model.lm_head(norm(x)).cpu().float()
            logprobs = torch.log_softmax(logits, dim=-1)
            entropy = -(logprobs * torch.exp(logprobs)).sum(dim=-1).mean()
            Y.append(entropy.item())

        # [TODO] Find a better algorithm to select layers
        def select_layer_based_on_entropy(entropy, delta: float = 0.1, k: int = 6) -> List[int]:
            num_layers = len(entropy)

                
            mx_idx = entropy.index(max(entropy))
            l = min(max(1, mx_idx), num_layers - skip_len - 1)
            best_l = None
            best_en_diff = float("inf")
            while l + skip_len <= num_layers - 1:
                # [l, l + skip_len)
                old_en = entropy[l - 1]
                min_en = min(entropy[l:l + skip_len])
                if best_en_diff > old_en - min_en:
                    best_en_diff = old_en - min_en
                    best_l = l - 1
                l += 1
                
            if kwargs.get("debug", False):
                print(
                    f"[DEBUG] Num_Layer\n{num_layers}\n"
                    f"[DEBUG] Entropy\n{entropy}\n"
                    f"[DEBUG] Best layer: {best_l}, entropy diff: {best_en_diff}, max_idx: {mx_idx}"
                )
                plot_with_debug([entropy], ["entropy"])
            
            if best_l and best_l + skip_len <= num_layers - 1:
                return list(range(best_l, best_l + skip_len))
            else:
                warnings.warn("sl-d strategy failed, call sl-h strategy")
                return choose_layers(config, strategy="sl-h")

        return select_layer_based_on_entropy(Y)

def _sl_decoding(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    sl_layers: List[int],
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: "BaseStreamer",
    prefix_logits: torch.Tensor=None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:    
    r"""
    Parameters:
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
    """
    # https://arxiv.org/abs/2407.10795
    if model.config.is_encoder_decoder:
        raise ValueError("Model is not decoder only")

    if prefix_logits is None:
        prefix_logits = input_ids
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    
    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    
    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
    
    this_peer_finished = False

    # prepare layers for sl decoding
    final_layer = model.config.get_text_config().num_hidden_layers
    
    if isinstance(sl_layers, list) and min(sl_layers) >= 0 and max(sl_layers) < final_layer:
        skip_layers = sl_layers
    elif isinstance(sl_layers, str) and sl_layers=="sl-h":
        skip_layers = choose_layers(model.config, strategy="sl-h")
    elif isinstance(sl_layers, str) and sl_layers=="sl-d":
        skip_layers = choose_layers(
            model.config, strategy="sl-d", model=model, prefix_logits=prefix_logits)
    else: raise ValueError("Invalid skip layers")
    hooks = set_skip_layers(model, skip_layers) # After generate, remove hooks
    
    
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # we should copy input_ids 2 times in batch_size dim
        input_ids = torch.cat([input_ids, input_ids], dim=0)
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        input_ids = input_ids[:batch_size]
        
        # extract amateur logits and expert logits
        final_layer_next_token_logits = outputs.logits[:batch_size, -1, :].detach().clone().float()
        expert_logits = outputs.logits[:batch_size, -1, :].float()
        amateu_logits = outputs.logits[batch_size:, -1, :].float()
        
        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue
        
        final_logits, base_logits = _relative_top_filter(expert_logits, amateu_logits)
        next_token_logits = final_logits - base_logits
        next_token_scores = logits_processor(input_ids, next_token_logits)
        
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (final_layer_next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if do_sample:  # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:  # argmax
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        # stop when each sentence is finished
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    # remove hooks
    for hook_ in hooks: hook_.remove()
    
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    else:
        return input_ids
    
@torch.no_grad()
def sl_generate(
    model: PreTrainedModel,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    sl_layers: Union[str, List[int]] = "sl-h",
    prefix_logits: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    model._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **kwargs)
    model._validate_model_kwargs(model_kwargs.copy())
    model._validate_assistant(assistant_model)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not model.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = model.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if model._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
        model_kwargs["num_logits_to_keep"] = 1

    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
    cache_name = "past_key_values" if "mamba" not in model.__class__.__name__.lower() else "cache_params"
    user_defined_cache = model_kwargs.get(cache_name)
    max_cache_length = generation_config.max_length
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not model.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    model._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )
    
    result = _sl_decoding(
        model,
        input_ids,
        sl_layers,
        prepared_logits_processor,
        prepared_stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        prefix_logits=prefix_logits,
        **model_kwargs,
    )
    
    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is not False  # Should check for `True` after v4.47
        and not is_torchdynamo_compiling()
        and hasattr(result, "past_key_values")
        and hasattr(result.past_key_values, "to_legacy_cache")
        and result.past_key_values.to_legacy_cache is not None
    ):
        # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
        should_convert_cache = generation_config.return_legacy_cache
        is_user_defined_cache = user_defined_cache is not None
        is_default_cache_type = (
            type(result.past_key_values) == DynamicCache  # noqa E721
            or (
                isinstance(result.past_key_values, EncoderDecoderCache)
                and type(result.past_key_values.model_attention_cache) == DynamicCache  # noqa E721
                and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
            )
        )
        if not is_user_defined_cache and is_default_cache_type:
            logger.warning_once(
                "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                "keep returning the legacy format, please set `return_legacy_cache=True`."
            )
            should_convert_cache = True
        if should_convert_cache:
            result.past_key_values = result.past_key_values.to_legacy_cache()
    return result