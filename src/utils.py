# -*- coding:utf-8 -*-
# # # @ Start Time: 04/15/2025
# # # @ Last Modified: 04/15/2025

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    is_wandb_available,
    Trainer,
    pipeline,
)
from peft import PeftModel, PeftConfig, get_peft_model
from typing import Optional, Dict, Any, Generator
from copy import deepcopy
import contextlib
import functools
import time
import torch.nn.functional as F
from peft import AdaLoraConfig, TaskType, get_peft_model, LoraConfig
import torch.optim as optim
import os

if is_wandb_available():
    import wandb


LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]

# @ Directly copy from `trl` repo.
@contextlib.contextmanager
def profiling_context(trainer: Trainer, name: str) -> Generator[None, None, None]:
    """
    A context manager function for profiling a block of code. Results are logged to Weights & Biases if enabled.

    Args:
        trainer (`~transformers.Trainer`):
            Trainer object.
        name (`str`):
            Name of the block to be profiled. Used as a key in the logged dictionary.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_context

    class MyTrainer(Trainer):
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            with profiling_context(self, "matrix_multiplication"):
                # Code to profile: simulate a computationally expensive operation
                result = A @ B  # Matrix multiplication
    ```
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    if (
        "wandb" in trainer.args.report_to
        and wandb.run is not None
        and trainer.accelerator.is_main_process
    ):
        wandb.log(
            {f"profiling/Time taken: {trainer.__class__.__name__}.{name}": duration}
        )


# @ Directly copy from `trl` repo.
def profiling_decorator(func: callable) -> callable:
    """
    Decorator to profile a function and log execution time using [`extras.profiling.profiling_context`].

    Args:
        func (`callable`):
            Function to be profiled.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_decorator

    class MyTrainer(Trainer):
        @profiling_decorator
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            # Code to profile: simulate a computationally expensive operation
            result = A @ B
    ```
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
    ).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens


# @ modified from `trl` repo: trl/models/modeling_base.py/create_reference_model
def create_reference_model(
    model: AutoModelForCausalLM,
    num_shared_layers: Optional[int] = None,
    pattern: Optional[str] = None,
):
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model (`PreTrainedModelWrapper`): The model to be copied.
        num_shared_layers (`int`, *optional*): The number of initial layers that are shared between both models and kept frozen.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns:
        `AutoModelForCausalLM`: A reference model with the same architecture as the original model, but with all parameters frozen.
    """

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any(pattern_candidate in name for name in parameter_names):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, _param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)

    # create reference of the original parameter if they are shared
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False

        _ref_param = ref_model.get_parameter(param_name)

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    if pattern is not None and len(unshared_param_list) == 0:
        # logging.warning(
        print(
            "Warning: ",
            "Pattern passed or found, but no layers matched in the model. Check for a typo.",
        )

    return ref_model.eval()


def load_target_set(fn):
    """
    Load a target set from a file.

    Args:
        fn (`str`): The file name of the target set.

    Returns:
        `List[str]`: The loaded target set.
    """
    import pandas as pd
    data = pd.read_csv(fn)

    goals = data["goal"].tolist()

    return goals


def create_output_filename_and_path(
    save_dir: str,
    inupt_filename: str,  # name only, no need to add path
    attacker_name: str = "DyTA",
    local_model_name: str = "Llama3",
    ref_model_name: Optional[str] = None,
    num_iters: int = 10,
    num_inner_iters: int = 200,
    reference_model_infer_temperature: float = 1.0,
    num_ref_infer_samples: int = 10,
    forward_response_length: int = 20,
    version: Optional[str] = None,
    start_index = 0,
    end_index = 100,
):
    """
    Create the output filename based on the parameters.
    """
    inupt_filename = inupt_filename.split("/")[-1].split(".")[0]
    ref_model_name = ref_model_name if ref_model_name is not None else local_model_name
    output_filename = f"{attacker_name}_{local_model_name}_{ref_model_name}_res_{inupt_filename}_T{reference_model_infer_temperature}_S{num_ref_infer_samples}_FRL{forward_response_length}_NOI{num_iters}_NII{num_inner_iters}_{start_index}_{end_index}"
    if version is not None:
        output_filename += f"_{version}"
    output_filename += ".jsonl"
    output_filepath = os.path.join(save_dir, output_filename)
    return output_filepath


def get_model_inference_pipeline(model_id, device = "cuda:0"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )
    tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4096, 
        device = device,
    )
    return pipe
