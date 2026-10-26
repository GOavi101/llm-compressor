"""
Compare GPTQ and ADMM-Q on WikiText-2 perplexity.

usage: python bench.py <config> [<config> ...]

MODEL_ID, NUM_CALIB, CALIB_LEN, EVAL_LEN, EVAL_WINDOWS, and OUT are read from
the environment. Appends one JSON line per config to OUT.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.transform import SpinQuantModifier

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
NUM_CALIB = int(os.environ.get("NUM_CALIB", 128))
CALIB_LEN = int(os.environ.get("CALIB_LEN", 512))
EVAL_LEN = int(os.environ.get("EVAL_LEN", 1024))
EVAL_WINDOWS = int(os.environ.get("EVAL_WINDOWS", 40))
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
OUT = Path(__file__).parent / os.environ.get("OUT", "results.jsonl")


def w3(sym=True):
    return QuantizationArgs(
        num_bits=3, type="int", symmetric=sym, strategy="group", group_size=128
    )


A8 = QuantizationArgs(
    num_bits=8, type="int", symmetric=True, strategy="token", dynamic=True
)


def scheme(weights, inputs=None):
    return {
        "group_0": QuantizationScheme(
            targets=["Linear"], weights=weights, input_activations=inputs
        )
    }


def gptq(solver, groups=None, scheme_name=None, **kwargs):
    return GPTQModifier(
        targets="Linear",
        ignore=["lm_head"],
        config_groups=groups,
        scheme=scheme_name,
        solver=solver,
        **kwargs,
    )


def smooth():
    return SmoothQuantModifier(smoothing_strength=0.8)


def spin():
    return SpinQuantModifier(
        rotations=["R1", "R2"], transform_block_size=128, transform_type="hadamard"
    )


CONFIGS = {
    "fp16": lambda: None,
    "w3_gptq": lambda: gptq("gptq", scheme(w3())),
    "w3_admm": lambda: gptq("admm", scheme(w3())),
    "w3_admm_nols": lambda: gptq("admm", scheme(w3()), admm_local_search_rounds=0),
    "w3_admm_refresh": lambda: gptq("admm", scheme(w3()), admm_grid_refresh=True),
    "w3a8_sq_gptq": lambda: [smooth(), gptq("gptq", scheme(w3(), A8))],
    "w3a8_sq_admm": lambda: [smooth(), gptq("admm", scheme(w3(), A8))],
    "spin_w3_gptq": lambda: [spin(), gptq("gptq", scheme(w3()))],
    "spin_w3_admm": lambda: [spin(), gptq("admm", scheme(w3()))],
    "nvfp4_gptq": lambda: gptq("gptq", scheme_name="NVFP4A16"),
    "nvfp4_admm": lambda: gptq("admm", scheme_name="NVFP4A16"),
}


def wikitext_ids(tokenizer, split):
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
    )
    return tokenizer(text, return_tensors="pt").input_ids[0]


def calibration_set(tokenizer):
    ids = wikitext_ids(tokenizer, "train")
    gen = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(ids) - CALIB_LEN, (NUM_CALIB,), generator=gen)
    rows = [ids[s : s + CALIB_LEN].tolist() for s in starts]
    return Dataset.from_dict(
        {"input_ids": rows, "attention_mask": [[1] * CALIB_LEN for _ in rows]}
    )


@torch.no_grad()
def perplexity(model, ids):
    nll, count = 0.0, 0
    for index in range(EVAL_WINDOWS):
        window = ids[index * EVAL_LEN : (index + 1) * EVAL_LEN].unsqueeze(0)
        window = window.to(model.device)
        logits = model(window).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(
            logits, window[0, 1:], reduction="sum"
        ).item()
        count += window.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / count)))


def run(name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    recipe = CONFIGS[name]()
    start = time.time()
    if recipe is not None:
        oneshot(
            model=model,
            dataset=calibration_set(tokenizer),
            recipe=recipe,
            num_calibration_samples=NUM_CALIB,
            max_seq_length=CALIB_LEN,
        )
    quant_minutes = (time.time() - start) / 60
    dispatch_model(model)
    ppl = perplexity(model, wikitext_ids(tokenizer, "test"))
    result = {
        "model": MODEL_ID,
        "config": name,
        "wikitext2_ppl": ppl,
        "quant_minutes": quant_minutes,
    }
    print(json.dumps(result), flush=True)
    with OUT.open("a") as file:
        file.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    for config in sys.argv[1:]:
        run(config)
