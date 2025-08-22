import os
import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG
import torch
import torch.nn.functional as F
import random
from typing import Optional, List, Dict, Tuple
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import argparse

from dataset import *
from utils import *


def merge_io(intervened: torch.Tensor, orig: torch.Tensor, t: Optional[int] = None, no_skip_front: int = 1) -> torch.Tensor:
    # Merge intervened and original inputs. If t is not None, keep the intervened input until t, otherwise keep it everywhere.
    # It does not intervene on the first no_skip_front tokens.
    outs = [orig[:, :no_skip_front]]
    if t is not None:
        outs.append(intervened[:, no_skip_front:t].to(orig.device))
        outs.append(orig[:, t:])
    else:
        outs.append(intervened[:, no_skip_front:].to(orig.device))

    return torch.cat(outs, dim=1)

def get_future(data: torch.Tensor, t: Optional[int]) -> torch.Tensor:
    # Get future tokens from position t onwards. If t is None, return all tokens.
    if t is not None:
        return data[:, t:]
    else:
        return data


def test_effect(llm: LanguageModel, prompt: str, positions: List[Optional[int]], no_skip_front: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    # Test effect of skipping a layer on all future layers and the output probabilities.
    # If multiple positions are provided, the maximum is taken over all positions.
    # If position is None, it measures the effect on all tokens, not just the future.
    # Note: Detach must be explicitly called on each saved tensor, ortherwise the graph will be kept and it will run out of memory.
    # This is despite being wrapped in torch.no_grad().


    # The idea is to run all interventions in a single session to avoid downloading intermediate activations,
    # which can be very large. The output of matmul in bfloat16 is sensitive to the kernel used by cuBLAS,
    # which changes with the batch size. So run everything in a single batch.

    with torch.no_grad():
        with llm.session(remote=REMOTE) as session:

            dall = torch.zeros(1)
            dall_out = torch.zeros(1)

            residual_log = []

            # Run the model to get the original residuals and output probabilities.
            with llm.trace(prompt) as tracer:
                for i, layer in enumerate(llm.model.layers):
                    if i == 0:
                        residual_log.clear()
                    residual_log.append(layer.output[0].detach().cpu().float() - layer.input[0].detach().cpu().float())
                    # here layer.input: [1, seq_len, d_model]; layer.output: [1, seq_len, d_model]

                residual_log = torch.stack(residual_log, dim=0)
                # residual_log: [layer, seq_len, d_model]
                outputs = llm.output.logits.detach().float().softmax(dim=-1).cpu()

            # Do intervention on each position.
            for t in positions:
                diffs = []
                out_diffs = []

                # Do intervention on each layer.
                for lskip in range(len(llm.model.layers)):
                    with llm.trace(prompt) as tracer:
                        new_logs = []

                        # Log all the layer outputs.
                        for i, layer in enumerate(llm.model.layers):
                            layer_inputs = layer.input[0]
                            # TODO: [bug] layer 0 input [1, seq_len, d_model]; layer 1 input [seq_len, d_model]

                            # skip the layer
                            if i == lskip:
                                # tracer.log(f"Skipping layer {i}")
                                layer_output = layer.output[0]
                                layer.output = merge_io(layer_inputs, layer_output, t, no_skip_front).unsqueeze(0)
                                # layer.output: [1, seq_len, d_model]

                            # tracer.log(f"layer_{i}.output", layer.output.shape)
                            
                            new_logs.append((layer.output[0].detach().cpu().float() - layer_inputs.detach().cpu().float()))
                            
                            # tracer.log(f"layer_{i}_output", layer.output.shape)

                        new_logs = torch.stack(new_logs, dim=0).float()

                        # Measure the relative difference of the residuals on the future tokens
                        relative_diffs = (get_future(residual_log, t) - get_future(new_logs, t)).norm(dim=-1) / get_future(residual_log, t).norm(dim=-1).clamp(min=1e-6)

                        # Take the max realtive difference over the sequence lenght
                        diffs.append(relative_diffs.max(dim=-1).values)

                        # Measure the max relative difference of the output probabilities on the future tokens
                        # outputs: [1, seq_len, vocab_size]
                        # get_future(outputs, t)).norm(dim=-1): [1, seq_len]
                        out_diffs.append((get_future(llm.output.logits.detach(), t).float().softmax(dim=-1).cpu() - get_future(outputs, t)).norm(dim=-1).max(dim=-1).values)

                # Concatenate effects over all layers.
                dall = torch.max(dall, torch.stack(diffs, dim=0))
                dall_out = torch.max(dall_out, torch.stack(out_diffs, dim=0))

            dall = dall.save()
            dall_out = dall_out.save()
    return dall, dall_out

def plot_layer_diffs(dall: torch.Tensor) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10,3))
    im = ax.imshow(dall.float().cpu().numpy(), vmin=0, vmax=1, interpolation="nearest")
    plt.ylabel("Layer skipped")
    plt.xlabel("Effect @ layer")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label='Relative change')
    return fig


def plot_logit_diffs(dall: torch.Tensor) -> plt.Figure:
    fig = plt.figure(figsize=(6,3))
    dall = dall.squeeze()
    plt.bar(list(range(dall.shape[0])), dall)
    plt.xlim(-1, dall.shape[0])
    plt.xlabel("Layer")
    plt.ylabel("Output change norm")
    return fig

def plot_effects(llm: LanguageModel, n_examples: int, test_fn, model_name, dataset):
    random.seed(123)

    max_future_layer = torch.zeros([1])
    max_future_out = torch.zeros([1])
    for idx, prompt in tqdm(enumerate(DATASET), desc=f"Testing {n_examples} samples"):
        diff_now, diff_out = test_fn(llm, prompt)
        max_future_layer = torch.max(max_future_layer, diff_now)
        max_future_out = torch.max(max_future_out, diff_out)

        if idx == n_examples - 1:
            break
        
    torch.save({
        "max_future_layer": max_future_layer.detach().cpu(),
        "max_future_out": max_future_out.detach().cpu()
    }, f"{output_dir}/layer_effects_{n_examples}exps.pt")
    
    return plot_layer_diffs(max_future_layer), plot_logit_diffs(max_future_out)
    
def test_future_max_effect(llm: LanguageModel, prompt: str, N_CHUNKS: int = 4):
    # Sample N_CHUNKS positions to intervene on and calculate the maximum effect on this single prompt.
    tokens = tokenize(llm, prompt)

    positions = list(range(8, len(tokens)-4, 8))
    random.shuffle(positions)
    positions = positions[:N_CHUNKS]

    return test_effect(llm, prompt, positions)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze relative contribution.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--remote', action='store_true', help='Whether to use the NDIF hosted model')
    parser.add_argument('--n_examples', type=int, default=10, help='Number of examples for the future effect tests')
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'hellaswag', 'aime24', 'addition', 'multiplication'], help='Dataset to use for testing')
    parser.add_argument("--length", type=int, default=6, help="Length of the addition/multiplication task")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    REMOTE = args.remote
    model_name = args.model_name
    N_EXAMPLES = args.n_examples
    
    if args.dataset == 'gsm8k':
        DATASET = GSM8K()
    elif args.dataset == 'hellaswag':
        DATASET = hellaswag()
    elif args.dataset == 'aime24':
        DATASET = AIME24()
    elif args.dataset == 'addition':
        DATASET = Addition(length=args.length)
    elif args.dataset == 'multiplication':
        DATASET = Multiplication(length=args.length)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Using model: {model_name}, remote: {REMOTE}, number of examples: {N_EXAMPLES}, dataset: {args.dataset}")
    
    model_path = get_model_path(model_name)
    if model_path is None:
        raise ValueError(f"Model name '{args.model_name}' not found in the model path mapping.")
    
    if not REMOTE:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        llm = LanguageModel(model_path, device_map="auto", quantization_config=bnb_config, torch_dtype=torch.bfloat16)
    else:
        llm = LanguageModel(model_path)

    llm.eval()

    if args.dataset in ['addition', 'multiplication']:
        output_dir = f"outputs/{args.dataset}_{args.length}/{model_name}"
    else:
        output_dir = f"outputs/{args.dataset}/{model_name}"
        
    os.makedirs(output_dir, exist_ok=True)

    layer_diff_fig, logit_diff_fig = plot_effects(llm, N_EXAMPLES, test_future_max_effect, model_name, dataset=args.dataset)
    layer_diff_fig.axes[0].set_title("Layer Effects:" + model_name)
    logit_diff_fig.axes[0].set_title("Logit Effects of each layer:" + model_name)
    
    layer_diff_fig.savefig(f"{output_dir}/layer_effects_layer_diff_{N_EXAMPLES}exps.pdf", bbox_inches='tight')
    logit_diff_fig.savefig(f"{output_dir}/layer_effects_logit_diff_{N_EXAMPLES}exps.pdf", bbox_inches='tight')
