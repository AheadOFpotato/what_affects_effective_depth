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


def measure_token_skip(llm: LanguageModel, q: str, a: str, prompts: List[str]) -> Tuple[torch.Tensor, List[str]]:
    # Replace each position in the residual stream (each layer + each token) with an uninformative
    # value and measure the effect on the answer.
    # Returns the effect map of shape (layers, tokens) and the the string token IDs.
    atokens = tokenize(llm, a, add_special_tokens=False)
    alen = len(atokens)

    prompt = q + a

    tokens = tokenize(llm, prompt)

    with torch.no_grad():
        with llm.session(remote=REMOTE) as session:
            # Collect uninformative residuals from 10 GSM8K examples
            baseline_residuals = {
                i: 0 for i in range(len(llm.model.layers)+1)
            }
            count = 0

            for i, nprompt in enumerate(prompts):

                with llm.trace(nprompt) as tracer:
                    for l in range(len(llm.model.layers)):
                        if l == 0:
                            # Special handling for the token embedding from before the first layer
                            baseline_residuals[0] += llm.model.layers[0].input[0].detach().sum(dim=0).cpu().float()
                        baseline_residuals[l+1] += llm.model.layers[l].output[0].detach().sum(dim=0).cpu().float()
                    count += llm.model.layers[0].output[0].shape[1]
                    # residual shape: [d_model]

            baseline_residuals = {k: (v / count) for k, v in baseline_residuals.items()}
            # session.log("baseline residuals", baseline_residuals)

            # Measure the output probability distribution without intervention
            with llm.trace(prompt):
                outs = llm.output.logits[:, -(alen+1):-1].detach().softmax(dim=-1)
                tracer.log("outs", outs.shape)
                # outs: [1, 2, 152064] 
                # [btz, answer_len, vocab]

            ls = []
            for l in range(len(llm.model.layers) + 1):
                ts = []
                for t in range(len(tokens)):
                    with llm.trace(prompt):
                        if l == 0:
                            # Intervene on the embeddings
                            layer = llm.model.layers[0]
                            layer.input[0][t, :] = baseline_residuals[l]
                            # layer.input[0]: [seq_len, d_model];
                            # baseline_residuals[l]: [d_model]
                        else:
                            # Intervene on the layer output
                            layer = llm.model.layers[l-1]
                            layer.output[0][t, :] = baseline_residuals[l]

                        # Measure the max effect on the output probability of any answer token
                        ts.append((outs - llm.output.logits[:, -(alen+1):-1].detach().softmax(dim=-1)).norm(dim=-1).max(dim=1).values.cpu())

                # Concatenate the token effects (a row of the plot)
                ls.append(torch.cat(ts, 0))

            # Concatenate layers (columns of the plot)
            result = torch.stack(ls, dim=0).save()

    return result, tokens

def plot_logit_effect(ls: torch.Tensor, tokens: List[str]) -> plt.Figure:
    ls = ls[:, :-1]
    tokens = tokens[:-1]
    tokens = [t if t != "<|begin_of_text|>" else "BOS" for t in tokens]
    fig, ax = plt.subplots(figsize=(5,5 * max(1, ls.shape[0] / 30)))
    im = ax.imshow(ls.float().cpu().numpy())
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right',rotation_mode="anchor", fontsize=8)
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label='Probability Difference Norm')
    return fig

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze relative contribution.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--remote', action='store_true', help='Whether to use the NDIF hosted model')
    parser.add_argument('--n_examples', type=int, default=10, help='Number of examples to calculate baseline residuals')
    parser.add_argument('--task', type=str, default='default', choices=['default', 'addition', 'multiplication'], help='Dataset to use for testing')
    parser.add_argument('--length', type=int, default=6, help='Length of the task')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    REMOTE = args.remote
    model_name = args.model_name
    N_EXAMPLES = args.n_examples
    
    print(f"Using model: {model_name}, remote: {REMOTE}, number of examples: {N_EXAMPLES}")
    
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
    
    prompts = []
    for i, prompt in enumerate(GSM8K()):
        if i >= N_EXAMPLES:
            break
        prompts.append(prompt)
        
    if args.task in ['addition', 'multiplication']:
        output_dir = f"outputs/{args.task}_{args.length}/{model_name}"
    else:
        output_dir = f"outputs/{args.task}/{model_name}"
            
    os.makedirs(output_dir, exist_ok=True)
    prompts = []

    for i, prompt in enumerate(GSM8K()):
        if i >= N_EXAMPLES:
            break
        prompts.append(prompt)
    
    if args.task == 'addition':
        prompt = Addition(length=args.length).generate_example()
        q = prompt.split('=')[0] + '= '
        a = prompt.split('=')[1].strip()
    elif args.task == 'multiplication':
        prompt = Multiplication(length=args.length).generate_example()
        q = prompt.split('=')[0] + '= '
        a = prompt.split('=')[1].strip()
    else:
        raise ValueError(f"Unknown task: {args.task}")

    erasure_effect, tokens = measure_token_skip(llm, q, a, prompts)
    torch.save({
        "erasure_effect": erasure_effect.detach().cpu(),
    }, f"{output_dir}/logit_erasure_{N_EXAMPLES}exps.pt")
    
    # save the tokens
    with open(f"{output_dir}/logit_erasure_tokens_{N_EXAMPLES}exps.txt", "w") as f:
        for token in tokens:
            f.write(f"{token}\n")
    
    # plot the results
    fig = plot_logit_effect(erasure_effect, tokens)
    fig.axes[0].set_title("Logit Erasure Effect: " + model_name)
    fig.savefig(f"{output_dir}/logit_erasure_{N_EXAMPLES}exps.pdf", bbox_inches='tight')