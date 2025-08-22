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

def get_igrads(llm, q: str, a: str):
    N_STEPS = 256     # How many total integration steps to take
    BLOCK_SIZE = 16   # Batch size for the integration steps

    atok = tokenize(llm, a, add_special_tokens=False)
    alen = len(atok)

    prompt = q + a

    alltok = tokenize(llm, prompt)

    with llm.session(remote=REMOTE) as session:

        igrads = []
        for l in tqdm(range(len(llm.model.layers))):
            l_igrads = 0

            for step_block in range(0, N_STEPS, BLOCK_SIZE):
                with llm.trace(prompt, debug=True) as tracer:
                    tracer.log(f"Layer {l} step {step_block} of {N_STEPS}")
                    orig_output = llm.model.layers[l].output[0].clone().detach() # [seq_len, d_model]

                    # Set the baseline to the mean activation
                    # TODO: mean over the d_model instead of seq_len?
                    baseline = orig_output.mean(dim=1, keepdims=True)  # [seq_len, 1]
                    
                    # Create a bunch of interpolated activations between the original activation and the baseline
                    # This also creates a batch from the single example that we had before this layer.
                    r = torch.arange(start=step_block, end=min(step_block + BLOCK_SIZE, N_STEPS), device=baseline.device, dtype=baseline.dtype) / N_STEPS
                    target = orig_output * r[:, None, None] + baseline * (1-r[:, None, None]) # [interpolated btz, seq_len, d_model]

                    # Overwrite the MLP output with the target
                    llm.model.layers[l].output = target  # [btz, seq_len, d_model]
                    llm.model.layers[l].output.requires_grad_()
                    
                    # Get the probability of the ground truth tokens. The GT token is the input of the
                    # embedding layer.
                    oclasses = F.softmax(llm.output.logits[:, :-1], dim=-1) # logits: [btz, seq_len-1, vocab_size], oclasses: [btz, seq_len-1, vocab_size]
                    tid = llm.model.embed_tokens.input[0][1:] # llm.model.embed_tokens.input: [1, seq_len], tid: [seq_len-1]

                    tid = tid.repeat(oclasses.shape[0], 1) # oclasses: [btz, seq_len-1, vocab_size], tid: [1, seq_len-1]
                    oprobs = oclasses.gather(-1, tid.unsqueeze(-1))  # [btz, seq_len-1, 1]

                    # Sum grad * activation diff for all different steps
                    igrad = (llm.model.layers[l].output.grad * (orig_output - baseline)).detach().cpu().float().sum(0) # sum over batch [btz, seq_len, d_model] -> [seq_len, d_model]
                    l_igrads = l_igrads + igrad

                    # Call backward. Should be done after the gradient hooks are set up, otherwise
                    # the grads will be empty.
                    oprobs[:, -alen:].sum().backward()

            # Save the grads for this layer
            igrads.append((l_igrads.sum(-1) / N_STEPS)) # [seq_len]

        result = torch.stack(igrads, dim=0).save()

    return result, alltok

def plot_igrads(layer_attributions, tokens):
    fig, ax = plt.subplots(figsize=[5, 5 * max(1, layer_attributions.shape[0] / 30)])

    # Remove the BOS token. Make sure that the limits are symmetric because of the colormap.
    r = layer_attributions[:, 1:].abs().max().item()

    tokens = [t if t != "<|begin_of_text|>" else "BOS" for t in tokens]

    im = ax.imshow(layer_attributions[:, :-1].float().cpu().numpy(), cmap="seismic", vmin=-r, vmax=r, interpolation="nearest")
    plt.xticks(range(len(tokens)-1), tokens[:-1], rotation=45, ha='right',rotation_mode="anchor", fontsize=8)
    plt.ylabel("Layer")
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    return fig

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze relative contribution.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--remote', action='store_true', help='Whether to use the NDIF hosted model')
    parser.add_argument('--n_examples', type=int, default=10, help='Number of examples for the future effect tests')
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
    
    igrads, tokens = get_igrads(llm, q, a)
    
    if args.task in ['addition', 'multiplication']:
        output_dir = f"outputs/{args.task}_{args.length}/{model_name}"
    else:
        output_dir = f"outputs/{args.task}/{model_name}"
            
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save({
        "igrads": igrads.detach().cpu(),
    }, f"{output_dir}/igrads_igrads_{N_EXAMPLES}exps.pt")

    # save tokens
    with open(f"{output_dir}/igrads_tokens_{N_EXAMPLES}exps.txt", "w") as f:
        for token in tokens:
            f.write(token + "\n")
    
    # plot the results
    igrads_fig = plot_igrads(igrads, tokens)
    igrads_fig.axes[0].set_title("Integrated Gradients: " + model_name)
    
    plt.savefig(f"{output_dir}/igrads_{N_EXAMPLES}exps.pdf", bbox_inches='tight')
    