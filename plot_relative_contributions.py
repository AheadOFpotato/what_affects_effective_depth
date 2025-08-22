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

def tokenize(llm: LanguageModel, prompt: str, add_special_tokens: bool = True) -> List[str]:
    # Tokenize a prompt and return the tokens as a list of strings.
    tokens = llm.tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    token_str = [s.replace("Ġ","_") for s in llm.tokenizer.convert_ids_to_tokens(tokens)]
    return token_str

def analyze_norms(llm, prompts):
    with torch.no_grad():
        with llm.session(remote=REMOTE) as session:
            att_cos_all = 0
            mlp_cos_all = 0
            layer_cos_all = 0

            mean_relative_contribution_att = 0
            mean_relative_contribution_mlp = 0
            mean_relative_contribution_layer = 0

            cnt = 0

            for i, prompt in tqdm(enumerate(prompts), desc="Analyzing norms"):
                with llm.trace(prompt):
                    att_cos = []
                    mlp_cos = []
                    layer_cos = []
                    relative_contribution_att = []
                    relative_contribution_mlp = []
                    relative_contribution_layer = []

                    for i, layer in enumerate(llm.model.layers):
                        # Relative contribution of the attention to the residual stream.

                        layer_inputs = layer.inputs[0][0]
                        self_attn_output = layer.self_attn.output[0]

                        relative_contribution_att.append(
                            (self_attn_output.detach().norm(dim=-1).float() / layer_inputs.detach().norm(dim=-1).float().clamp(min=1e-6)).sum(1).cpu()
                        )

                        # Relative contribution of the MLP to the residual stream. The corresponding
                        # accumulation point is after the self-attention.
                        mlp_input = (self_attn_output + layer_inputs).detach()
                        mlp_output = layer.mlp.output
                        relative_contribution_mlp.append(
                            (mlp_output.detach().norm(dim=-1).float() / mlp_input.norm(dim=-1).clamp(min=1e-6).float()).sum(1).cpu()
                        )

                        # Relative contribution of the layer to the residual stream.
                        layer_output = layer.output[0]
                        layer_diff = (layer_output - layer_inputs).detach()
                        relative_contribution_layer.append(
                            (layer_diff.norm(dim=-1).float() / layer_inputs.detach().norm(dim=-1).float().clamp(min=1e-6)).sum(1).cpu()
                        )

                        # Cosine similarities between the same points as the relative contributions above.
                        att_cos.append(F.cosine_similarity(self_attn_output.detach(), layer_inputs.detach(), dim=-1).sum(1).cpu().float())
                        mlp_cos.append(F.cosine_similarity(mlp_output.detach(), mlp_input, dim=-1).sum(1).cpu().float())
                        layer_cos.append(F.cosine_similarity(layer_diff, layer_inputs.detach(), dim=-1).sum(1).cpu().float())

                        if i == 0:
                            cnt += layer_output.shape[1]

                    mean_relative_contribution_att += torch.cat(relative_contribution_att, dim=0)
                    mean_relative_contribution_mlp += torch.cat(relative_contribution_mlp, dim=0)
                    mean_relative_contribution_layer += torch.cat(relative_contribution_layer, dim=0)

                    att_cos_all += torch.cat(att_cos, dim=0)
                    mlp_cos_all += torch.cat(mlp_cos, dim=0)
                    layer_cos_all += torch.cat(layer_cos, dim=0)

            att_cos_all = (att_cos_all / cnt).save()
            mlp_cos_all = (mlp_cos_all / cnt).save()
            layer_cos_all = (layer_cos_all / cnt).save()

            mean_relative_contribution_att = (mean_relative_contribution_att / cnt).save()
            mean_relative_contribution_mlp = (mean_relative_contribution_mlp / cnt).save()
            mean_relative_contribution_layer = (mean_relative_contribution_layer / cnt).save()

    return (mean_relative_contribution_att, mean_relative_contribution_mlp, mean_relative_contribution_layer,
            att_cos_all, mlp_cos_all, layer_cos_all)

def sort_zorder(bars):
    for group in zip(*[container for container in bars]):
        # 'group' is a tuple of bar objects at one x position from different groups.
        z = len(group)
        # Sort the bars by height (lowest first).
        for bar in sorted(group, key=lambda b: abs(b.get_height())):
            bar.set_zorder(z)
            z -= 1
            
def plot_residual_stats(llm, att, mlp, layer):
    plt.figure(figsize=(6,3))
    bars = []
    bars.append(plt.bar([x for x in range(len(llm.model.layers))], att.float().cpu().numpy(), label="Attention", width=1.1))
    bars.append(plt.bar([x for x in range(len(llm.model.layers))], mlp.float().cpu().numpy(), label="MLP", width=1.1))
    bars.append(plt.bar([x for x in range(len(llm.model.layers))], layer.float().cpu().numpy(), label="Attention + MLP", width=1.1))
    plt.legend()
    sort_zorder(bars)
    plt.xlim(-0.5, len(llm.model.layers)-0.5)
    plt.xlabel("Layer index ($l$)")

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
    
    prompts = []
    for i, prompt in enumerate(DATASET):
        if i >= N_EXAMPLES:
            break
        prompts.append(prompt)

    if args.dataset in ['addition', 'multiplication']:
        output_dir = f"outputs/{args.dataset}_{args.length}/{model_name}"
    else:
        output_dir = f"outputs/{args.dataset}/{model_name}"
        
    os.makedirs(output_dir, exist_ok=True)
    

    rc_att, rc_mlp, rc_layer, att_cos, mlp_cos, layer_cos = analyze_norms(llm, prompts)
    torch.save({
        "rc_att": rc_att,
        "rc_mlp": rc_mlp,
        "rc_layer": rc_layer,
        "att_cos": att_cos,
        "mlp_cos": mlp_cos,
        "layer_cos": layer_cos
    }, f"{output_dir}/relative_contribution.pt")
    
    plot_residual_stats(llm, rc_att, rc_mlp, rc_layer)
    plt.ylim(0, 1.5)
    plt.ylabel("Mean Relative Contribution")
    plt.title("Relative Contribution of Attention and MLP to the Residual Stream: " + model_name)
    plt.savefig(f"{output_dir}/relative_contribution_norm_contribution.pdf", bbox_inches='tight')

    plot_residual_stats(llm, att_cos, mlp_cos, layer_cos)
    plt.ylabel("Cosine similarity")
    plt.title("Cosine Similarity of Attention and MLP to the Residual Stream: " + model_name)
    plt.savefig(f"{output_dir}/relative_contribution_cosine_similarity.pdf", bbox_inches='tight')