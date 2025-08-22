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

def count_metric(lout: torch.Tensor, llayer: torch.Tensor, k_list: list = [1, 3, 5, 10]) -> float:
    '''
    calculate Mean Reciprocal Rank (MRR) and Hits@k for each layer
    '''
    # lout: [btz, seq_len, vocab_size]
    # llayer: [btz, seq_len, vocab_size]
    metrics = {}
    
    # batch_size is always 1 in this case
    target = lout[0].topk(1, dim=-1).indices
    
    rank = (llayer[0].argsort(dim=-1, descending=True) == target).nonzero()[:, 1] # seq_len
    rank = rank + 1
    mrr = (1 / (rank)).sum()
    metrics["mrr"] = mrr
    
    for k in k_list:
        hits_at_k = (rank <= k).float().sum()
        metrics[f"hits@{k}"] = hits_at_k

    return metrics

def run_logitlens(llm, prompts, K=5):

    with torch.no_grad():
        with llm.session(remote=REMOTE) as session:

            res_kl_divs = 0
            res_overlaps = 0
            # res_mrr = 0
            # res_hits1 = 0
            # res_hits3 = 0
            # res_hits5 = 0
            # res_hits10 = 0

            cnt = 0

            # Iterate over all prompts.
            for prompt in tqdm(prompts, desc="Running LogitLens of {} prompts".format(len(prompts))):
                kl_divs = []
                overlaps = []
                # mrr = []
                # hits1 = []
                # hits3 = []
                # hits5 = []
                # hits10 = []
                
                layer_logs = []

                # Iterate over all layers first and get the final logits.
                with llm.trace(prompt):
                    for l in range(len(llm.model.layers)):
                        # Run the LM head and final layernorm on each layer's output
                        tap = llm.model.layers[l].inputs[0][0]
                        layer_logs.append(llm.lm_head(llm.model.norm(tap)).detach().float())
                    out_logits = llm.output.logits
                # nnsight.log("out_logits", out_logits.shape)
                # Compute the final logprobs and top-K predictions
                lout = out_logits.float().log_softmax(-1)
                otopl = lout.topk(K, dim=-1).indices

                # 1 for each token in the top-K, 0 for others.
                real_oh = F.one_hot(otopl, llm.model.embed_tokens.weight.shape[0]).sum(-2)

                for l in range(len(llm.model.layers)):
                    # Compute the KL divergence between the final output and the logitlens outputs.
                    llayer = layer_logs[l].log_softmax(-1)

                    kl_divs.append((llayer.exp() * (llayer - lout)).sum(-1).sum().detach())

                    # Also compute the top-K predictions for each layer.
                    itopl = llayer.topk(K, dim=-1).indices

                    # Compute the top-K mask for the logitlens predictions
                    logitlens_oh = F.one_hot(itopl.to(real_oh.device), llm.model.embed_tokens.weight.shape[0]).sum(-2)

                    # Compute overlap
                    overlaps.append((logitlens_oh.unsqueeze(-2).float() @ real_oh.unsqueeze(-1).float() / K).sum())
                    
                    # metrics = count_metric(lout, llayer, k_list=[1, 3, 5, 10])
                    # mrr.append(metrics["mrr"])
                    # hits1.append(metrics["hits@1"])
                    # hits3.append(metrics["hits@3"])
                    # hits5.append(metrics["hits@5"])
                    # hits10.append(metrics["hits@10"])

                res_kl_divs = res_kl_divs + torch.stack(kl_divs, dim=0)
                res_overlaps = res_overlaps + torch.stack(overlaps, dim=0)
                # res_mrr = res_mrr + torch.stack(mrr, dim=0)
                # res_hits1 = res_hits1 + torch.stack(hits1, dim=0)
                # res_hits3 = res_hits3 + torch.stack(hits3, dim=0)
                # res_hits5 = res_hits5 + torch.stack(hits5, dim=0)
                # res_hits10 = res_hits10 + torch.stack(hits10, dim=0)
                
                # nnsight.log("kl_divs", res_kl_divs)
                # nnsight.log("overlaps", res_overlaps)
                # nnsight.log("mrr", res_mrr)
                # nnsight.log("hits@1", res_hits1)

                cnt += out_logits.shape[1]

            res_kl_divs = res_kl_divs / cnt
            res_overlaps = res_overlaps / cnt
            # res_mrr = res_mrr / cnt
            # res_hits1 = res_hits1 / cnt
            # res_hits3 = res_hits3 / cnt
            # res_hits5 = res_hits5 / cnt
            # res_hits10 = res_hits10 / cnt

            res_kl_divs = res_kl_divs.save()
            res_overlaps = res_overlaps.save()
            # res_mrr = res_mrr.save()
            # res_hits1 = res_hits1.save()
            # res_hits3 = res_hits3.save()
            # res_hits5 = res_hits5.save()
            # res_hits10 = res_hits10.save()

    return res_kl_divs, res_overlaps
# , res_mrr, res_hits1, res_hits3, res_hits5, res_hits10

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
            bnb_8bit_compute_dtype=torch.bfloat16,
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
    
    res_kl_divs, res_overlaps = run_logitlens(llm, prompts)
    # , res_mrr, res_hits1, res_hits3, res_hits5, res_hits10
    torch.save({
        "res_kl_divs": res_kl_divs.detach().cpu(),
        "res_overlaps": res_overlaps.detach().cpu(),
        # "res_mrr": res_mrr.detach().cpu(),
        # "res_hits1": res_hits1.detach().cpu(),
        # "res_hits3": res_hits3.detach().cpu(),
        # "res_hits5": res_hits5.detach().cpu(),
        # "res_hits10": res_hits10.detach().cpu(),
    }, f"{output_dir}/logitlens_{N_EXAMPLES}exps.pt")

    # plot the results
    plt.figure(figsize=(5,2))
    plt.bar(range(len(llm.model.layers)), res_kl_divs.cpu().numpy())
    plt.ylabel("KL Divergence")
    plt.xlabel("Layer")
    plt.title("LogitLens KL Divergence: " + model_name)
    plt.savefig(f"{output_dir}/logitlens_kl_{N_EXAMPLES}exps.pdf", bbox_inches='tight')

    plt.figure(figsize=(5,2))
    plt.bar(range(len(llm.model.layers)), res_overlaps.cpu().numpy())
    plt.ylabel("Overlap")
    plt.xlabel("Layer")
    plt.title("LogitLens Overlap: " + model_name)
    plt.savefig(f"{output_dir}/logitlens_overlap_{N_EXAMPLES}exps.pdf", bbox_inches='tight')