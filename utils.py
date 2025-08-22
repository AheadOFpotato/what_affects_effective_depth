from nnsight import LanguageModel
from typing import Optional, List, Dict, Tuple

def get_model_path(model_name):
    MODEL_PATH = {
        # deepseek models
        "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        # qwen-1.5b
        "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-Math-1.5B": "Qwen/Qwen2.5-Math-1.5B",
        # qwen-7b
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-Math-7B": "Qwen/Qwen2.5-Math-7B",
        # qwen-14b
        "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
        "Qwen2.5-14B": "Qwen/Qwen2.5-14B",
        # qwen-32b
        "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
        "Qwen2.5-32B": "Qwen/Qwen2.5-32B",
    }
    return MODEL_PATH.get(model_name)

def tokenize(llm: LanguageModel, prompt: str, add_special_tokens: bool = True) -> List[str]:
    # Tokenize a prompt and return the tokens as a list of strings.
    tokens = llm.tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    token_str = [s.replace("Ġ","_") for s in llm.tokenizer.convert_ids_to_tokens(tokens)]
    return token_str