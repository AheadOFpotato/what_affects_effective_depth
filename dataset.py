import datasets
from typing import Optional, List, Dict, Tuple
import random
random.seed(42)

# Dataset for the GSM8K benchmark.
class GSM8K:
    def __init__(self):
        self.dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")

    @staticmethod
    def format_example(example: Dict[str, str]) -> str:
        # Format GSM8K example according to the LM evaluation harness.
        question = example["question"]
        answer = example["answer"].split("####")
        assert len(answer) == 2
        res = f"Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"
        return f"{res}\n{answer[0]}The final answer is {answer[1].strip()}"

    def __iter__(self):
        for example in self.dataset:
            yield self.format_example(example)
            
            
class hellaswag:
    def __init__(self):
        self.dataset = datasets.load_dataset("Rowan/hellaswag", split="validation")
    
    @staticmethod
    def format_example(example: Dict[str, str]) -> str:
        ending = example["endings"][int(example["label"])]
        ctx = example["ctx"]
        if ending.startswith(","):
            return f"{ctx}{ending}"
        else:
            return f"{ctx} {ending}"
        
    def __iter__(self):
        for example in self.dataset:
            yield self.format_example(example)

class AIME24:
    def __init__(self):
        self.dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        print("filtering AIME24 dataset to fit within 400 tokens.")

    @staticmethod
    def format_example(example: Dict[str, str]) -> str:
        question = example["Problem"]
        solution = example["Solution"]

        res = f"Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"\\boxed{{[answer]}}\" where [answer] is the response to the problem.\n\n{solution}"

        return res
    
    def __iter__(self):
        for example in self.dataset:
            res = self.format_example(example)
            if len(self.tokenizer(res).input_ids) > 400:
                continue
            yield res
            
            
class Addition:
    def __init__(self, length):
        self.length = length

    def generate_example(self) -> str:
        a = random.randint(10**(self.length-1), 10**self.length - 1)
        b = random.randint(10**(self.length-1), 10**self.length - 1)
        return f"Give the answer directly: {a} + {b} = {a+b}"

    def __iter__(self):
        for _ in range(100):
            yield self.generate_example()
            
class Multiplication:
    def __init__(self, length):
        self.length = length

    def generate_example(self) -> str:
        a = random.randint(10**(self.length-1), 10**self.length - 1)
        b = random.randint(10**(self.length-1), 10**self.length - 1)
        return f"Give the answer directly: {a} * {b} = {a*b}"

    def __iter__(self):
        for _ in range(100):
            yield self.generate_example()