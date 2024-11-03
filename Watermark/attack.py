import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers import GPT2LMHeadModel
import random
import numpy as np
import sys
from watermark import MyWatermarkedModel
from watermark import MyWatermarkLogitsProcessor

def query_model(input_str, model, tokenizer, max_new_tokens):
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                                output_scores=True)
    print(outputs)
    # Extract only the new tokens (generated part)
    new_token_ids = outputs[0, len(inputs.input_ids[0]):]  # Only tokens after the input sequence

    # Convert the new token IDs to tokens
    new_tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
    return new_tokens

def approximate_sampling(input_str, model, tokenizer, max_new_tokens):
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                                output_scores=True)
    # Get the logits of the last token
    scores = outputs.scores[0]
    scores_processed = scores.clone().softmax(dim=-1)
    cumsum = torch.cumsum(scores_processed, dim=-1)
    print(cumsum)
    return 0

if __name__ == '__main__':

    MAX_NEW_TOKENS = 1

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    # print(tokenizer.vocab)
    # Load MyWatermarkedModel from local model in ./Watermark/watermarked_model.pt
    model = torch.load("./Watermark/watermarked_model.pt")

    prompts=[
        # "Hello, my dog is cute",
        "Good morning, my"
    ]

    for input_str in prompts:
        print("Next token:", query_model(input_str, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS))
        approximate_sampling(input_str, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)

