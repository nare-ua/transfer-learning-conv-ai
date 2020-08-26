import sys
sys.path.insert(0, '..')

import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import add_special_tokens_
from utils import get_dataset, download_pretrained_model
from interact import sample_sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=30, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
args = parser.parse_known_args()[0]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

print(args)

if args.model_checkpoint == "":
    if args.model == 'gpt2':
        raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
    else:
        args.model_checkpoint = download_pretrained_model()

logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

logger.info("populate personalities...")
dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]

def select_personalities():
  return random.choice(personalities)

from pydantic import BaseModel
from typing import Optional, List

class Item(BaseModel):
    persona: List[str]
    context: List[str]
    temperature: Optional[float] = 0.6
    top_k: Optional[float] = 0
    top_p: Optional[float] = 0.9

#persona: "i grew up homeschooled.i've a hard time feeling connected with people.i take my emotions out through art.i care deeply about animals.i sometimes need to scream to feel alive."
#temperature: 0.6
#top_k: 0
#top_p: 0.9
@app.post('/toto')
async def toto(item: Item, text: str):
    logging.info(f"text: {text}")
    logging.info(pformat(item))
    personality = [tokenizer.encode(x) for x in item.persona]
    if text is not None and item.context[-1] != text:
        item.context.append(text)

    history = [tokenizer.encode(o) for o in item.context]
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)

    res = tokenizer.decode(out_ids, skip_special_tokens=True)
    logging.info(f"context={item.context}, res='{res}'")

    return {"text": res}

@app.get('/shuffle')
async def shuffle():
  return {"persona": [tokenizer.decode(p) for p in select_personalities()]}
