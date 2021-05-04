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

OPENAI_API_KEY="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"
#
import os
import openai
openai.api_key = OPENAI_API_KEY

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import csv
from collections import defaultdict

diag_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
diag_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")


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

mapa = defaultdict(list)
with open('ver1.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')

    k, v = next(spamreader)
    mapa[k].append(v)
    prev = k
    for k, v in spamreader:
        if k == '': k = prev
        else: prev = k
        mapa[k].append(v)

print(mapa.keys())
print("GREETING FILE LOADED....")

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

class DialogItem(BaseModel):
    history: List[str]
    context: str


# read up predefined greeting list
PREDEFINED_GREETINGS_OLD = {
    'entrance': ["you come back again?", "fuck off", "you are not welcomed here", "what did you learn today?", "why did you come back"],
    'kitchen': ["are you pigging out again?", "stop fuck eating", "we have no food"],
    'morning': ["good morning", "i hoped you don't wake up", "sleep more"],
    'evening': ["good evening", "how was your day", "you can't afford me. you dumb ass uncle fucker"],
    #'goodnight': ["good night", "good sleep baby", "don't look at me you faggot"],
    'goodnight': ["The U.S. is ready to engage in talks about North Korea’s nuclear program even as it maintains pressure on Kim Jong Un’s regime, the Washington Post reported, citing an interview with Vice President Mike Pence. "],
    'default': ["How are you?", "how are you doing?", "you look so bored", "don't come near to me"]
}
PREDEFINED_GREETINGS = mapa
DEFAULT_PREDEFINED_GREETINGS = ["How are you?", "how are you doing?", "you look so bored", "don't come near to me"]

CHILDNAME = 'Julie'
ALIANAME = 'ALIA'

class DialogItem2(BaseModel):
    history: List[List[str]]
    context: str
    backend: str = 'dialoggpt'

@app.get('/pre')
async def pre():
    return list(mapa.keys())

@app.post('/dialog_new')
async def diag_new(item: DialogItem2):
    tokenizer = diag_tokenizer
    model = diag_model
    # clean up
    item.history = [o for o in item.history if o[1] is not None and len(str(o[1])) > 0]
    history = [o[1] for o in item.history]
    item.context = item.context or 'default'

    if len(item.history) == 0:
        greeting_list = PREDEFINED_GREETINGS.get(item.context, DEFAULT_PREDEFINED_GREETINGS)
        ix = random.randint(0, len(greeting_list)-1)
        text = greeting_list[ix].format(name=CHILDNAME)
        return {"history": [[ALIANAME, text]]}

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    if item.backend == 'dialoggpt':
        chat_history_ids = [tokenizer.encode(o + tokenizer.eos_token, return_tensors='pt') for o in history]
        #chat_history_ids += [new_user_input_ids]

        # append the new user input tokens to the chat history
        chat_history_ids = torch.cat(chat_history_ids, dim=-1)

        end_ix = chat_history_ids.shape[-1]

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(chat_history_ids, max_length=100,top_k=50,do_sample=True, top_p=0.95)
        res = tokenizer.decode(chat_history_ids[:, end_ix:][0], skip_special_tokens=True)
    else:
        prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"
        prompt += ''.join(f'\n{o[0]}:{o[1]}' for o in item.history)
        prompt += f'\n{ALIANAME}:'
        print("-------prompt----------------")
        print(prompt)
        print("-----------------------------")

        res = openai.Completion.create(
          engine="davinci",
          prompt=prompt,
          temperature=0.9,
          max_tokens=150,
          top_p=1,
          frequency_penalty=0.0,
          presence_penalty=0.6,
          stop=["\n", f" {CHILDNAME}:", f" {ALIANAME}:"]
        )

        res = res['choices'][0]['text']

    print(f"response({item.backend}): {res}")

    new_history = item.history + [[ALIANAME, res]]
    print("new history=", new_history)
    return {"history": new_history, 'context': item.context}

import random
@app.post('/dialog')
async def toto2(item: DialogItem):
    tokenizer = diag_tokenizer
    model = diag_model

    # clean up
    item.history = [c for c in item.history if c is not None and len(str(c)) > 0]

    item.context = item.context or 'greeting'
    print("item.context=", item.context)

    if len(item.history) == 0:
        greeting_list = PREDEFINED_GREETINGS.get(item.context, PREDEFINED_GREETINGS['default'])
        ix = random.randint(0, len(greeting_list)-1)
        text = greeting_list[ix].format(name=CHILDNAME)
        return {"history": [text]}

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    chat_history_ids = [tokenizer.encode(o + tokenizer.eos_token,
                                         return_tensors='pt') for o in
                        item.history]
    #chat_history_ids += [new_user_input_ids]

    # append the new user input tokens to the chat history
    chat_history_ids = torch.cat(chat_history_ids, dim=-1)

    end_ix = chat_history_ids.shape[-1]

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(chat_history_ids, max_length=100,top_k=50,do_sample=True, top_p=0.95)
    res = tokenizer.decode(chat_history_ids[:, end_ix:][0], skip_special_tokens=True)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(res))

    return {"history": item.history + [res]}

@app.get('/shuffle')
async def shuffle():
  return {"persona": [tokenizer.decode(p) for p in select_personalities()]}
