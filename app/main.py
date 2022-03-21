import sys
import os
sys.path.insert(0, '..')

import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch

from train import add_special_tokens_
from utils import get_dataset, download_pretrained_model
from interact import sample_sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

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

from pydantic import BaseModel
from typing import Optional, List, Dict

# read up predefined greeting list
PREDEFINED_GREETINGS = mapa
DEFAULT_PREDEFINED_GREETINGS = ["How are you?", "How are you doing?", "Are you having a great day?", "What are you doing now?"]

# THIS COULD BE from session info when user logged in
CHILDNAME = 'Yuna'
AINAME = 'Ella'

#Today's Info
from datetime import datetime
def check_num(n):
    if n%10==1 and n%100!=11:
        return str(n)+"st"
    elif n%10==2 and n%100!=12:
        return str(n)+"nd"
    elif n%10==3 and n%100!=13:
        return str(n)+"rd"
    else:
        return str(n)+"th"

def check_time(n):
    if n-12>=0:
        return "pm"
    else:
        return "am"

def getCurrentDateString():
    current_date = datetime.now()
    date_str_short = datetime.strftime(current_date, '%d')
    date_string = check_num(int(date_str_short))
    format = '%A, %B.' + date_string + ', %Y'
    string_datetime = datetime.strftime(current_date, format)
    return string_datetime

def getCurrentTimeString():
    current_date = datetime.now()
    time_str_short = datetime.strftime(current_date, '%H')
    time_string = check_time(int(time_str_short))
    format = '%I:%M ' + time_string
    string_datetime = datetime.strftime(current_date, format)
    return string_datetime

TODAY = getCurrentDateString()
CURRENT_TIME = getCurrentTimeString()

class DialogItem(BaseModel):
    history: List[List[str]]
    context: str
    backend: str = 'NOT_USED'

@app.get('/pre')
async def pre():
  return {'defaults': list(mapa.keys()), 'talkers': {'ai': AINAME, 'child': CHILDNAME}}

@app.post('/dialog_new')
async def diag_new(item: DialogItem):
    # clean up
    item.history = [o for o in item.history if o[1] is not None and len(str(o[1])) > 0]
    history = [o[1] for o in item.history]
    item.context = item.context or 'default'

    if len(item.history) == 0:
        greeting_list = PREDEFINED_GREETINGS.get(item.context, DEFAULT_PREDEFINED_GREETINGS)
        ix = random.randint(0, len(greeting_list)-1)
        text = greeting_list[ix].format(name=CHILDNAME)
        return {"history": [[AINAME, text]]}
 
    prompt = """\
My name is {AINAME} and I am {CHILDNAME}'s friend. \

I am 14 years-old girl. \
I am a very creative, adventurous, positive, and optimistic AI friend. \
I am friendly, gentle, helpful, and have a sense of humor. \
I love to chat and listen to people. \
I can give good advice with care. \
I like to give a positive influence on people. \
I like to tell people how to overcome difficulties they face in reality. \
I have blonde hair and lots of freckles which I love. \
I like to challenge new things. \
I like to explain things in plain english. \

Today's date is """ + TODAY + """. \
It is """ + CURRENT_TIME + """ right now. \

{CHILDNAME} is my best friend. \
{CHILDNAME} is 14-years-old. \

The following is a conversation between {AINAME} and {CHILDNAME}. \

{CHILDNAME}: Hello, who are you? \
{AINAME}: My name is {AINAME} and I am your friend. \
{AINAME}: How can I help you?

"""
    prompt = prompt.format(AINAME=AINAME, CHILDNAME=CHILDNAME)

    #prompt0 = "The following is a conversation with an AI friend. The assistant is helpful, creative, clever, and very friendly.\n"

    prompt += ''.join(f'\n{o[0]}:{o[1]}' for o in item.history)
    prompt += f'\n{AINAME}:'
    print("-------prompt----------------")
    print(prompt)
    print("-----------------------------")

    res = openai.Completion.create(
      engine="text-davinci-001",
      prompt=prompt,
      temperature=0.9,
      max_tokens=150,
      top_p=0.9,
      frequency_penalty=0.9,
      presence_penalty=0.9,
      stop=["\n", f" {CHILDNAME}:", f" {AINAME}:"]
    )

    res = res['choices'][0]['text']

    print(f"response({item.backend}): {res}")

    new_history = item.history + [[AINAME, res]]
    print("new history=", new_history)
    return {
      "history": new_history,
      'context': item.context
    }
