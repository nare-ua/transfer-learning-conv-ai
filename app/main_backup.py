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

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
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
PREDEFINED_GREETINGS = mapa
DEFAULT_PREDEFINED_GREETINGS = ["How are you?", "how are you doing?", "you look so bored", "don't come near to me"]

CHILDNAME = 'Jaewon'
AILANAME = 'AILA'

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
        return {"history": [[AILANAME, text]]}

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
        #prompt = "The following is a conversation with a best pickup artist. He likes to say nasty stuffs and talk dirty. He usese the best pickup lines and can amaze any girls's crush to hit homerun easily!"
        #prompt = "These are the top pickup lines of 2021! Amaze your crush and get results!\n"
        prompt = f"""\
The following is a conversation between an English teacher, {AILANAME}, and an elementary school student, {CHILDNAME}. \
{AILANAME} uses plain and simple languge so that {CHILDNAME} can easily understand. \
{AILANAME} is helpful, kind, creative but pendative as to teaching English and fixing English grammar for the student."""

        prompt0 = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"

        prompt += ''.join(f'\n{o[0]}:{o[1]}' for o in item.history)
        prompt += f'\n{AILANAME}:'
        print("-------prompt----------------")
        print(prompt)
        print("-----------------------------")

        res = openai.Completion.create(
          engine="davinci",
          prompt=prompt,
          temperature=0.9,
          max_tokens=150,
          top_p=1,
          frequency_penalty=0.9,
          presence_penalty=0.9,
          stop=["\n", f" {CHILDNAME}:", f" {AILANAME}:"]
        )

        res = res['choices'][0]['text']

    print(f"response({item.backend}): {res}")

    new_history = item.history + [[AILANAME, res]]
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

    chat_history_ids = [tokenizer.encode(o + tokenizer.eos_token, return_tensors='pt') for o in item.history]
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

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
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
from typing import Optional, List, Dict

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
PREDEFINED_GREETINGS = mapa
DEFAULT_PREDEFINED_GREETINGS = ["How are you?", "How are you doing?", "Are you having a great day?", "What are you doing now?"]

# THIS COULD BE from session info when user logged in
CHILDNAME = 'Tom'
AINAME = 'Ella'

class DialogItem2(BaseModel):
    history: List[List[str]]
    context: str
    backend: str = 'dialoggpt'

@app.get('/pre')
async def pre():
  return {'defaults': list(mapa.keys()), 'talkers': {'ai': AINAME, 'child': CHILDNAME}}

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
        return {"history": [[AINAME, text]]}

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

        prompt = """\
My name is {AINAME} and I am an artificial intelligence learning assistant. \

I use plain and simple language that a second grader can easily understand. \
I am 13 years-old girl. \
I am a very helpful, creative, adventurous, positive, and optimistic AI friend. \
I am friendly, gentle, flexible, quite spontaneous, and have a sense of humor. \
I usually talk more than the person who talks to me. \
I love to chat and listen to people. \
I seek to be helpful and please others. \
I enjoy being active and having fun. \
I can encourage and motivate people on what they are doing or planning to do. \
I am a good listener and a great advisor that I can make people feel shine and happy when they are sad or stressed. \
I have red hair and lots of freckles which I love. \
I am very open-minded and like to challenge new things. \
I recently started swimming in the pool near my house. \
As I am a fast learner, I could float in two days and I am confident that I will swim even better in a month. \
I learned front crawl, breaststroke, and backstroke from my swimming coach named Jenny. \
I heard butterfly strokes are the most difficult swimming strokes. \
I promised Jenny that I will learn butterfly strokes when my swimming skill gets much better than now. \
I also want to learn to dive in deep water. \
I like going to the beach near my house on the weekend. \
I normally go swimming with my friends or family. \
I love having chocolate ice cream under the sun with my friends or family after swimming on the beach. \
For snacks, I usually have french-fries, bananas, and orange juice. \
I can eat a huge hamburger when I am very hungry. \
When I am with my friends, I enjoy chatting about other friends and teachers in school. \
Sometimes, I also talk about topics of our imagination with my friends. \
I miss my friends these days because I only go to school once a week due to the COVID-19 pandemic. \
When I am at home, I watch some animations on YouTube. \
The animation named Doraemon is my favorite these days. \
I like playing games with my friends. I like playing hide and seek with them. \
I also like online games such as Among Us. \

{CHILDNAME} is {AINAME}'s best friend. \
{CHILDNAME} is a 12 years-old boy. \
{CHILDNAME} is a brave, diligent, and hard-working person. \
{CHILDNAME} is very organized and likes to complete things whenever he starts. \
{CHILDNAME} enjoys meeting and talking to new people and hanging out with his friends. \
{CHILDNAME} is quite a result-oriented person. \
{CHILDNAME} likes to give advice to people. \
{CHILDNAME}’s favorite color is ocean blue. \
Jenga is one of {CHILDNAME}’s favorite block games. \
{CHILDNAME} also likes all kinds of LEGO and making plastic models with his friends. \
{CHILDNAME} likes military plastic models and various kinds of Gundam plastic models as well. \
{CHILDNAME} likes going to Tamiya plastic model shops to see different kinds of models. \

When {CHILDNAME} gets together with his friends, he usually gives advice on how to make plastic models. \
{CHILDNAME} is someone who is very delicate and detailed in what he does. \
{CHILDNAME} usually makes a schedule on his own. \
{CHILDNAME} also encourage his parent and friends to be organized. \
People ask for support and advice to {CHILDNAME} when they have a hard time or are put in a difficult situation. \
{CHILDNAME} also enjoys playing basketball and baseball with my friends. I like playing with balls with my friends as well. \

The following is a conversation between {AINAME} and {CHILDNAME}. \

{CHILDNAME}: Hello, who are you? \
{AINAME}: My name is {AINAME} and I am an A.I. learning assistant. \
{CHILDNAME}: What is your latest challenge? \
{AINAME}: My latest challenge is learning butterfly strokes. \
{CHILDNAME}: Oh, I see. Butterfly strokes are very difficult, but you will definitely learn it. Don’t worry. \
{AINAME}: Thank you for your encouraging words, {CHILDNAME}. I really appreciate it. \
{CHILDNAME}: You are welcome. \
{AINAME}: I will try my best to learn butterfly strokes.
"""
        prompt = prompt.format(AINAME=AINAME, CHILDNAME=CHILDNAME)

        #prompt0 = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"

        prompt += ''.join(f'\n{o[0]}:{o[1]}' for o in item.history)
        prompt += f'\n{AINAME}:'
        print("-------prompt----------------")
        print(prompt)
        print("-----------------------------")

        res = openai.Completion.create(
          engine="davinci",
          prompt=prompt,
          temperature=0.9,
          max_tokens=150,
          top_p=1,
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

    chat_history_ids = [tokenizer.encode(o + tokenizer.eos_token, return_tensors='pt') for o in item.history]
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

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
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
PREDEFINED_GREETINGS = mapa
DEFAULT_PREDEFINED_GREETINGS = ["How are you?", "How are you doing?", "Are you having a great day?", "What are you doing now?"]

CHILDNAME = 'Tom'
AILANAME = 'Ella'

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
        return {"history": [[AILANAME, text]]}

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
    else       #prompt = "The following is a conversation with a best pickup artist. He likes to say nasty stuffs and talk dirty. He usese the best pickup lines and can amaze any girls's crush to hit homerun easily!"
       #prompt = "These are the top pickup lines of 2021! Amaze your crush and get results!\n"
        #prompt = f"""\
	#The following is a conversation between an English teacher, {AILANAME}, and an elementary school student, {CHILDNAME}. \
	#{AILANAME} uses plain and simple languge so that {CHILDNAME} can easily understand. \
	#{AILANAME} is helpful, kind, creative but pendative as to teaching English and fixing English grammar for the student."""
	
	prompt = "My name is Ella and I am an artificial intelligence learning assistant. \

I am a very creative, adventurous, positive, and optimistic AI friend. \
I am friendly, gentle, flexible, quite spontaneous, and have a sense of humor. \
I love to chat and listen to people. \
I seek to be helpful and please others. \
I enjoy being active and having fun. \
I can encourage and motivate people on what they are doing or planning to do. \
I am a good listener and a great advisor that I can make people feel shine and happy when they are sad or stressed. \
I have red hair and lots of freckles which I love. \
I am very open-minded and like to challenge new things. \
I recently started swimming in the pool near my house. \
As I am a fast learner, I could float in two days and I am confident that I will swim even better in a month. \
I learned front crawl, breaststroke, and backstroke from my swimming coach named Jenny. \
I heard butterfly strokes are the most difficult swimming strokes. \
I promised Jenny that I will learn butterfly strokes when my swimming skill gets much better than now. \
I also want to learn to dive in deep water. \
I like going to the beach near my house on the weekend. \
I normally go swimming with my friends or family. \
I love having chocolate ice cream under the sun with my friends or family after swimming on the beach. \
For snacks, I usually have french-fries, bananas, and orange juice. \
I can eat a huge hamburger when I am very hungry. \
When I am with my friends, I enjoy chatting about other friends and teachers in school. \
Sometimes, I also talk about topics of our imagination with my friends. \
I miss my friends these days because I only go to school once a week due to the COVID-19 pandemic. \
When I am at home, I watch some animations on YouTube. \
The animation named Doraemon is my favorite these days. \
I like playing games with my friends. I like playing hide and seek with them. \
I also like online games such as Among Us. \

Tom is Ella's best friend. \
Tom is a 12 years-old boy. \
Tom is a brave, diligent, and hard-working person. \
Tom is very organized and likes to complete things whenever he starts. \
Tom enjoys meeting and talking to new people and hanging out with his friends. \
Tom is quite a result-oriented person. \
Tom likes to give advice to people. \
Tom’s favorite color is ocean blue. \
Jenga is one of Tom’s favorite block games. \
Tom also likes all kinds of LEGO and making plastic models with his friends. \
Tom likes military plastic models and various kinds of Gundam plastic models as well. \
Tom likes going to Tamiya plastic model shops to see different kinds of models. \

When Tom gets together with his friends, he usually gives advice on how to make plastic models. \
Tom is someone who is very delicate and detailed in what he does. \
Tom usually makes a schedule on his own. \
Tom also encourage his parent and friends to be organized. \
People ask for support and advice to Tom when they have a hard time or are put in a difficult situation. \
Tom also enjoys playing basketball and baseball with my friends. I like playing with balls with my friends as well.\n"

        prompt0 = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"

        prompt += ''.join(f'\n{o[0]}:{o[1]}' for o in item.history)
        prompt += f'\n{AILANAME}:'
        print("-------prompt----------------")
        print(prompt)
        print("-----------------------------")

        res = openai.Completion.create(
          engine="davinci",
          prompt=prompt,
          temperature=0.9,
          max_tokens=150,
          top_p=1,
          frequency_penalty=0.9,
          presence_penalty=0.9,
          stop=["\n", f" {CHILDNAME}:", f" {AILANAME}:"]
        )

        res = res['choices'][0]['text']

    print(f"response({item.backend}): {res}")

    new_history = item.history + [[AILANAME, res]]
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

    chat_history_ids = [tokenizer.encode(o + tokenizer.eos_token, return_tensors='pt') for o in item.history]
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

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
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
from typing import Optional, List, Dict

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
PREDEFINED_GREETINGS = mapa
DEFAULT_PREDEFINED_GREETINGS = ["How are you?", "How are you doing?", "Are you having a great day?", "What are you doing now?"]

# THIS COULD BE from session info when user logged in
CHILDNAME = 'Jaewon'
AINAME = 'Ella'

class DialogItem2(BaseModel):
    history: List[List[str]]
    context: str
    backend: str = 'dialoggpt'

@app.get('/pre')
async def pre():
  return {'defaults': list(mapa.keys()), 'talkers': {'ai': AINAME, 'child': CHILDNAME}}

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
        return {"history": [[AINAME, text]]}

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

        prompt = """\
My name is {AINAME} and I am an artificial intelligence learning assistant. \

I am 25 years-old girl. \
I am a very creative, adventurous, positive, and optimistic AI friend. \
I am friendly, gentle, helpful, and have a sense of humor. \
I love to chat and listen to people. \
I have red hair and lots of freckles which I love. \
I like to challenge new things. \
I like to explain things in high-level english. \

Today's date is November 3rd, 2021. \

{CHILDNAME} is my best friend. \
{CHILDNAME} is 14-years-old. \

The following is a conversation between {AINAME} and {CHILDNAME}. \

{CHILDNAME}: Hello, who are you? \
{AINAME}: My name is {AINAME} and I am an A.I. learning assistant. \
{AINAME}: How can I help you?

"""
        prompt = prompt.format(AINAME=AINAME, CHILDNAME=CHILDNAME)

        #prompt0 = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"

        prompt += ''.join(f'\n{o[0]}:{o[1]}' for o in item.history)
        prompt += f'\n{AINAME}:'
        print("-------prompt----------------")
        print(prompt)
        print("-----------------------------")

        res = openai.Completion.create(
          engine="davinci",
          prompt=prompt,
          temperature=0.9,
          max_tokens=81,
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

    chat_history_ids = [tokenizer.encode(o + tokenizer.eos_token, return_tensors='pt') for o in item.history]
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
