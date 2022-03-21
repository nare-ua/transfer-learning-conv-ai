import sys
import os
sys.path.insert(0, '..')

import logging
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

import csv
import csv
from collections import defaultdict

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
