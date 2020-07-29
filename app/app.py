import sys
sys.path.insert(0, '.')

import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model
from interact import sample_sequence

from flask_cors import CORS

from flask import (
  Flask,
  jsonify,
  render_template,
  request,
  send_from_directory,
)
def get_slug(personality):
  slug = "-".join([x for x in personality.replace(' ', '.').replace("'",".").replace(",",".").split(".") if len(x) > 0])
  return slug
# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='', static_folder='')
CORS(app)

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.info(pformat(args))

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

logger.info("Sample a personality")
dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]

#SLUG_MAPA = {get_slug(p): p for p in personalities}

def select_personalities():
  return random.choice(personalities)
  #logger.info("Selected personality: %s", tokenizer.decode(chain(*personality_list)))
  #return personality

#personality = select_personalities()
#personality = tokenizer.decode(chain(*personality_list))
#personality_list = [tokenizer.decode(p) for p in personality_list]
#for p in personality_list:
#  print(p)

@app.route('/dist/<path:path>')
def send_js(path):
  return send_from_directory('dist', path)

#persona: "i grew up homeschooled.i've a hard time feeling connected with people.i take my emotions out through art.i care deeply about animals.i sometimes need to scream to feel alive."
#temperature: 0.6
#top_k: 0
#top_p: 0.9
@app.route('/toto2',  methods=['POST'])
def toto2():
  req = request.get_json()
  text = request.args.get('text')
  print("INPUT=", text)
  print(pformat(req))
  personality = [tokenizer.encode(x) for x in req['persona']]
  history = [tokenizer.encode(o) for o in req['context']]
  if text is not None:
    history.append(tokenizer.encode(text))
  with torch.no_grad():
    out_ids = sample_sequence(personality, history, tokenizer, model, args)
  out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
  # TODO: attention output how?
  print("OUT=", out_text)
  return jsonify({"text": out_text})

@app.route('/messages/toto',  methods=['POST'])
def toto():
  req = request.get_json()
  text = request.args.get('text')
  print("text=", text)
  print("req=", req)
  print("persona=", req['persona'].split('.'))
  personality = [tokenizer.encode(x) for x in req['persona'].split('.')]
  print("personality=", personality)
  history = [tokenizer.encode(o['content']) for o in req['context']]
  if text is not None:
    history.append(tokenizer.encode(text))
  print(len(history), history)
  with torch.no_grad():
    out_ids = sample_sequence(personality, history, tokenizer, model, args)
  out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
  # TODO: attention?
  return jsonify({"text": out_text})
  #return jsonify(d)

#@app.route('/persona/<slug>'):
#def persona(slug):
#  SLUG_MAPA[slug]
#  return render_template('index.html', personality_list=[])

@app.route('/shuffle')
def shuffle():
  personality_list = select_personalities()
  personality = "".join([tokenizer.decode(p) for p in personality_list])
  slug = "-".join([x for x in personality.replace(' ', '.').replace("'",".").replace(",",".").split(".") if len(x) > 0])
  print(personality)
  print(slug)
  return jsonify({"slug": slug, 'text': personality})

@app.route('/shuffle2')
def shuffle2():
  return jsonify({"persona": [tokenizer.decode(p) for p in select_personalities()]})

@app.route('/')
def index():
  return render_template('index.html', personality_list=[])

if __name__ == "__main__":
  app.run(host='0.0.0.0')
