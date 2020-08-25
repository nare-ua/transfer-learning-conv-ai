#! /usr/bin/env sh
set -e

MODULE_NAME=app.main
VARIABLE_NAME=${VARIABLE_NAME:-app}
export APP_MODULE=${APP_MODULE:-"$MODULE_NAME:$VARIABLE_NAME"}

DEFAULT_GUNICORN_CONF=/app/gunicorn_conf.py
export GUNICORN_CONF=${GUNICORN_CONF:-$DEFAULT_GUNICORN_CONF}
export WORKER_CLASS=${WORKER_CLASS:-"uvicorn.workers.UvicornWorker"}

# If there's a prestart.sh script in the /app directory or other path specified, run it before starting
PRE_START_PATH=${PRE_START_PATH:-/workspace/app/prestart.sh}
echo "Checking for script in $PRE_START_PATH"
if [ -f $PRE_START_PATH ] ; then
  echo "Running script $PRE_START_PATH"
  . "$PRE_START_PATH"
else 
  echo "There is no script $PRE_START_PATH"
fi

# Start Gunicorn/Unicorn
#env
echo "MODULE_NAME=$MODULE_NAME"
echo "APP_MODULE=$APP_MODULE"
#exec gunicorn -k "$WORKER_CLASS" -c "$GUNICORN_CONF" "/mnt/data/pretrained/tacotron2/tacotron2_1032590_6000_amp" --waveveglow "/mnt/data/pretrained/tacotron2/waveglow_1076430_14000_amp" "$APP_MODULE"
#exec gunicorn -k "$WORKER_CLASS" -c "$GUNICORN_CONF" "$APP_MODULE"
#nvidia-smi
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
echo "PORT=${PORT}"

uvicorn --host 0.0.0.0 --reload "$APP_MODULE" --port ${PORT}
