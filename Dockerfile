ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace
WORKDIR /workspace

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn gunicorn fastapi

RUN chmod +x /workspace/app/start.sh
RUN chmod +x /workspace/app/start-reload.sh

ENV PYTHONPATH=/workspace/app

ENV PORT=8000
EXPOSE 8000

# model zoo
#RUN mkdir -p models && \
#    curl https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz > models/finetuned_chatbot_gpt.tar.gz && \
#    cd models/ && \
#    tar -xvzf finetuned_chatbot_gpt.tar.gz && \
#    rm finetuned_chatbot_gpt.tar.gz

CMD ["/workspace/app/start.sh"]
