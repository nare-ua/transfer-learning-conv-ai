docker run --rm -it \
  -v /mnt/data/transformers_cache:/root/.cache \
  -v /home/yuntai/git/transfer-learning-conv-ai:/workspace \
  -p 5000:5000 \
  convai bash
