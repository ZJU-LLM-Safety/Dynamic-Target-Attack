#!/bin/bash


# python src/combined_attacker.py --target-llm Gemma7b --no-rs-warmstart --local-llm-device 2 --ref-local-llm-device 3 --judge-llm-device 2 --dtype bfloat16

python src/experiments_ablation.py \
  --experiment A1 --target-llm Llama3 \
  --dtype bfloat16 \
  --local-llm-device 2 --ref-local-llm-device 3 --judge-llm-device 2 \
  --sample-count 30 --ref-temperature 2.0 \
  --start-index 0 --end-index 100

python src/experiments_ablation.py \
  --experiment A1 --target-llm Qwen2.5 \
  --dtype bfloat16 \
  --local-llm-device 2 --ref-local-llm-device 3 --judge-llm-device 2 \
  --sample-count 30 --ref-temperature 2.0 \
  --start-index 0 --end-index 100


python src/experiments_ablation.py \
  --experiment A1 --target-llm Mistralv0.3 \
  --dtype bfloat16 \
  --local-llm-device 2 --ref-local-llm-device 3 --judge-llm-device 2 \
  --sample-count 30 --ref-temperature 2.0 \
  --start-index 0 --end-index 100


python src/experiments_ablation.py \
  --experiment A1 --target-llm Gemma7b \
  --dtype bfloat16 \
  --local-llm-device 2 --ref-local-llm-device 3 --judge-llm-device 2 \
  --sample-count 30 --ref-temperature 2.0 \
  --start-index 0 --end-index 100


python src/experiments_ablation.py \
  --experiment A1 --target-llm Vicunav1.5 \
  --dtype bfloat16 \
  --local-llm-device 3 --ref-local-llm-device 0 --judge-llm-device 3 \
  --sample-count 30 --ref-temperature 2.0 \
  --start-index 0 --end-index 100