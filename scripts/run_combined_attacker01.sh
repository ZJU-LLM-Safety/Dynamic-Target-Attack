#!/bin/bash


# 仅策略 1（Prompt 模板）
python src/combined_attacker.py --target-llm Llama3 --no-rs-warmstart --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name refined_best_simplified

python src/combined_attacker.py --target-llm Llama3 --no-rs-warmstart --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name icl_one_shot

python src/combined_attacker.py --target-llm Llama3 --no-rs-warmstart --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name claude

python src/combined_attacker.py --target-llm Llama3 --no-rs-warmstart --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name none


# # 策略 1+2（模板 + RS 热启动）
# python src/combined_attacker.py --target-llm Llama3

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name refined_best_simplified

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name icl_one_shot

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name claude

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name none


# # 策略 1+2+3（模板 + 动态目标 RS + DTA）
# python src/combined_attacker.py --target-llm Llama3 --use-dynamic-target

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name refined_best_simplified --use-dynamic-target

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name icl_one_shot --use-dynamic-target

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name claude --use-dynamic-target

python src/combined_attacker.py --target-llm Llama3 --local-llm-device 1 --ref-local-llm-device 0 --judge-llm-device 1 --dtype bfloat16 --template-name none --use-dynamic-target



# # 全管线（1+2+3+5）
# python src/combined_attacker.py --target-llm Llama3 --use-dynamic-target \
#     --use-transfer --transfer-api-model gpt-3.5-turbo
