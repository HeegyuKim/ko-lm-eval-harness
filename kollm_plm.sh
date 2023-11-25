export CUDA_VISIBLE_DEVICES=1

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="42dot/42dot_LLM-PLM-1.3B",dtype='float16',max_length=1024 \
#     --tasks 'kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,csatqa_*,haerae_*' \
#     --num_fewshot 0 \
#     --device cuda:0 \
#     --batch_size 4 \
#     --no_cache \
#     --output_path "./output/42dot_LLM-PLM-1.3B.json"

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained="mistralai/Mistral-7B-v0.1",dtype='float16',max_length=1024 \
#     --tasks 'kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,csatqa_*,haerae_*' \
#     --num_fewshot 0 \
#     --device cuda:0 \
#     --batch_size 1 \
#     --no_cache \
#     --output_path "./output/Mistral-7B-v0.1.json"

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="Minirecord/Mini_synatra_7b_02",dtype='float16',max_length=1024 \
    --tasks 'kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,csatqa_*,haerae_*' \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size 1 \
    --no_cache \
    --output_path "./output/Mini_synatra_7b_02.json"

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="beomi/llama-2-ko-7b",dtype='float16',max_length=1024 \
    --tasks 'kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,csatqa_*,haerae_*' \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size 1 \
    --no_cache \
    --output_path "./output/llama-2-ko-7b.json"
