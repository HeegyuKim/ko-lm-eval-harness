export CUDA_VISIBLE_DEVICES=0
PLM="42dot/42dot_LLM-PLM-1.3B"
eval() {
    PEFT=$1 #"heegyu/42dot-1.3B-mt-KOpen-Platypus-dolphin"
    PEFT_REVISION=$2 #"steps-25000"

    python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="$PLM",dtype='float16',max_length=1024,peft="$PEFT",peft_revision="$PEFT_REVISION" \
    --tasks 'kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,csatqa_*,haerae_*' \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size 4 \
    --no_cache \
    --output_path "./output/$PEFT-$PEFT_REVISION.json"
}

eval "heegyu/42dot-1.3B-KOR-OpenOrca-Platypus-5e-5" "epoch-1"
eval "heegyu/42dot-1.3B-KOR-OpenOrca-Platypus-5e-5" "epoch-2"
eval "heegyu/42dot-1.3B-KOR-OpenOrca-Platypus-5e-5" "epoch-3"