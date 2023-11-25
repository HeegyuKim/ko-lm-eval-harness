export CUDA_VISIBLE_DEVICES=1
PLM="mistralai/Mistral-7B-v0.1"
eval() {
    PEFT=$1 #"heegyu/42dot-1.3B-mt-KOpen-Platypus-dolphin"
    PEFT_REVISION=$2 #"steps-25000"

    python main.py \
    --model hf-causal-experimental \
    --model_args pretrained="$PLM",peft="$PEFT",dtype='float16',peft_revision="$PEFT_REVISION",max_length=1024 \
    --tasks 'kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg,csatqa_*,haerae_*' \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size 1 \
    --no_cache \
    --output_path "./output/$PEFT-$PEFT_REVISION.json"
}

PEFT='heegyu/Mistral-7B-v0.1-OKI-v20231124-1e-5'
eval $PEFT "epoch-1"
eval $PEFT "epoch-2"
eval $PEFT "epoch-3"