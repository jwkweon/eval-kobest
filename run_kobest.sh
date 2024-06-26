# ./run_all.sh jyoung105/KoR-Orca-Platypus-13B-neft

export CUDA_VISIBLE_DEVICES=$2
export TOKENIZERS_PARALLELISM=false

RESULT_DIR='results/all'
TASKS='kobest_hellaswag,kobest_copa,kobest_boolq,kobest_sentineg'
GPU_NO=0

MODEL=$1 #/home/beomi/coding-ssd2t/EasyLM/llama-2-ko-7b

MODEL_PATH=$(echo $MODEL | awk -F/ '{print $(NF-1) "/" $NF}')

echo "mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS"
mkdir -p $RESULT_DIR/$MODEL_PATH/$CURRENT_TRAINED_TOKENS

python main.py \
--model hf-causal-experimental \
--model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
--tasks $TASKS \
--num_fewshot 0 \
--no_cache \
--batch_size 256 \
--output_path $RESULT_DIR/$MODEL/0_shot.json

# python main.py \
# --model hf-causal-experimental \
# --model_args pretrained=$MODEL,use_accelerate=true,trust_remote_code=true \
# --tasks $TASKS \
# --num_fewshot 5 \
# --no_cache \
# --batch_size 64 \
# --output_path $RESULT_DIR/$MODEL/5_shot.json