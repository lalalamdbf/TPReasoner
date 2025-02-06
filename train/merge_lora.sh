CUDA_VISIBLE_DEVICES=0 python ./src/export_model.py \
    --model_name_or_path ./models/Llama-2-7b-hf  \
    --adapter_name_or_path ./checkpoint/reclor/cd/Llama-2-7b-hf-cd \
    --template alpaca \
    --finetuning_type lora \
    --export_dir ./checkpoint/reclor/cd/Llama-2-7b-hf-cd-merged \
    --export_size 2 \
    --export_legacy_format False