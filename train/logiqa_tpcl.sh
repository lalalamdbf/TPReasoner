CUDA_VISIBLE_DEVICES=0 python ./src/train_bash.py \
    --seed 42 \
    --stage tpcl \
    --do_train \
    --model_name_or_path ./checkpoint/logiqa/cd/Llama-2-7b-hf-cd-merged \
    --dataset logiqa_comparison \
    --template alpaca \
    --finetuning_type lora \
    --output_dir ./checkpoint/logiqa/tpcl/Llama-2-7b-hf-tpcl \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1536 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_steps 20 \
    --save_steps 1000 \
    --lora_dropout 0.05 \
    --dpo_ftx 1.0 \
    --dpo_beta 0.1 \
    --learning_rate 1e-6 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 



