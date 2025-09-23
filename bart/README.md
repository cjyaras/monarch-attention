# Monarch Attention for BART

Testing Monarch Attention on long sequence summarization task using the [BART](https://huggingface.co/facebook/bart-base) model and the [Booksum](https://huggingface.co/datasets/kmfoda/booksum) dataset.

## Training

[BART](https://huggingface.co/facebook/bart-base) is an encoder-decoder model which is capable of text summarization.
We use the BART model to summarize the book chapter in the [Booksum](https://huggingface.co/datasets/kmfoda/booksum) dataset.

Unfortunately, the pre-trained BART supports the sequence length up to 1024. To increase the sequence length limit, we fine-tuned the model with the maximum input length of 8192 and the maximum summary length of 512.

To train the model, install packages below:
```bash
pip install wandb datasets accelerate
```

Example training command with the softmax attention:
```bash
#!/bin/bash
lr=1e-4
wd=0.0
epochs=5


export SCRIPT_ARGS=" \
--run_name bart-base-booksum \
--model_name_or_path facebook/bart-base \
--output_dir bart/finetuned/output \
--dataset_name kmfoda/booksum \
--text_column chapter \
--summary_column summary_text \
--num_train_epochs $epochs \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing \
--eval_strategy steps \
--eval_steps 400 \
--save_strategy epoch \
--save_total_limit 1 \
--learning_rate $lr \
--weight_decay $wd \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--do_train \
--do_eval \
--bf16 \
--bf16_full_eval \
--ddp_timeout 10800 \
--report_to wandb \
--dataloader_pin_memory \
--dataloader_num_workers 4 \
--trust_remote_code \
--max_source_length 8192 \
--max_target_length 512 \
--predict_with_generate \
"
# change --num_processes for different number of GPUs.
accelerate launch --num_processes 2 --main_process_port 29551 bart/run_summarization.py $SCRIPT_ARGS
```

## Fine-tuning BART with `monarchattention`

The following script trains a BART model after replacing the softmax attention with the `monarchattention`.
```bash
lr=1e-4
wd=0.0
epochs=5

ATTN_TYPE="monarch-attention"
BLOCK_SIZE=64
NUM_STEPS=2
export SCRIPT_ARGS=" \
--run_name bart-base-booksum-${ATTN_TYPE}-b${BLOCK_SIZE}-t${NUM_STEPS} \
--model_name_or_path facebook/bart-base \
--output_dir $SCRATCH_DIR/finetuned_ma/b${BLOCK_SIZE}_t${NUM_STEPS}/output \
--dataset_name kmfoda/booksum \
--text_column chapter \
--summary_column summary_text \
--num_train_epochs $epochs \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing \
--eval_strategy steps \
--eval_steps 400 \
--save_strategy epoch \
--save_total_limit 1 \
--learning_rate $lr \
--weight_decay $wd \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--do_train \
--do_eval \
--bf16 \
--bf16_full_eval \
--ddp_timeout 10800 \
--report_to wandb \
--dataloader_pin_memory \
--dataloader_num_workers 4 \
--trust_remote_code \
--max_source_length 8192 \
--max_target_length 512 \
--predict_with_generate \
--attention_type ${ATTN_TYPE} \
--block_size ${BLOCK_SIZE} \
--num_steps ${NUM_STEPS}
"
# change --num_processes for different number of GPUs.
accelerate launch --num_processes 2 --main_process_port 29551 bart/run_summarization.py $SCRIPT_ARGS
```

## Evaluation

Run `bart/benchmark.py --model_checkpoint_path="./bart/finetuned/output/"`. The script loads the fine-tuned model from `bart/finetuned/output/`.
