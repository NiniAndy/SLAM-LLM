#!/bin/bash
root=/ssd/zhuang/code
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=$root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=$root/SLAM-LLM
cd $run_dir
code_dir=examples/phoneme_llm
project_dir=$run_dir/$code_dir
echo "Project dir: $project_dir"
export TORCH_ELASTIC_ERROR_FILE=$project_dir/torch_error.log


llm_path=/ssd/zhuang/code/LLM/Qwen2.5-3B-Instruct/
train_data_path=/ssd/zhuang/code/FunASR/examples/code_switching/DATA/libri_aishell_asru_data/train/audio_datasets.jsonl
val_data_path=/ssd/zhuang/code/FunASR/examples/asru2019/DATA/data/test/audio_datasets.jsonl

output_dir=$project_dir/exp
mkdir -p "${output_dir}"
output_dir=$output_dir/qwen-v2_5-3B-mixcs-$(date +"%Y%m%d")
echo "Save path: $output_dir"


hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=qwen-v2.5-3b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2048 \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++train_config.model_name=asr \
++train_config.num_epochs=10 \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=150000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=4000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.gradient_accumulation_steps=2 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++log_config.log_file=$output_dir/log.txt \
++log_config.log_interval=500 \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_phoneme_llm.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 4 \
        --master_port=29503 \
        $code_dir/finetune_phoneme_llm.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false \
        ++train_config.use_fp16=true \
        $hydra_args
fi
