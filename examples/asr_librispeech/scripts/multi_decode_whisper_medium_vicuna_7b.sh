#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1


# multi demo
inference_device="cuda" #"cpu", "cuda:0", "cuda:1"
CUDA_VISIBLE_DEVICES="0,1,2,3"
# dataset
feats_dir="/ssd/zhuang/code/FunASR/examples/asru2019/DATA" #feature output dictionary
#test_sets="dev_clean dev_other test_clean test_other"
test_sets="test"
#test_sets="test"
inference_scp="audio_datasets.jsonl"
# model detail
exp_dir=/ssd/zhuang/code/SLAM-LLM/examples/asr_librispeech
model_dir=whisper-medium-linear-qwen-v2_5-7B-asru2019-20250410/model
inference_checkpoint=asr_epoch_2_step_36348.pt
ckpt_path=$exp_dir/exp/$model_dir
speech_encoder_path=/ssd/zhuang/code/LLM/whisper/medium.pt
llm_path=/ssd/zhuang/code/LLM/Qwen2.5-7B-Instruct-1M


if [ ${inference_device} == "cuda" ]; then
    nj=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
else
    inference_batch_size=1
    CUDA_VISIBLE_DEVICES=""
    for JOB in $(seq ${nj}); do
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"-1,"
    done
fi


for dset in ${test_sets}; do
  inference_dir="${exp_dir}/exp/${model_dir}/inference-${inference_checkpoint}/${dset}"
  _logdir="${inference_dir}/logdir"
  echo "inference_dir: ${inference_dir}"

  mkdir -p "${_logdir}"
  data_dir="${feats_dir}/data/${dset}"
  key_file=${data_dir}/${inference_scp}

  split_scps=
  for JOB in $(seq "${nj}"); do
      split_scps+=" ${_logdir}/keys.${JOB}.jsonl"
  done
  /ssd/zhuang/code/SLAM-LLM/src/slam_llm/inference/utils/split_scp.pl "${key_file}" ${split_scps}

  gpuid_list_array=(${CUDA_VISIBLE_DEVICES//,/ })
  for JOB in $(seq ${nj}); do
      {
        id=$((JOB-1))
        gpuid=${gpuid_list_array[$id]}
        export CUDA_VISIBLE_DEVICES=${gpuid}
        python $exp_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$inference_dir \
        ++model_config.llm_name="qwen-v2.5-7b" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=3584 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path="${_logdir}/keys.${JOB}.jsonl" \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=80 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$ckpt_path \
        ++decode_log="${inference_dir}/${JOB}" \
        ++ckpt_path="${ckpt_path}/${inference_checkpoint}" &> ${_logdir}/log.${JOB}.txts
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
      }&

  done
  wait

  mkdir -p ${inference_dir}/1best_recog
  for f in token score text; do
      if [ -f "${inference_dir}/${JOB}/1best_recog/${f}" ]; then
        for JOB in $(seq "${nj}"); do
            cat "${inference_dir}/${JOB}/1best_recog/${f}"
        done | sort -k1 >"${inference_dir}/1best_recog/${f}"
      fi
  done
done