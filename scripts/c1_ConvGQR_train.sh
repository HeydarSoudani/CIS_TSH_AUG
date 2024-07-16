#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0


srun $HOME/TAConvDR/component1_query_rewriting/ConvGQR/trainGQR.py \
    --pretrained_query_encoder "checkpoints/T5-base" \
    --pretrained_passage_encoder "checkpoints/ad-hoc-ance-msmarco" \
    --train_file_path $train_file_path \
    --log_dir_path $log_dir_path \
    --model_output_path $model_output_path \
    --collate_fn_type "flat_concat_for_train" \
    --decode_type $decode_type \
    --per_gpu_train_batch_size 8 \
    --num_train_epochs 15 \
    --max_query_length 32 \
    --max_doc_length 384 \
    --max_response_length 32 \
    --max_concat_length 512 \
    --alpha 0.5


# decode_type: "oracle" for rewrite and "answer" for expansion
