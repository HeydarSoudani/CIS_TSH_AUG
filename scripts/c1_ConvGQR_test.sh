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

srun $HOME/TAConvDR/component1_query_rewriting/ConvGQR/test_GQR.py \
    --model_checkpoint_path $model_checkpoint_path \
    --test_file_path $test_file_path \
    --output_file_path $output_file_path \
    --collate_fn_type "flat_concat_for_test" \
    --decode_type $decode_type \
    --per_gpu_eval_batch_size 32 \
    --max_query_length 32 \
    --max_doc_length 384 \
    --max_response_length 32 \
    --max_concat_length 512

# decode_type: "oracle" for rewrite and "answer" for expansion

