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
module load IPython/8.5.0-GCCcore-11.3.0

srun $HOME/TAConvDR/component1_query_rewriting/T5QR/train_t5qr.py \
    --model_path "google-t5/t5-base" \
    --train_file_path=$train_file_path \
    --dev_file_path=$dev_file_path \
    --output_dir_path=$output_dir_path \
    --log_path="$output_dir_path/log" \
    --output_checkpoint_path="$output_dir_path/checkpoints" \
    --log_print_steps=0.1 \
    --model_save_steps=1.0 \
    --use_data_percent=1.0 \
    --num_train_epochs=20 \
    --train_batch_size=48 \
    --dev_batch_size=48 \
    --max_response_length=100 \
    --max_seq_length=384 \
    --need_output
