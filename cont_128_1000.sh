#!/bin/bash
# Training script for diffusion model with mask support

# SLURM configuration
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/users/scxcw/ilvr_cardiac/logs/1000_%j.log
#SBATCH --error=/users/scxcw/ilvr_cardiac/logs/1000_%j.err
#SBATCH --mail-user=scxcw@leeds.ac.uk

# Print start time
start_time=`date +%s`
echo "Job started at: $(date)"

# Change to project directory
cd /users/scxcw/ilvr_cardiac

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Securely load the API key from the local, untracked file
if [ -f "secrets.sh" ]; then
    source secrets.sh
else
    echo "Warning: secrets.sh not found. WANDB_API_KEY may not be set."
fi

# Create log directory if it doesn't exist
mkdir -p /scratch/scxcw/results/July2025_dm/128_log_1000_mask

# Run training with correct channel configuration
# Configuration: 1 image channel + 1 mask channel = 2 input channels
# With learn_sigma=True: 4 output channels (2 * 2)
python scripts/image_train.py \
    --resume_checkpoint /scratch/scxcw/results/July2025_dm/128_log_1000_mask/model330000.pt \
    --resume_wandb_id vluyocdn \ 
    --data_dir /scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset114_MNMs/nnUNetPlans_2d \
    --log_dir /scratch/scxcw/results/July2025_dm/128_log_1000_mask \
    --attention_resolutions 16 \
    --class_cond False \
    --diffusion_steps 1000 \
    --dropout 0.0 \
    --image_size 128 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 128 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --timestep_respacing 100 \
    --in_channels 4 \
    --out_channels 8 \
    --mask_dir /scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset114_MNMs/nnUNetPlans_2d \
    --num_mask_classes 4 \
    --lr 1e-4 \
    --batch_size 32 \
    --microbatch 4 \
    --log_samples_interval 1000 \
    --log_interval 100 \


# Record command execution time
end_command=`date +%s%N`
command_duration=$((end_command - start_command))

# Print end time and duration
end_time=`date +%s`
runtime=$((end_time - start_time))
echo "Job ended at: $(date)"
echo "Total job duration: $(printf '%02d:%02d:%02d' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))) (hh:mm:ss)"

