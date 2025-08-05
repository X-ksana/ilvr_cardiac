#!/bin/bash
# Sampling script for diffusion model with mask support

# SLURM configuration
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --job-name=dm_sample_mask
#SBATCH --output=sample_mask_%j.log
#SBATCH --error=sample_mask_%j.err

# Print start time
start_time=`date +%s`
echo "Job started at: $(date)"

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define the values for D and N
D_values=(2 4 8)
N_values=(25 50)

# Define the base command and other flags
MODEL_FLAGS="--attention_resolutions 16 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --in_channels 4 --out_channels 8 --mask_channels 3"
COMMON_ARGS="--batch_size 1 --num_samples 100 --timestep_respacing 100 --model_path /scratch/scxcw/results/July2025_dm/128_log_1000_mask/best_ema_0.9999_409298.pt   --base_samples /scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset027_ACDC/nnUNetPlans_2d
 --scale 6"

#python scripts/ilvr_sample.py  --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --model_path /scratch/scxcw/results/July2025_dm/128_log_1000_mask/best_ema_0.9999_409298.pt --base_samples /scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset027_ACDC/nnUNetPlans_2d--down_N 32 --range_t 20 --save_dir output

# Loop over each combination of D and N
for D in "${D_values[@]}"; do
    for N in "${N_values[@]}"; do
        # Define a unique save directory
        SAVE_DIR="results/samples/mask_D${D}_N${N}"

        # Construct the specific command for the current values of D and N
        COMMAND="python scripts/image_sample.py $MODEL_FLAGS $COMMON_ARGS --D $D --N $N --save_dir $SAVE_DIR"

        # Print the command being run
        echo "Running: $COMMAND"

        # Run the command
        $COMMAND
    done
done

# Record command execution time
end_command=`date +%s%N`
command_duration=$((end_command - start_command))
formatted_duration=$(printf '%02d:%02d:%02d' $(($command_duration/3600)) $(($command_duration%3600/60)) $(($command_duration%60)))
echo "Command execution time: $formatted_duration (hh:mm:ss)"

# Print end time and total duration
end_time=`date +%s`
echo "Job ended at: $(date)"
total_duration=$(($end_time - $start_time))
formatted_total=$(printf '%02d:%02d:%02d' $(($total_duration/3600)) $(($total_duration%3600/60)) $(($total_duration%60)))
echo "Total job duration: $formatted_total (hh:mm:ss)"
