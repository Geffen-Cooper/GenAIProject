#!/bin/bash

# Define the list of items
corruptions=("frost" "gaussian_noise" "glass_blur" "contrast" "pixelate")

# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..10}; do
#         python train.py --logname=fc_eval --batch_size=16 --lr=3e-3 --corr=$corr --seed=$i --finetune_config=fc &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#     done
# done
# wait


# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..10}; do
#         python train.py --logname=first_conv_eval --batch_size=16 --lr=3e-4 --corr=$corr --seed=$i --finetune_config=first_conv &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#     done
# done
# wait

# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..10}; do
#         python train.py --logname=last_three_bn_eval --batch_size=16 --lr=3e-3 --corr=$corr --seed=$i --finetune_config=last_three_bn &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#     done
# done
# wait

# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..10}; do
#         python train.py --batch_size=16 --lr=3e-3 --corr=$corr --seed=$i --finetune_config=none &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#     done
# done
# wait

# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..10}; do
#         python train.py --logname=last_two_bn_eval --batch_size=16 --lr=3e-3 --corr=$corr --seed=$i --finetune_config=last_two_bn &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#     done
# done
# wait

# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..10}; do
#         python train.py --logname=last_bn_eval --batch_size=16 --lr=3e-3 --corr=$corr --seed=$i --finetune_config=last_bn &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#     done
# done
# wait





# Create training set
# =========================

# # Log file path
# LOG_FILE="gen_data_script_log.txt"

# # Get the start time
# start_time=$(date +"%Y-%m-%d %H:%M:%S")

# # Log the start time
# echo "Script started at: $start_time" >> "$LOG_FILE"

# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..1000}; do
#         python train.py --logname=gen_last_bn --batch_size=16 --lr=1e-3 --epochs=10 --corr=$corr --seed=$i --finetune_config=last_bn &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#         echo "corr: $corr, i: $i" >> "$LOG_FILE"
#     done
# done
# wait

# # Log the end time
# echo "Script ended at: $end_time" >> "$LOG_FILE"

# # Calculate total time
# start_seconds=$(date -d "$start_time" +%s)
# end_seconds=$(date -d "$end_time" +%s)
# total_seconds=$((end_seconds - start_seconds))

# # Format the total time
# hours=$((total_seconds / 3600))
# minutes=$(( (total_seconds % 3600) / 60 ))
# seconds=$((total_seconds % 60))

# # Log the total time
# echo "Total time taken: ${hours}h ${minutes}m ${seconds}s" >> "$LOG_FILE"

# Log file path
LOG_FILE="gen_data_script_log5.txt"

# Get the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# Log the start time
echo "Script started at: $start_time" >> "$LOG_FILE"

# Loop through each item in the list
for corr in "${corruptions[@]}"; do
    # Loop from 1 to 10
    for i in {1..1000}; do
        python train.py --logname=gen_last_bn_eval_undiverse --batch_size=16 --lr=1e-2 --epochs=5 --corr=$corr --seed=$i --finetune_config=last_bn &
        # Limit the number of concurrent processes to 4
        if (( $(jobs -r -p | wc -l) >= 4 )); then
            wait -n
        fi
        echo "corr: $corr, i: $i" >> "$LOG_FILE"
    done
done
wait

# Log the end time
echo "Script ended at: $end_time" >> "$LOG_FILE"

# Calculate total time
start_seconds=$(date -d "$start_time" +%s)
end_seconds=$(date -d "$end_time" +%s)
total_seconds=$((end_seconds - start_seconds))

# Format the total time
hours=$((total_seconds / 3600))
minutes=$(( (total_seconds % 3600) / 60 ))
seconds=$((total_seconds % 60))

# Log the total time
echo "Total time taken: ${hours}h ${minutes}m ${seconds}s" >> "$LOG_FILE"