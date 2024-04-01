#!/bin/bash

# Define the list of items
corruptions=("frost" "gaussian_noise" "glass_blur" "contrast" "pixelate")

# # Loop through each item in the list
# for corr in "${corruptions[@]}"; do
#     # Loop from 1 to 10
#     for i in {1..10}; do
#         python train.py --batch_size=32 --lr=1e-3 --corr=$corr --seed=$i --finetune_config=fc &
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
#         python train.py --batch_size=32 --lr=1e-3 --corr=$corr --seed=$i --finetune_config=first_conv &
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
#         python train.py --batch_size=32 --lr=1e-3 --corr=$corr --seed=$i --finetune_config=last_three_bn &
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
#         python train.py --batch_size=32 --lr=1e-3 --corr=$corr --seed=$i --finetune_config=none &
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
#         python train.py --batch_size=32 --lr=1e-3 --corr=$corr --seed=$i --finetune_config=last_two_bn &
#         # Limit the number of concurrent processes to 4
#         if (( $(jobs -r -p | wc -l) >= 4 )); then
#             wait -n
#         fi
#     done
# done
# wait

# Loop through each item in the list
for corr in "${corruptions[@]}"; do
    # Loop from 1 to 10
    for i in {1..10}; do
        python train.py --batch_size=32 --lr=1e-3 --corr=$corr --seed=$i --finetune_config=last_bn &
        # Limit the number of concurrent processes to 4
        if (( $(jobs -r -p | wc -l) >= 4 )); then
            wait -n
        fi
    done
done
wait