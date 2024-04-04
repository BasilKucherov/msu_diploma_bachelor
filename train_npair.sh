#!/bin/bash

echo "Run NPairLoss_SC_25CS_1250SPC"
python3 train.py --config ./train_configs/NPairLoss_SC_25CS_1250SPC.json
echo "Finished NPairLoss_SC_25CS_1250SPC"

echo "Run NPairLoss_MSWC_RU_50CS_500SPC"
python3 train.py --config ./train_configs/NPairLoss_MSWC_RU_50CS_500SPC.json
echo "Finished NPairLoss_MSWC_RU_50CS_500SPC"

echo "Run NPairLoss_MSWC_EN_100CS_500SPC"
python3 train.py --config ./train_configs/NPairLoss_MSWC_EN_100CS_500SPC.json
echo "Finished NPairLoss_MSWC_EN_100CS_500SPC"

echo "Run NPairLoss_MSWC_EN_500CS_100SPC"
python3 train.py --config ./train_configs/NPairLoss_MSWC_EN_500CS_100SPC.json
echo "Finished NPairLoss_MSWC_EN_500CS_100SPC"
