#!/bin/bash

echo "Run LiftedStructuredLoss_SC_25CS_1250SPC"
python3 train.py --config ./train_configs/LiftedStructuredLoss_SC_25CS_1250SPC.json
echo "Finished LiftedStructuredLoss_SC_25CS_1250SPC"

echo "Run LiftedStructuredLoss_MSWC_RU_50CS_500SPC"
python3 train.py --config ./train_configs/LiftedStructuredLoss_MSWC_RU_50CS_500SPC.json
echo "Finished LiftedStructuredLoss_MSWC_RU_50CS_500SPC"

echo "Run LiftedStructuredLoss_MSWC_EN_100CS_500SPC"
python3 train.py --config ./train_configs/LiftedStructuredLoss_MSWC_EN_100CS_500SPC.json
echo "Finished LiftedStructuredLoss_MSWC_EN_100CS_500SPC"

echo "Run LiftedStructuredLoss_MSWC_EN_500CS_100SPC"
python3 train.py --config ./train_configs/LiftedStructuredLoss_MSWC_EN_500CS_100SPC.json
echo "Finished LiftedStructuredLoss_MSWC_EN_500CS_100SPC"
