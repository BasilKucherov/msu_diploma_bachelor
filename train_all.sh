#!/bin/bash

echo "Run CrossEntropyLoss_MSWC_RU_50CS_500SPC"
python3 train.py --config ./train_configs/CrossEntropyLoss_MSWC_RU_50CS_500SPC.json
echo "Finished CrossEntropyLoss_MSWC_RU_50CS_500SPC"

echo "Run CrossEntropyLoss_MSWC_EN_100CS_500SPC"
python3 train.py --config ./train_configs/CrossEntropyLoss_MSWC_EN_100CS_500SPC.json
echo "Finished CrossEntropyLoss_MSWC_EN_100CS_500SPC"

echo "Run CrossEntropyLoss_MSWC_EN_500CS_100SPC"
python3 train.py --config ./train_configs/CrossEntropyLoss_MSWC_EN_500CS_100SPC.json
echo "Finished CrossEntropyLoss_MSWC_EN_500CS_100SPC"

