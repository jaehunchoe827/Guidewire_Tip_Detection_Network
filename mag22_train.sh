#!/bin/bash

echo "Training ver3_default.yaml"
python3 -m engine.main --train --config ver3_default.yaml 
sleep 3

echo "Training ver3_default_unfreeze_3.yaml"
python3 -m engine.main --train --config ver3_default_unfreeze_3.yaml 
sleep 3

echo "Training ver3_default_more_crop.yaml"
python3 -m engine.main --train --config ver3_default_more_crop.yaml 
sleep 3

echo "Training ver3_default_unfreeze_3_warmup_1.yaml"
python3 -m engine.main --train --config ver3_default_unfreeze_3_warmup_1.yaml 
sleep 3

echo "All models trained successfully!"