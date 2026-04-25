#!/bin/bash

echo "Training ver3_default_unfreeze_4_warmup_2_5.yaml"
python3 -m engine.main --train --config ver3_default_unfreeze_4_warmup_2_5.yaml 
sleep 3

echo "Training ver3_default.yaml"
python3 -m engine.main --train --config ver3_default.yaml 
sleep 3

echo "All models trained successfully!"