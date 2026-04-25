#!/bin/bash

echo "Training ver3_default_hidden_128.yaml"
python3 -m engine.main --train --config ver3_default_hidden_128.yaml 
sleep 3

echo "Training ver3_default_high_weight_decay.yaml"
python3 -m engine.main --train --config ver3_default_high_weight_decay.yaml 
sleep 3

echo "All models trained successfully!"