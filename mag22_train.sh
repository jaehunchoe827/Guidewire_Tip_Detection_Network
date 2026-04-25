#!/bin/bash

echo "Training ver3_default_ema_long_soft.yaml"
python3 -m engine.main --train --config ver3_default_ema_long_soft.yaml 
sleep 3

echo "All models trained successfully!"