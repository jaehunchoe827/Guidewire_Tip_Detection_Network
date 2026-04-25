#!/bin/bash

echo "Training ver3_default_epoch_45.yaml"
python3 -m engine.main --train --config ver3_default_epoch_45.yaml 
sleep 3

echo "All models trained successfully!"