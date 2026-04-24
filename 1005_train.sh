#!/bin/bash

echo "Training ver3_default_more_crop.yaml"
python3 -m engine.main --train --config ver3_default_more_crop.yaml 
sleep 3

echo "All models trained successfully!"
