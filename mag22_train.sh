#!/bin/bash

echo "Training ver3_default.yaml"
python3 -m engine.main --train --config ver3_default.yaml 
sleep 3

echo "All models trained successfully!"
