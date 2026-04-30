#!/bin/bash

echo "Training default_epochs_75.yaml"
python3 -m engine.main --train --config default_epochs_75.yaml 
sleep 3

echo "All models trained successfully!"