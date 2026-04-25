# Guidewire detection network.
This repo is implemeted based on jahongir7174's [YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)

## TODO
consider using pre-normalization, and layer norm instead of batch norm

### Installation
1. create and activate conda env
conda env create --file conda_environment.yaml
2. After installation, you must add the following line to your ~/.bashrc
```
export CUBLAS_WORKSPACE_CONFIG=:16:8
```
3. please download pretrained yolo11 weights from
[yolo11m](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_m.pt)
[yolo11l](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_l.pt)


### Train
* **For reproducibility**, you must uncomment the below line in utils.util.py
    ```
    # torch.use_deterministic_algorithms(True)
    ```
```
python3 -m engine.main --train --config ver0_default.yaml
```

### Dataset structure
    ├── datasets
        ├── guidewire
            ├── 01_04
                ├── Images
                    ├── 1234.jpg
                    ├── 2222.jpg (not all images have corresponding labels.)
                    ├── 3333.jpg
                ├── Labels
                    ├── 1234.jpg
                    ├── 3333.jpg
            ├── 05_09
                ├── Image
                    ├── ...
                ├── Labels
                    ├── ...
            ├── ...

#### Reference
* https://github.com/jahongir7174/YOLOv11-pt
