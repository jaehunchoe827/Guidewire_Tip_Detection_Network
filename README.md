# Guidewire Tip Detection Network.

> This implementation is based on jahongir7174's [YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)

## Installation

This repo was tested on Ubuntu 22.04.5 (test environment: i9-14900K + RTX 5090 + 64GB DDR5 RAM).

1. Create and activate the conda environment:
   ```bash
   conda env create --file conda_environment.yaml
   conda activate gwtd
   ```
2. After installation, add the following line to your `~/.bashrc` (required for
   deterministic CUDA behavior):
   ```bash
   export CUBLAS_WORKSPACE_CONFIG=:16:8
   ```

## Quick Start

1. Download the pre-trained parameters and dataset:
   ```bash
   curl -L "https://www.dropbox.com/scl/fi/hakd3tw5znmxiqr4m6mww/magposenet_pretrained.tar?rlkey=5bbnto6ezzvjssp67h6idjqsr&st=7zrko0ey&dl=1" -o gwtd_pretrained.tar
   ```
2. Extract the archive:
   ```bash
   tar --strip-components=1 -xvf gwtd_pretrained.tar
   ```
3. Run the test:
   ```bash
   python3 -m engine.test --config default.yaml 
   ```

## Reproduce the Result

You can reproduce the results by training the networks yourself
```bash
python3 -m engine.main --train --config default.yaml 
```

### Recommended Environment Setup (disable CPU sleep)

For reproducing the results, the following setup is recommended for faster and more
consistent training:

1. Open the GRUB config:
   ```bash
   sudo nano /etc/default/grub
   ```
2. Find the line `GRUB_CMDLINE_LINUX_DEFAULT`. The default is likely:
   ```bash
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
   ```
   Update it to:
   ```bash
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_aspm=off intel_idle.max_cstate=1"
   ```
   - `pcie_aspm=off` disables PCIe power saving.
   - `intel_idle.max_cstate=1` limits the CPU to idle states 0 and 1 (disabling deeper
     sleep states).
3. Apply the changes and reboot:
   ```bash
   sudo update-grub
   sudo reboot
   ```
4. After reboot, verify that only two idle states are displayed:
   ```bash
   cpupower idle-info
   ```
