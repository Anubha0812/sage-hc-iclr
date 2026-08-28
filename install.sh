#!/bin/bash -i

conda create -n sage-hc-iclr python=3.11
conda activate sage-hc-iclr

module load cuda/12.5 # If "module" is configured, can be ignored

pip install torch torchvision torchaudio

pip install -r requirements.txt
