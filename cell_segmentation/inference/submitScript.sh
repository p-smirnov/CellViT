#! /bin/bash

source ~/.bashrc

mamba activate cellvit_env3

cd /home/smirnov/Github/CellViT/cell_segmentation/inference

python cell_detection.py --model /g/huber/users/smirnov/CellViT-SAM-H-x20.pth --magnification=20 process_wsi --wsi_path "/g/huber/users/smirnov/tcga_slides/GBM/ffpe/00e49ebc-adc3-4179-9a5b-ff97f20de671/TCGA-02-0440-01Z-00-DX1.4fef88c9-eff7-4e00-be19-d0db2871329a.svs" --patched_slide_path "/scratch/smirnov/tcga_cellvit/ffpe/GBM/TCGA-02-0440-01Z-00-DX1.4fef88c9-eff7-4e00-be19-d0db2871329a/"

