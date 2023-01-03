#!/bin/bash

T=`date +%m%d%H%M`

export PYTHONPATH=$ROOT:$PYTHONPATH

python visgt_demo.py ../data/MMOTU/OTU_3d/images/90.JPG ../data/MMOTU/OTU_3d/annotations/90.PNG ../experiments/pspnet_r50-d8_769x769_20k_MMOTU/config/pspnet_r50-d8_769x769_20k_MMOTU.py ../experiments/pspnet_r50-d8_769x769_20k_MMOTU/results/iter_18000.pth --device cuda:0 --palette mmotu
