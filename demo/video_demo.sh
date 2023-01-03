#!/bin/bash

T=`date +%m%d%H%M`
cfg='../configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
cpt='../checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img='./demo.png'
ouf='./results/'

export PYTHONPATH=$ROOT:$PYTHONPATH
            
python video_demo.py $img $cfg $cpt --output-file $ouf --show --show-wait-time 100 2>&1 | tee ./logs/demo.log
