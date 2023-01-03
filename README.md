# ST-DASegNet

This repo is the implementation of "Self-Training Guided Disentangled Adaptation for Cross-Domain Remote Sensing Image Semantic Segmentation". we refer to  [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMGeneration](https://github.com/open-mmlab/mmgeneration). Many thanks to SenseTime and their two excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig1.png" width = "100%" alt="MMOTU"/></td>
    <td><img src="PaperFigs\Fig4.png" width = "100%" alt="DS2Net"/></td>
    </tr>
</table>

## Dataset Preparation

**Potsdam**

**Vaihingen**

**LoveDA**

## ST-DASegNet

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.4
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html); Please don't forget to install mmsegmentation with

     ```
     cd MMOTU_DS2Net
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training

**mit_b5.pth** : [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing) (Before training Segformer or DS<sup>2</sup>Net_T, loading ImageNet-pretrained mit_b5.pth is very useful. We provide this pretrained backbone here. The pretrained backbone has already been transformed to fit for our repo.)

#### Task1: Single-modality semantic segmentation

<table>
    <tr>
    <td><img src="PaperFigs\SSeg.jpg" width = "100%" alt="Single-Modality semantic segmentation"/></td>
    </tr>
</table>
  
     cd MMOTU_DS2Net
     
     ./tools/dist_train.sh ./experiments/pspnet_r50-d8_769x769_20k_MMOTU/config/pspnet_r50-d8_769x769_20k_MMOTU.py 2

#### Task2: UDA semantic segmentation

<table>
    <tr>
    <td><img src="PaperFigs\UDASeg.jpg" width = "100%" alt="UDA Multi-Modality semantic segmentation"/></td>
    </tr>
</table>

     cd MMOTU_DS2Net
     
     ./tools/dist_train.sh ./experiments/DS2Net_segformerb5_769x769_40k_MMOTU/config/DS2Net_segformerb5_769x769_40k_MMOTU.py 2

#### Task3: Single-modality recognition: 

<table>
    <tr>
    <td><img src="PaperFigs\SCls.jpg" width = "100%" alt="Single-Modality recognition"/></td>
    </tr>
</table>

### Testing

#### Task1: Single-modality semantic segmentation
  
     cd MMOTU_DS2Net
     
     ./tools/dist_test.sh ./experiments/pspnet_r50-d8_769x769_20k_MMOTU/config/pspnet_r50-d8_769x769_20k_MMOTU.py ./experiments/pspnet_r50-d8_769x769_20k_MMOTU/results/iter_80000.pth --eval mIoU

#### Task2: UDA semantic segmentation

     cd MMOTU_DS2Net
     
     ./tools/dist_test.sh ./experiments/DS2Net_segformerb5_769x769_40k_MMOTU/config/DS2Net_segformerb5_769x769_40k_MMOTU.py ./experiments/DS2Net_segformerb5_769x769_40k_MMOTU/results/iter_40000.pth --eval mIoU
     
## Description of MMOTU/DS<sup>2</sup>Net
- https://arxiv.org/abs/2207.06799 

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.
