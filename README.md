# ST-DASegNet

This repo is the implementation of "Self-Training Guided Disentangled Adaptation for Cross-Domain Remote Sensing Image Semantic Segmentation". we refer to  [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMGeneration](https://github.com/open-mmlab/mmgeneration). Many thanks to SenseTime and their two excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig1.png" width = "100%" alt="Cross-Domain RS Semantic Segmentation"/></td>
    <td><img src="PaperFigs\Fig4.png" width = "100%" alt="ST-DASegNet"/></td>
    </tr>
</table>

## Dataset Preparation

We select Postsdam, Vaihingen and LoveDA as benchmark datasets and create train, val, test list for researchers to follow. 

**In the following, we provide the detailed commands for dataset preparation.**

**Potsdam**
     
     Move the ‘3_Ortho_IRRG.zip’ and ‘5_Labels_all_noBoundary.zip’ to Potsdam_IRRG folder 
     Move the ‘2_Ortho_RGB.zip’ and ‘5_Labels_all_noBoundary.zip’ to Potsdam_RGB folder
     python tools/convert_datasets/potsdam.py yourpath/ST-DASegNet/data/Potsdam_IRRG/ --clip_size 512 --stride_size 512
     python tools/convert_datasets/potsdam.py yourpath/ST-DASegNet/data/Potsdam_RGB/ --clip_size 512 --stride_size 512

**Vaihingen**

     Move the 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' to Vaihingen_IRRG folder 
     python tools/convert_datasets/vaihingen.py yourpath/ST-DASegNet/data/Vaihingen_IRRG/ --clip_size 512 --stride_size 256

**LoveDA**
    
     Unzip Train.zip, Val.zip, Test.zip and create Train, Val and Test list for Urban and Rural

## ST-DASegNet

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.4
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd ST-DASegNet
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training

**mit_b5.pth** : [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing) For SegFormerb5 based ST-DASegNet training, we provide ImageNet-pretrained backbone here.

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
