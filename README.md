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

We select deeplabv3 and Segformerb5 as baselines. Actually, we use deeplabv3+, which is a more advanced version of deeplabv3. After evaluating, we find that deeplabv3+ has little modification compared to deeplabv3 and has little advantage than deeplabv3.

For LoveDA results, we evaluate on test datasets and submit to online server (https://github.com/Junjue-Wang/LoveDA) (https://codalab.lisn.upsaclay.fr/competitions/424). We also provide the evaluation results on validation dataset.

1. Potsdam IRRG to Vaihingen IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2
     ```

2. Vaihingen IRRG to Potsdam IRRG:

    ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2
     ```

3. Potsdam RGB to Vaihingen IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_PotsdamRGB2Vaihingen.py 2
     ```
     
4. Vaihingen RGB to Potsdam IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2PotsdamRGB.py 2
     ```

5. LoveDA Rural to Urban

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2U.py 2
     ```

6. LoveDA Urban to Rural

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_U2R.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_U2R.py 2
     ```

### Testing
  
Trained with the above commands, you can get a trained model to test the performance of your model.   

1. Testing commands

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mIoU   
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mFscore 
     ```

2. Testing cases

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mFscore 
     ```
     
     ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mFscore 
     ```

The ArXiv version of this paper will be release soon!

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.
