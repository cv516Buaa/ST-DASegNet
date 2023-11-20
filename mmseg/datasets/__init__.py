# Copyright (c) OpenMMLab. All rights reserved.
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                               RepeatDataset)
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .isprs import ISPRSDataset
from .loveda import LoveDADataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .potsdam import PotsdamDataset
from .stare import STAREDataset
from .voc import PascalVOCDataset

## added by LYU: 2022/10/27
from .pv_forAdap import PVDataset_forAdap
from .LoveDA_forAdap import LoveDADataset_forAdap

## added by LYU: 2023/11/7
from .PC_forAdap import PCDataset_forAdap
## added by LYU: 2023/11/13
from .GR_forAdap import GRDataset_forAdap
## added by LYU: 2023/11/18
from .SR_forAdap import SRDataset_forAdap

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
    'ISPRSDataset', 'PotsdamDataset', 'PVDataset_forAdap', 'LoveDA_forAdap', 
    'PCDataset_forAdap', 'GRDataset_forAdap', 'SRDataset_forAdap'
]
