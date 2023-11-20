# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import numpy as np
import cv2
import tifffile
import mmcv
import pandas as pd
import csv
#import gdal
#from osgeo import gdal

S2_ori_CLASSES = ('ignore', 'water', 'BareGround-Artificial', 'BareGround-Natural', 'Snow-Ice', 'Woody', 'Cultivated', 'Semi-Natural')
S2_ori_PALETTE = [[0, 0, 0], [0, 0, 255], [136, 136, 136], [209, 164, 109], [245, 245, 255], [214, 76, 43], [24, 104, 24], [0, 255, 0]]

TRAIN_tiles = ['35JNN', '35JQG', '35KKP']
VAL_tiles = ['35LMC', '35MNT']
TRAINVAL_chips = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
PLATFORM = ['S2'] 

def parse_args():
    parser = argparse.ArgumentParser(
        description='Sentinel-2 dataset preprocessing')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args

def read_label(url, out_dir, img_prefix):
    #print(url)
    #print(out_dir)
    #print(img_prefix)
    label = tifffile.imread(url)
    def indexmap2colormap(indexmap):
        r = np.zeros((indexmap.shape[0], indexmap.shape[1], 1))
        g = np.zeros((indexmap.shape[0], indexmap.shape[1], 1))
        b = np.zeros((indexmap.shape[0], indexmap.shape[1], 1))
        for l in range(len(S2_ori_PALETTE)):
            r[indexmap == l] = S2_ori_PALETTE[l][0]
            g[indexmap == l] = S2_ori_PALETTE[l][1]
            b[indexmap == l] = S2_ori_PALETTE[l][2]
        colormap_tmp = np.zeros((indexmap.shape[0], indexmap.shape[1], 3))
        colormap_tmp[:, :, 0] = b.squeeze()
        colormap_tmp[:, :, 1] = g.squeeze()
        colormap_tmp[:, :, 2] = r.squeeze()
        colormap_tmp = colormap_tmp.astype(np.uint8)
        #cv2.imshow("1", colormap_tmp.astype(np.uint8))
        #cv2.waitKey(0)
        return colormap_tmp
    label = label.astype(np.uint8)
    colormap_tmp = indexmap2colormap(label[:, :, 0])
    indexmap_path = os.path.join(out_dir, img_prefix + "_RGB" + '.png')
    cv2.imwrite(indexmap_path, label[:, :, 0])
    colormap_path = os.path.join(out_dir, img_prefix + "_RGB_colormap" + '.png')
    cv2.imwrite(colormap_path, colormap_tmp)

def read_image(url, out_dir, img_prefix):
    #print(url)
    #print(out_dir)
    #print(img_prefix)
    img_B2_url = os.path.join(url, img_prefix + '_B02_10m.tif')
    img_B3_url = os.path.join(url, img_prefix + '_B03_10m.tif')
    img_B4_url = os.path.join(url, img_prefix + '_B04_10m.tif')
    img_B8_url = os.path.join(url, img_prefix + '_B08_10m.tif')
    img_B2 = tifffile.imread(img_B2_url)
    img_B3 = tifffile.imread(img_B3_url)
    img_B4 = tifffile.imread(img_B4_url)
    img_B8 = tifffile.imread(img_B8_url)
    ## 16bit -> 8bit
    img_B2_8bit = (img_B2 - np.min(img_B2)) / (np.max(img_B2) - np.min(img_B2))
    img_B3_8bit = (img_B3 - np.min(img_B3)) / (np.max(img_B3) - np.min(img_B3))
    img_B4_8bit = (img_B4 - np.min(img_B4)) / (np.max(img_B4) - np.min(img_B4))
    img_B8_8bit = (img_B8 - np.min(img_B8)) / (np.max(img_B8) - np.min(img_B8))
    ## RGB image concatenation
    RGB_tmp = np.zeros((img_B2.shape[0], img_B2.shape[1], 3))
    RGB_tmp[:, :, 0] = img_B2_8bit
    RGB_tmp[:, :, 1] = img_B3_8bit
    RGB_tmp[:, :, 2] = img_B4_8bit
    #RGB_tmp[:, :, 3] = img_B8_8bit
    RGB_tmp = RGB_tmp * 255
    RGB_tmp = RGB_tmp.astype(np.uint8)
    out_path = os.path.join(out_dir, img_prefix + "_RGB" + '.png')
    #print(out_path)
    cv2.imwrite(out_path, RGB_tmp)

def read_csv(csv_path):
    selected_img_with_date = []
    with open(csv_path, "r", encoding="gbk") as csvfile:
        csvreader = csv.reader(csvfile)
        #csvreader = pd.read_csv(csvfile)
        for row in csvreader:
            if row[0] != '':
                selected_img_with_date.append(row[1])
    return selected_img_with_date
                

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir
    print(dataset_path)
    print(out_dir)
    
    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'train', 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'train', 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'val', 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'val', 'ann_dir'))
    out_img_dir_train = osp.join(out_dir, 'train', 'img_dir')
    out_ann_dir_train = osp.join(out_dir, 'train', 'ann_dir')
    out_img_dir_val = osp.join(out_dir, 'val', 'img_dir')
    out_ann_dir_val = osp.join(out_dir, 'val', 'ann_dir')

    ## Training & Val
    for tile_id, tile_name in enumerate(VAL_tiles):
        print("********TILE processing: ********", tile_name)
        for chip_id, chip_name in enumerate(TRAINVAL_chips):
            print("####CHIP processing: ####", chip_name)
            data_prefix = tile_name + '_' + chip_name + '_'
            for pl_id, pl_name in enumerate(PLATFORM):
                print("--PLATFORM SELECTION--: ", pl_name)
                data_path = os.path.join(dataset_path, tile_name, chip_name, pl_name)
                ## all folder list generation
                for root, dirs, files in os.walk(data_path):
                    if len(dirs) > 0:
                        for k in range(len(files)):
                            if 'tif' in files[k]:
                                lbl_filename = files[k]
                            if 'csv' in files[k]:
                                csv_filename = files[k]
                        csv_path = os.path.join(data_path, csv_filename)
                        img_selected_list = read_csv(csv_path)
                        for j in range(len(dirs)):
                            ## YYYYMMDD date formate
                            ## filter dataset with csv file
                            if dirs[j][-8:] in img_selected_list:
                                ## (1) image generation
                                img_path = os.path.join(data_path, dirs[j])
                                read_image(img_path, out_img_dir_val, dirs[j])
                                ## (2) label generation
                                lbl_path = os.path.join(data_path, lbl_filename)
                                read_label(lbl_path, out_ann_dir_val, dirs[j])
                
    '''
    ## 1. read label
    label_url = os.path.join(dataset_path, '35JNN_00_2018_LC_10m.tif')
    label = read_label(label_url, out_dir)
    '''
    '''
    ## 2. read image and select bands: 
    ## 2.1 scheme1: R-G-B -> B4-B3-B2
    ## 2.2 scheme2: R-G-B-Nir -> B4-B3-B2-B8
    image_url = os.path.join(dataset_path, '35JNN_00_20181121')
    image = read_image(image_url, out_dir)
    '''
    
    
if __name__ == '__main__':
    main()
