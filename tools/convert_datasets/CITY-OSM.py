# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CITY-OSM dataset (Paris&Chicago) to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args

def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    # Original image of CITY-OSM dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    image = mmcv.imread(image_path)

    h, w, c = image.shape
    cs = args.clip_size
    ss = args.stride_size

    '''
    num_rows = math.ceil((h - cs) / ss) if math.ceil(
        (h - cs) / ss) * ss + cs >= h else math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) if math.ceil(
        (w - cs) / ss) * ss + cs >= w else math.ceil((w - cs) / ss) + 1
    '''

    ## modified by LYU: 2022/10/09
    num_rows = math.floor((h - cs) / ss) if math.ceil(
        (h - cs) / ss) * ss + cs >= h else math.ceil((h - cs) / ss) + 1
    num_cols = math.floor((w - cs) / ss) if math.ceil(
        (w - cs) / ss) * ss + cs >= w else math.ceil((w - cs) / ss) + 1
    
    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    '''
    xmin = x * cs
    ymin = y * cs
    '''
    ## modified by LYU: 2022/10/09
    xmin = x * ss
    ymin = y * ss

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + cs > w, w - xmin - cs, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + cs > h, h - ymin - cs, np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + cs, w),
        np.minimum(ymin + cs, h)
    ],
                     axis=1)

    if to_label:
        ## modified by LYU: 2023/11/7: Background: [255, 255, 255]; Road: [255, 0, 255], Building: [0, 0, 255]
        color_map = np.array([[255, 255, 255], [255, 0, 0], [0, 0, 255]])
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        if 'label' in image_path:
            image_name = osp.basename(image_path).split('/')[-1][:-11]
        else:
            image_name = osp.basename(image_path).split('/')[-1][:-10]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{image_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir

    print(dataset_path)
    print(out_dir)

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))

    src_path_list = glob.glob(os.path.join(dataset_path, '*.png'))
    prog_bar = mmcv.ProgressBar(len(src_path_list))
    for i, src_path in enumerate(src_path_list):
        if 'label' in src_path:
            dst_dir = osp.join(out_dir, 'ann_dir')
            clip_big_image(src_path, dst_dir, args, to_label=True)
        else:
            dst_dir = osp.join(out_dir, 'img_dir')
            clip_big_image(src_path, dst_dir, args, to_label=False)
        prog_bar.update()
    print('Done!')
    '''
    img = cv2.imread("./data/paris46_labels_1536_2560_2048_3072.png")
    PALETTE = [[255, 255, 255], [255, 0, 0], [0, 0, 255]]
    print(img.shape)
    print(np.unique(img))
    r = np.zeros((img.shape[0], img.shape[1], 1))
    g = np.zeros((img.shape[0], img.shape[1], 1))
    b = np.zeros((img.shape[0], img.shape[1], 1))
    for l in range(3):
        r[img[:, :, 0] == l] = PALETTE[l][0]
        g[img[:, :, 0] == l] = PALETTE[l][1]
        b[img[:, :, 0] == l] = PALETTE[l][2]
    
    gt_mask_tmp = np.zeros((img.shape[0], img.shape[1], 3))
    gt_mask_tmp[:, :, 0] = b.squeeze()
    gt_mask_tmp[:, :, 1] = g.squeeze()
    gt_mask_tmp[:, :, 2] = r.squeeze()
    cv2.imshow("1", gt_mask_tmp)
    cv2.waitKey(0)
    '''

if __name__ == '__main__':
    main()
