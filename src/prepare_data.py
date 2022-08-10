import os
import cv2
import random


dir = '../data/'
rgb_files = sorted(os.listdir(os.path.join(dir, 'rgb')))
rgb_files = ['../data/original_data/0001_rgb.jpg', '../data/original_data/0002_rgb.jpg', '../data/original_data/0003_rgb.jpg']
nir_files = sorted(os.listdir(os.path.join(dir, 'nir')))
nir_files = ['../data/original_data/0001_nir.jpg', '../data/original_data/0002_nir.jpg', '../data/original_data/0003_nir.jpg']
gt_files = sorted(os.listdir(os.path.join(dir, 'gt')))
gt_files = ['../data/original_data/0001_gt.jpg', '../data/original_data/0002_gt.jpg', '../data/original_data/0003_gt.jpg']

def random_crop(rgb, nir, gt):
    a = 1692 - 1520
    b = 1001 - 950
    x, y = random.randint(0, a), random.randint(0, b)
    crop_rgb = rgb[x:x+1520, y:y+950, :]
    crop_nir = nir[x:x+1520, y:y+950]
    crop_gt = gt[x:x+1520, y:y+950, :]
    return crop_rgb, crop_nir, crop_gt

def random_flip(rgb, nir, gt):
    flip_flag = random.randint(0, 2) - 1
    flip_rgb = cv2.flip(rgb, flip_flag)
    flip_nir = cv2.flip(nir, flip_flag)
    flip_gt = cv2.flip(gt, flip_flag)
    return flip_rgb, flip_nir, flip_gt

def random_rotate(rgb, nir, gt):
    rotate_int = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    rotate_flag = random.randint(0, 2)
    rotate_rgb = cv2.rotate(rgb, rotate_int[rotate_flag])
    rotate_nir = cv2.rotate(nir, rotate_int[rotate_flag])
    rotate_gt = cv2.rotate(gt, rotate_int[rotate_flag])
    return rotate_rgb, rotate_nir, rotate_gt

def data_enhance():
    index = 60
    for i in range(3):
        rgb_file, nir_file, gt_file = rgb_files[i], nir_files[i], gt_files[i]
        for j in range(5):
            rgb, nir, gt = cv2.imread(rgb_file), cv2.imread(nir_file), cv2.imread(gt_file)
            rgb, nir, gt = random_crop(rgb, nir, gt)
            rgb, nir, gt = random_flip(rgb, nir, gt)
            rgb, nir, gt = random_rotate(rgb, nir, gt)
            cv2.imwrite('../data/rgb/' + str(index) + '_rgb.jpg', rgb)
            cv2.imwrite('../data/nir/' + str(index) + '_nir.jpg', nir)
            cv2.imwrite('../data/gt/' + str(index) + '_gt.jpg', gt)
            index += 1


data_enhance()