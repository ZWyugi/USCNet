# -*- coding: utf-8 -*-
# Time    : 2023/10/30 16:03
# Author  : fanc
# File    : load_data.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
from skimage.transform import resize
import SimpleITK as sitk
from scipy.ndimage import zoom
from collections import defaultdict
from skimage.morphology import dilation, ball, closing
from scipy.ndimage import gaussian_filter
import pandas as pd

def split_data(data_dir, infos_name, filter_volume, rate=0.8):
    with open(os.path.join(data_dir, infos_name), 'r', encoding='utf-8') as f:
        infos = json.load(f)

    infos = list(filter(lambda x: x['volume'] >= filter_volume, infos))
    # 创建一个字典，用于按类别存储数据
    class_data = defaultdict(list)
    for info in infos:
        label = info['label']  # 假设数据集中每个样本都有'label'字段表示类别
        class_data[label].append(info)

    train_infos = []
    test_infos = []

    # 对每个类别进行分层抽样
    for label, data in class_data.items():
        random.seed(1900)
        random.shuffle(data)
        num_samples = len(data)
        train_num = int(rate * num_samples)
        train_infos.extend(data[:train_num])
        test_infos.extend(data[train_num:])

    return train_infos, test_infos

# class MyDataset(Dataset):
#     def __init__(self, data_dir, infos, input_size, phase='train', task=[0, 1]):
#         '''
#         task: 0 :seg,  1 :cla
#         '''
#         task = [int(i) for i in re.findall('\d', str(task))]
#         img_dir = os.path.join(data_dir, 'imgs_nii')
#         mask_dir = os.path.join(data_dir, 'mask_nii')
#         self.seg = False
#         self.cla = False
#         if 0 in task:
#             self.seg = True
#         if 1 in task:
#             self.cla = True
#
#         self.input_size = tuple([int(i) for i in re.findall('\d+', str(input_size))])
#         self.img_dir = img_dir
#         if self.cla:
#             self.labels = [i['label'] for i in infos]
#         if self.seg:
#             self.mask_dir = mask_dir
#         # self.labels = [[1, 0] if int(i['label']) == 1 else [0, 1] for i in infos]
#
#         self.ids = [i['id'] for i in infos]
#         self.phase = phase
#         self.labels = torch.tensor(self.labels, dtype=torch.float)
#
#     def __len__(self):
#         return len(self.ids)
#
#     def __getitem__(self, i):
#         img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
#         if self.seg:
#             mask = sitk.ReadImage(os.path.join(self.mask_dir, f"{self.ids[i]}-mask.nii.gz"))
#         else:
#             mask = None
#         if self.phase == 'train':
#             img, mask = self.train_preprocess(img, mask)
#         else:
#             img, mask = self.val_preprocess(img, mask)
#         if self.cla:
#             label = self.labels[i]
#         else:
#             label = 1
#
#         img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
#         if self.seg:
#             mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
#         if self.cla:
#             label = self.labels[i].unsqueeze(0)
#
#         return img, mask, label
#
#     def train_preprocess(self, img, mask):
#         img, mask = self.resample(itkimage=img, itkmask=mask)
#         # mask = self.resample(mask)
#         # print(img.shape, mask.shape)
#         if self.seg:
#             assert img.shape == mask.shape, "img and mask shape not match"
#             # img, mask = self.crop(img, mask)
#         img = self.normalize(img)
#         img, mask = self.resize(img, mask)
#
#         return img, mask
#     def val_preprocess(self, img, mask):
#         img, mask = self.resample(img, mask)
#         # mask = self.resample(mask)
#         if self.seg:
#             assert img.shape == mask.shape, "img and mask shape not match"
#         # img, mask = self.crop(img, mask)
#         img = self.normalize(img)
#         img, mask = self.resize(img, mask)
#
#         return img, mask
#
#     def crop(self, img, mask):
#         crop_img = img
#         crop_mask = mask
#         # amos kidney mask
#         crop_mask[crop_mask == 2] = 1
#         crop_mask[crop_mask != 1] = 0
#         target = np.where(crop_mask == 1)
#         [d, h, w] = crop_img.shape
#         [max_d, max_h, max_w] = np.max(np.array(target), axis=1)
#         [min_d, min_h, min_w] = np.min(np.array(target), axis=1)
#         [target_d, target_h, target_w] = np.array([max_d, max_h, max_w]) - np.array([min_d, min_h, min_w])
#         z_min = int((min_d - target_d / 2) * random.random())
#         y_min = int((min_h - target_h / 2) * random.random())
#         x_min = int((min_w - target_w / 2) * random.random())
#
#         z_max = int(d - ((d - (max_d + target_d / 2)) * random.random()))
#         y_max = int(h - ((h - (max_h + target_h / 2)) * random.random()))
#         x_max = int(w - ((w - (max_w + target_w / 2)) * random.random()))
#
#         z_min = np.max([0, z_min])
#         y_min = np.max([0, y_min])
#         x_min = np.max([0, x_min])
#
#         z_max = np.min([d, z_max])
#         y_max = np.min([h, y_max])
#         x_max = np.min([w, x_max])
#
#         z_min = int(z_min)
#         y_min = int(y_min)
#         x_min = int(x_min)
#
#         z_max = int(z_max)
#         y_max = int(y_max)
#         x_max = int(x_max)
#         crop_img = crop_img[z_min: z_max, y_min: y_max, x_min: x_max]
#         crop_mask = crop_mask[z_min: z_max, y_min: y_max, x_min: x_max]
#
#         return crop_img, crop_mask
#
#     def resample(self, itkimage, itkmask, new_spacing=[1, 1, 1]):
#         # spacing = itkimage.GetSpacing()
#         img = sitk.GetArrayFromImage(itkimage)
#         if self.seg:
#             mask = sitk.GetArrayFromImage(itkmask)
#         else:
#             mask = None
#         # # MASK 膨胀腐蚀操作
#         # kernel = ball(5)  # 3D球形核
#         # # 应用3D膨胀
#         # dilated_mask = dilation(mask, kernel)
#         # mask = closing(dilated_mask, kernel)
#         # resize_factor = spacing / np.array(new_spacing)
#         # resample_img = zoom(img, resize_factor, order=0)
#         # resample_mask = zoom(mask, resize_factor, order=0, mode='nearest')
#         return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)
#
#     def normalize(self, img):
#
#         # CT值范围选取
#         min = 0
#         max = 2000
#         img[img < min] = min
#         img[img > max] = max
#
#         # std = np.std(img)
#         # avg = np.average(img)
#         # return (img - avg + std) / (std * 2)
#         return (img - min) / (max - min)
#
#     def resize(self, img, mask):
#         # img = np.transpose(img, (2, 1, 0))
#         # mask = np.transpose(mask, (2, 1, 0))
#         rate = np.array(self.input_size) / np.array(img.shape)
#         try:
#             img = zoom(img, rate.tolist(), order=0)
#             if self.seg:
#                 mask = zoom(mask, rate.tolist(), order=0, mode='nearest')
#         except Exception as e:
#             print(e)
#             img = resize(img, self.input_size)
#             if self.seg:
#                 mask = resize(mask, self.input_size, order=0)
#         # # MASK 膨胀腐蚀操作
#         # kernel = ball(5)  # 3D球形核
#         # # 应用3D膨胀
#         # dilated_mask = dilation(mask, kernel)
#         # mask = closing(dilated_mask, kernel)
#
#         # 高斯滤波去噪
#         # img = gaussian_filter(img, sigma=1)
#         # 中值滤波去噪
#         # from scipy.ndimage import median_filter
#         # img = median_filter(img, size=3)
#
#         return img, mask

class MyDataset(Dataset):
    def __init__(self, data_dir, infos, phase='train'):
        with open('configs/dataset.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        # img_dir = os.path.join(data_dir, 'cropped_img')
        # mask_dir = os.path.join(data_dir, 'cropped_mask')
        img_dir = config['img_dir']
        mask_dir = config['mask_dir']
        data_dir = config['data_dir']
        clinical_dir = config['clinical_dir']
        self.use_clinical = False
        if clinical_dir:
            self.use_clinical = True
            self.clinical = pd.read_excel(os.path.join(data_dir, clinical_dir))
        self.img_dir = os.path.join(data_dir, img_dir)
        self.labels = [i['label'] for i in infos]
        self.mask_dir = os.path.join(data_dir, mask_dir)
        self.ids = [i['sid'] for i in infos]
        self.pids = [i['pid'] for i in infos]
        self.phase = phase
        self.labels = torch.tensor(self.labels, dtype=torch.float)


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
        mask = sitk.ReadImage(os.path.join(self.mask_dir, f"{self.ids[i]}.nii.gz"))
        if self.phase == 'train':
            img, mask = self.train_preprocess(img, mask)
        else:
            img, mask = self.val_preprocess(img, mask)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
        label = self.labels[i].unsqueeze(0)

        if self.use_clinical:
            pid = self.pids[i]
            # print(pid)
            clinical = self.clinical[self.clinical['pid']==pid].fillna(0)
            # print(clinical)
            clinical = torch.tensor(np.array(clinical.values[0][1:], dtype=np.float32), dtype=torch.float32)
        else:
            clinical = 0

        return img, mask, label, clinical

    def train_preprocess(self, img, mask):
        img, mask = self.resample(itkimage=img, itkmask=mask)
        img = self.normalize(img)
        return img, mask
    def val_preprocess(self, img, mask):
        img, mask = self.resample(img, mask)
        img = self.normalize(img)
        return img, mask


    def resample(self, itkimage, itkmask):
        # spacing = itkimage.GetSpacing()
        img = sitk.GetArrayFromImage(itkimage)
        mask = sitk.GetArrayFromImage(itkmask)


        return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)

    def normalize(self, img):
        # CT值范围选取
        min = -400
        max = 2000
        img[img < min] = min
        img[img > max] = max

        return (img - min) / (max - min)


def my_dataloader(data_dir, infos, batch_size=1, shuffle=True, num_workers=0):
    dataset = MyDataset(data_dir, infos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':

    data_dir = r'C:\Users\Asus\Desktop\KidneyStone\data'
    train_info, test_info = split_data(data_dir, rate=0.8)
    train_dataloader = my_dataloader(data_dir, train_info, batch_size=1)
    test_dataloader = my_dataloader(data_dir, test_info, batch_size=1)
    for i, (image, mask, label) in enumerate(train_dataloader):
        print(image, mask.max(), label.shape, label[:, 0].shape)

        new_image = sitk.GetImageFromArray(image.numpy()[0][0])
        new_image.SetSpacing([1, 1, 1])
        sitk.WriteImage(new_image, os.path.join(data_dir, f'{i}.nii.gz'))

        new_mask = sitk.GetImageFromArray(mask.numpy()[0][0])
        new_mask.SetSpacing([1, 1, 1])
        sitk.WriteImage(new_mask, os.path.join(data_dir, f'{i}-mask.nii.gz'))
        break

        # nifti_image = nib.Nifti1Image(image.numpy()[0][0], affine=None)
        # nib.save(nifti_image, os.path.join(data_dir, f'process_img_{i}.nii.gz'))
        # nifti_image = nib.Nifti1Image(mask.numpy()[0][0], affine=None)
        # nib.save(nifti_image, os.path.join(data_dir, f'process_mask_{i}.nii.gz'))
    #

    # for i, (image, mask, label) in enumerate(test_dataloader):
    #     print(i,  image.shape, mask.shape, label)
