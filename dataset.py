import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import random
from skimage.transform import resize

from torchvision import transforms


## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts


        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        # lst_label = [f for f in lst_data if f.startswith('label')]
        # lst_input = [f for f in lst_data if f.startswith('input')]
        #
        # lst_label.sort()
        # lst_input.sort()
        #
        # self.lst_label = lst_label
        # self.lst_input = lst_input

        # lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):

        pick = random.randrange(1, 5)
        randomsize = random.randrange(1, 7)

        img_size = 256

        if pick <= 1:

            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.empty_circle_mask(img_size)
            mask = (1 - mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data

        elif pick <= 2:

            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.circle_mask(img_size)
            mask = (1 - mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data

        elif pick <= 3:

            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.random_empty_circle_mask(img_size, randomsize)
            mask = (1 - mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data


        else:
            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.make_mask(img_size)
            mask = (1-mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data


    def make_mask(self, length):
        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))
        numVertex = random.randint(7, 16)

        startPos=[]
        for i in range(numVertex):
            tmpY = random.randint(50, 200)
            tmpX = random.randint(50, 200)

            tmpList = [tmpY, tmpX]
            startPos.append(tmpList)

        for i in range(len(startPos) - 1):
            mask = cv2.line(mask, startPos[i], startPos[i + 1], (1, 1, 1), 18)

        return mask


    def empty_circle_mask(self, length):
        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))

        mask = cv2.circle(mask, (95, 130), 30, (1, 1, 1), 20)
        mask = cv2.circle(mask, (165, 130), 30, (1, 1, 1), 20)

        return mask

    def random_empty_circle_mask(self, length, random_size):

        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))

        mask = cv2.circle(mask, (95, 130), 30, (1, 1, 1), 3 * random_size)
        mask = cv2.circle(mask, (165, 130), 30, (1, 1, 1), 3 * random_size)

        return mask

    def circle_mask(self, length):
        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))

        mask = cv2.circle(mask, (95, 130), 30, (1, 1, 1), -1)
        mask = cv2.circle(mask, (165, 130), 30, (1, 1, 1), -1)

        return mask




class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mask_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.task = task
        self.opts = opts


        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        lst_mask = os.listdir(self.mask_dir)
        lst_mask = [f for f in lst_mask if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        # lst_label = [f for f in lst_data if f.startswith('label')]
        # lst_input = [f for f in lst_data if f.startswith('input')]
        #
        # lst_label.sort()
        # lst_input.sort()
        #
        # self.lst_label = lst_label
        # self.lst_input = lst_input

        # lst_data.sort()
        self.lst_data = lst_data
        self.lst_mask = lst_mask

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):

        # pick = random.randrange(2, 3)
        pick = 1
        random_size = random.randrange(1,2)

        img_size = 256

        if pick <= 1:

            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            mask = plt.imread(os.path.join(self.mask_dir, self.lst_mask[index]))

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask = np.expand_dims(mask,axis=0)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)
            mask = resize(mask, (img_size, img_size,1), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            # mask = self.empty_circle_mask(img_size)
            # mask = (1 - mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data

        elif pick <= 2:

            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.random_empty_circle_mask(img_size, random_size=random_size)
            mask = (1 - mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data

        elif pick <= 3:

            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.circle_mask(img_size)
            mask = (1 - mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data


        else:
            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.make_mask(img_size)
            mask = (1-mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data


    def make_mask(self, length):
        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))
        numVertex = random.randint(7, 16)

        startPos=[]
        for i in range(numVertex):
            tmpY = random.randint(50, 200)
            tmpX = random.randint(50, 200)

            tmpList = [tmpY, tmpX]
            startPos.append(tmpList)

        for i in range(len(startPos) - 1):
            mask = cv2.line(mask, startPos[i], startPos[i + 1], (1, 1, 1), 18)

        return mask


    def random_empty_circle_mask(self, length, random_size):

        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))

        mask = cv2.circle(mask, (95, 130), 30, (1, 1, 1), 10 * random_size)
        mask = cv2.circle(mask, (165, 130), 30, (1, 1, 1), 10 * random_size)

        return mask


    def empty_circle_mask(self, length):
        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))

        mask = cv2.circle(mask, (95, 130), 30, (1, 1, 1), 20)
        mask = cv2.circle(mask, (165, 130), 30, (1, 1, 1), 20)

        return mask


    def circle_mask(self, length):
        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))

        mask = cv2.circle(mask, (95, 130), 40, (1, 1, 1), -1)
        mask = cv2.circle(mask, (165, 130), 40, (1, 1, 1), -1)

        return mask



class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None, center_pos=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts

        self.center_pos = center_pos

        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        # lst_label = [f for f in lst_data if f.startswith('label')]
        # lst_input = [f for f in lst_data if f.startswith('input')]
        #
        # lst_label.sort()
        # lst_input.sort()
        #
        # self.lst_label = lst_label
        # self.lst_input = lst_input

        # lst_data.sort()
        self.lst_data = lst_data


    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):

        pick = 1

        img_size = 256

        if pick <= 1:

            img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sz = img.shape

            # img = img[self.center_pos[1] - 256 : self.center_pos[1] + 256, self.center_pos[0] - 256 : self.center_pos[0] + 256]
            img = resize(img, (img_size, img_size), anti_aliasing=True)

            # if sz[0] > sz[1]:
            #     img = img.transpose((1, 0, 2))

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.dtype == np.uint8:
                img = img / 255.0

            data = {'label': img}

            mask = self.empty_circle_mask(img_size)
            mask = (1 - mask)
            masked_img = img * mask
            data['masked_img'] = masked_img
            data['mask'] = mask

            if self.transform:
                data = self.transform(data)

            data = self.to_tensor(data)

            return data


    def empty_circle_mask(self, length):
        imgW = length
        imgH = length
        mask = np.zeros((imgH, imgW, 1))

        # mask = cv2.circle(mask, (75, 75), 40, (1, 1, 1), 18)
        # mask = cv2.circle(mask, (180, 75), 40, (1, 1, 1), 18)
        mask = cv2.circle(mask, (95, 125), 26, (1, 1, 1), 16)
        mask = cv2.circle(mask, (165, 125), 26, (1, 1, 1), 16)

        return mask


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)
        #
        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        # Updated at Apr 5 2020
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # input = (input - self.mean) / self.std
        # label = (label - self.mean) / self.std
        #
        # data = {'label': label, 'input': input}

        # Updated at Apr 5 2020
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomFlip(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            # label = np.fliplr(label)
            # input = np.fliplr(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            # label = np.flipud(label)
            # input = np.flipud(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        # data = {'label': label, 'input': input}

        return data


class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # input, label = data['input'], data['label']
    # h, w = input.shape[:2]

    keys = list(data.keys())

    h, w = data[keys[0]].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # input = input[id_y, id_x]
    # label = label[id_y, id_x]
    # data = {'label': label, 'input': input}

    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = value[id_y, id_x]

    return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1],
                                                    self.shape[2]))

        return data