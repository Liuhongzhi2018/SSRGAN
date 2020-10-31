import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
from PIL import Image
import torchvision.transforms as transforms
# import torch.nn.functional as F


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def norm(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class HyperDatasetValid(udata.Dataset):
    def __init__(self, mode='valid', opt=None):
        if mode != 'valid':
            raise Exception("Invalid mode!", mode)
        data_path = './Dataset/Valid'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        self.keys.sort()
        self.opt = opt

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        # hyper = np.float32(np.array(mat['rad']))
        hyper = np.transpose(hyper, [2, 1, 0])
        transform_hyper = self.get_transform(cur=1, opt=self.opt)
        hyper = transform_hyper(hyper)

        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        transform_rgb = self.get_transform(cur=2, opt=self.opt)
        rgb = transform_rgb(rgb)

        mat.close()
        return rgb, hyper

    def get_transform(self, cur, opt, params=None, method=Image.BICUBIC, normalize=True):
        transform_list = []
        transform_list += [transforms.ToTensor()]

        # if normalize:
        #     mt = [0.5] * opt.output_nc if cur == 1 else [0.5] * opt.input_nc
        #     mtp = tuple(mt)
        #     # print("mtp: ", mtp)
        #     transform_list += [transforms.Normalize(mtp, mtp)]

        return transforms.Compose(transform_list)


class HyperDatasetTrain(udata.Dataset):
    def __init__(self, mode='train', opt=None):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        # data_path = './Dataset/Train1'
        data_path = './Dataset/Train'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()
        self.opt = opt

    def __len__(self):
        # print("length")
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        # hyper = np.float32(np.array(mat['rad']))
        hyper = np.float32(np.array(mat['cube']))  # (482, 512, 31)  CWH
        hyper = np.transpose(hyper, [2, 1, 0])
        # print("hyper shape: {}".format(hyper.shape))
        # print("hyper max {} hyper min {}".format(hyper.max(), hyper.min()))
        # hyper max 1.0 hyper min 0.004312408156692982
        transform_hyper = self.get_transform(cur=1, opt=self.opt)
        hyper = transform_hyper(hyper)

        rgb = np.float32(np.array(mat['rgb']))     # (482, 512, 3)   CWH
        rgb = np.transpose(rgb, [2, 1, 0])
        # print("RGB shape: {}".format(rgb.shape))
        # print("rgb max {} rgb min {}".format(rgb.max(), rgb.min()))
        # rgb max 227.0 rgb min 1.0
        transform_rgb = self.get_transform(cur=2, opt=self.opt)
        rgb = transform_rgb(rgb)

        mat.close()
        # print("RGB shape: {} hyper shape: {}".format(rgb.shape, hyper.shape))
        # RGB shape: torch.Size([3, 482, 512]) hyper shape: torch.Size([31, 482, 512])
        return rgb, hyper

    def get_transform(self, cur, opt, params=None, method=Image.BICUBIC, normalize=True):
        transform_list = []
        # if 'resize' in opt.resize_or_crop:
        #     osize = [opt.loadSize, opt.loadSize]
        #     transform_list.append(transforms.Scale(osize, method))

        # elif 'scale_width' in opt.resize_or_crop:
        #     transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

        # if 'crop' in opt.resize_or_crop:
        #     transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

        # if opt.resize_or_crop == 'none':
        #     base = float(2 ** opt.n_downsample_global)
        #     if opt.netG == 'local':
        #         base *= (2 ** opt.n_local_enhancers)
        #     transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

        # if opt.isTrain and not opt.no_flip:
        #     transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]

        # if normalize:
        #     mt = []
        #     if cur == 1:
        #         mt = [0.5] * opt.output_nc
        #     elif cur == 2:
        #         mt = [0.5] * opt.input_nc
        #     mtp = tuple(mt)
        #     # print("mtp: ", mtp)
        #     transform_list += [transforms.Normalize(mtp, mtp)]

        return transforms.Compose(transform_list)
