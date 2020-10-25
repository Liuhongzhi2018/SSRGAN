import os
import os.path
# import h5py  # error appear
# from scipy.io import loadmat
import cv2
import glob
import numpy as np
import argparse
import hdf5storage
# import random

from scipy.io import loadmat


parser = argparse.ArgumentParser(description="SpectralSRGAN")
parser.add_argument("--data_path", type=str, default='./NTIRE2020', help="data path")
parser.add_argument("--patch_size", type=int, default=64, help="data patch size")
parser.add_argument("--stride", type=int, default=32, help="data patch stride")
parser.add_argument("--train_data_path", type=str, default='./Dataset/Train', help="preprocess_data_path")
# parser.add_argument("--train_data_path2", type=str, default='./Dataset/Train2', help="preprocess_data_path")
# parser.add_argument("--train_data_path3", type=str, default='./Dataset/Train3', help="preprocess_data_path")
# parser.add_argument("--train_data_path4", type=str, default='./Dataset/Train4', help="preprocess_data_path")
opt = parser.parse_args()


def main():
    # if not os.path.exists(opt.train_data_path):
    #     os.makedirs(opt.train_data_path)
    # if not os.path.exists(opt.train_data_path2):
    #     os.makedirs(opt.train_data_path2)
    # if not os.path.exists(opt.train_data_path3):
    #     os.makedirs(opt.train_data_path3)
    # if not os.path.exists(opt.train_data_path4):
    #     os.makedirs(opt.train_data_path4)
    # process_data(patch_size=opt.patch_size, stride=opt.stride, mode='train')

    os.makedirs(opt.train_data_path, exist_ok=True)
    process_data(mode='train')


def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    print("img.shape: ", img.shape, " patch.shape: ", patch.shape)
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


# def process_data(patch_size, stride, mode):
def process_data(mode):
    if mode == 'train':
        print("\nprocess training set ...\n")
        # patch_num = 1
        filenames_hyper = glob.glob(os.path.join(opt.data_path, 'NTIRE2020_Train_Spectral', '*.mat'))
        filenames_rgb = glob.glob(os.path.join(opt.data_path, 'NTIRE2020_Train_Clean', '*.png'))
        filenames_hyper.sort()
        filenames_rgb.sort()
        # for k in range(1):  # make small dataset
        for k in range(len(filenames_hyper)):
            # print([filenames_hyper[k], filenames_rgb[k]])
            print("filenames_hyper: ", [filenames_hyper[k], filenames_rgb[k]])
            # load hyperspectral image
            # mat = h5py.File(filenames_hyper[k], 'r')
            mat = loadmat(filenames_hyper[k])
            # print('mat:\n',mat)
            print('mat:\n', mat['cube'].shape)
            hyper = np.float32(np.array(mat['cube']))
            # print('hyper:\n',hyper.shape)  #  (482, 512, 31)
            # hyper = np.transpose(hyper, [0, 2, 1])
            # hyper = np.transpose(hyper, [2, 1, 0])
            # print('hyper transpose:\n',hyper.shape)
            hyper = normalize(hyper, max_val=1., min_val=0.)
            print('hyper:\n', hyper.shape)  # hyper:(482, 512, 31)  HWC
            # load rgb image
            rgb = cv2.imread(filenames_rgb[k])  # imread -> BGR model
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            # rgb = np.transpose(rgb, [2, 0, 1])
            # rgb = np.transpose(rgb, [2, 1, 0])
            print('rgb:\n', rgb.shape)    # rgb:(482, 512, 3)  HWC
            rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
            # creat patches
            # patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
            # patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)

            # add data ：重组patches
            # print("patches_hyper shape: ", patches_hyper.shape)
            # for j in range(patches_hyper.shape[3]):
            #     print("generate training sample #%d" % patch_num)
            #     sub_hyper = patches_hyper[:, :, :, j]
            #     sub_rgb = patches_rgb[:, :, :, j]
            #     train_data_path_array = [opt.train_data_path1, opt.train_data_path2, opt.train_data_path3, opt.train_data_path4]
            #     random.shuffle(train_data_path_array)
            #     train_data_path = os.path.join(train_data_path_array[0], 'train'+str(patch_num)+'.mat')
            #     # hdf5storage.savemat(train_data_path, {'rad': sub_hyper}, format='7.3')
            #     hdf5storage.savemat(train_data_path, {'cube': sub_hyper}, format='7.3')
            #     hdf5storage.savemat(train_data_path, {'rgb': sub_rgb}, format='7.3')
            #     patch_num += 1

            train_data_path = os.path.join(opt.train_data_path, 'train'+str(k)+'.mat')
            hdf5storage.savemat(train_data_path, {'cube': hyper}, format='7.3')
            hdf5storage.savemat(train_data_path, {'rgb': rgb}, format='7.3')

        print("\ntraining set: # HSI mat %d\n" % (k+1))  # training set: # samples 94500


if __name__ == '__main__':
    main()
