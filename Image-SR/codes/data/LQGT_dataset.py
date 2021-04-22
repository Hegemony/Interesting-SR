import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from data.data_loader import noiseDataset


class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']   # img格式或者lmdb格式
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])  # 调用data/util文件下的get_image_paths函数
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]
        # print(opt['aug'])
        if self.opt['phase'] == 'train':
            if opt['aug'] and 'noise' in opt['aug']:   # 暂时不使用噪声
                self.noises = noiseDataset(opt['noise_data'], opt['GT_size']/opt['scale'])

        # print(self.opt.items())

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']        # scale为4
        GT_size = self.opt['GT_size']    # GT_size为128

        # get GT image  获取GroundTruth图像
        GT_path = self.paths_GT[index]    # 根据index索引获取图片路径
        if self.data_type == 'lmdb':      # 不执行
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)  # GT_env为None，是img格式的图像 用opencv读取为BGR格式，维度是H*W*C
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)
        # change color space if necessary
        if self.opt['color']:  # opt['color']为RGB， 目标域为RGB
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LQ image  如果有LQ图像则获取Low Quality图像
        if self.paths_LQ: # 如果有LQ的图片，就直接加载
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)

        else:  # down-sampling on-the-fly  否则对原始GT图像进行下采样得到LQ图像
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)



        if self.opt['phase'] == 'train':

            # if the image size is too small， 不满足img_GT.shape处理成img_GT.shape
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:  # GT_size为128
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale  # 128 // 4 = 32

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]  # 随机选取一个点，以该点为单位找到 32 * 32的子块
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)       # 上面得到的随机数 * 4 找到原图对应的位置，尺寸为 128 * 128 的子块
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate  图像增强，翻转和旋转
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition
        # 加入sobel算子
        if self.opt['sobel']:
            H, W, C = img_LQ.shape
            img_LQ_S = util.channel_convert(C, self.opt['color_s'],
                                          [img_LQ])[0]
            img_LQ_S = img_LQ_S * 255
            ##### sobel算法 ##########
            x = cv2.Sobel(img_LQ_S, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(img_LQ_S, cv2.CV_16S, 0, 1)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            img_LQ_S = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            ##### sobel算法 ##########
            img_LQ_S = img_LQ_S / 255
            # img_LQ_S = img_LQ_S[:, :, :]
            img_LQ_S = np.reshape(img_LQ_S, (img_LQ_S.shape[0], img_LQ_S.shape[1], 1))  # 64 * 64 * 1
            img_LQ_S = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_S, (2, 0, 1)))).float()  # HWC to CHW， numpy to tensor
            # print('img_lq_s:', img_LQ_S.shape)


        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]  # BGR to RGB
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()  # HWC to CHW， numpy to tensor
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        # print('img_lq:', img_LQ.shape)


        if self.opt['phase'] == 'train':
            # add noise to LR during train
            if self.opt['aug'] and 'noise' in self.opt['aug']:
                noise = self.noises[np.random.randint(0, len(self.noises))]
                img_LQ = torch.clamp(img_LQ + noise, 0, 1)

        if LQ_path is None:
            LQ_path = GT_path
        # print(self.opt.keys())
        if self.opt['sobel']:  # 加入sobel
            # print(self.opt.keys())
            return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path, 'LQ_S': img_LQ_S}

        else: # original
            # print(self.opt.keys(), '~~~~')
            return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)  # 一个batch里面图片的路径数，也就对应图片数

