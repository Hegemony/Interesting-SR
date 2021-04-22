import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import cv2


class LRDataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.LR_env = None  # environment for lmdb

        # read image list from lmdb or image files
        self.paths_LR, _ = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        assert self.paths_LR, 'Error: LR paths are empty.'

    def __getitem__(self, index):
        LR_path = None

        # get LR image
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(self.LR_env, LR_path)
        H, W, C = img_LR.shape

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # 加入sobel算子
        if self.opt['sobel']:
            H, W, C = img_LR.shape
            img_LR_S = util.channel_convert(C, self.opt['color_s'],
                                            [img_LR])[0]
            img_LR_S = img_LR_S * 255
            ##### sobel算法 ##########
            x = cv2.Sobel(img_LR_S, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(img_LR_S, cv2.CV_16S, 0, 1)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            img_LR_S = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            ##### sobel算法 ##########
            img_LR_S = img_LR_S / 255
            # img_LR_S = img_LR_S[:, :, :]
            img_LR_S = np.reshape(img_LR_S, (img_LR_S.shape[0], img_LR_S.shape[1], 1))  # 64 * 64 * 1
            img_LR_S = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_LR_S, (2, 0, 1)))).float()  # HWC to CHW， numpy to tensor
            # print('img_lq_s:', img_LQ_S.shape)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        # return {'LQ': img_LR, 'LQ_path': LR_path}
        if self.opt['sobel']:  # 加入sobel
            # print(self.opt.keys())
            return {'LQ': img_LR, 'LQ_path': LR_path, 'LQ_S': img_LR_S}

        else:  # original
            # print(self.opt.keys(), '~~~~')
            return {'LQ': img_LR, 'LQ_path': LR_path}

    def __len__(self):
        return len(self.paths_LR)

