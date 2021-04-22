'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:  # 分布式训练
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])  # 每个gpu用6个线程
            batch_size = dataset_opt['batch_size']    # batch为16
            shuffle = True
        # return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        #                                    num_workers=num_workers, sampler=sampler, drop_last=True,
        #                                    pin_memory=False)  #  drop_last = True数据能整除 batch_size

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=False,
                                           pin_memory=False)  # drop_last = False 数据不能整除batch_size
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):   #创建数据集
    mode = dataset_opt['mode']
    if mode == 'LR':
        from codes.data.LR_dataset import LRDataset as D
    elif mode == 'LQGT':   # 配置文件中默认的mode为'LQGT'
        from codes.data.LQGT_dataset import LQGTDataset as D
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)   #利用data/LQDT_dataset/LQGTdataset创建数据集

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
