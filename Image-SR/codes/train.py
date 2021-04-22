import os
import math
import argparse
import random
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from codes.data.data_sampler import DistIterSampler

import codes.options.options as option
from codes.utils import util
from codes.data import create_dataloader, create_dataset
from codes.models import create_model


def init_dist(backend='nccl', **kwargs):  # 分布式训练的代码
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')  # 加载.yml文件的路径
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)  # 调用options/options.py下的parse函数，完成参数配置， 传入的args.opt为上面的路径

    #### distributed training settings   分布式训练
    if args.launcher == 'none':  # disabled distributed training  不能分布式训练
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:  # 可以分布式训练 （该段不执行）
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):  # 分布式训练默认恢复检查点（该段不执行）
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0) 该部分就是正常训练
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(
                log_dir='../tb_logger/' + opt['name'])  # tb_logger/Corrupted_noise/下的是tensorboard文件
    else:  # 该部分为分布式训练
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)  # 对于opt中，key没有对应Value的属性设置为None

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True  # 使用Cudnn加速训练
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    train_loader = None
    val_loader = None
    best_psnr = 0  # 最好的PSNR
    best_iter = 0  # 最好的迭代次数
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # print('\n\n\n\n\n\n\n\n', dataset_opt)
            train_set = create_dataset(dataset_opt)  # 调用data/__init__.py/create_dataset函数创建训练集
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            # math.ceil 对一个数进行上取整。 训练集大小/batch大小，训练集为800张图片 batch为16, train_size = 50,取50个batch将一个训练集取完
            total_iters = int(opt['train']['niter'])  # 总迭代次数 60001
            total_epochs = int(
                math.ceil(total_iters / train_size))  # epoch = 总迭代次数 / train_size = 60000 / 50 = 1200epoch
            if opt['dist']:  # 分布式训练
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt,
                                             train_sampler)  # 调用data/__init__.py/create_dataloader函数创建训练集
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)  # 调用models/__init__.py 创建模型

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):  # 共1200个epoch
        if opt['dist']:  # 分布式训练
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):  # 一次从训练集中取出一个batch
            current_step += 1  # current_step每次从train_loader取出一个batch为一步
            if current_step > total_iters:
                break
            #### update learning rate 更新学习率 在/models/base_model.py下的update_learning_rate函数
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)  # 调用models/SRGAN_model.py的feed_data函数
            model.optimize_parameters(current_step)  # 调用models/SRGAN_model.py的optimize_parameters函数，优化参数，定义损失函数计算公式

            #### log
            if current_step % opt['logger']['print_freq'] == 0:  # 每迭代100次也就是每两个epoch 打印一条记录日志
                logs = model.get_current_log()
                message = '<epoch:{:5d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())  # iter代表迭代的次数
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train'][
                'val_freq'] == 0 and rank <= 0 and val_loader is not None:  # val_freq 默认为5e3， 即每5000次迭代打印验证一次
                # avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = 0.0    # original
                avg_psnr = avg_ssim = val_pix_err_f = val_pix_err_nf = val_mean_color_err = 0.0  # myself
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)  # 送入验证集数据
                    model.test()  # 进行测试

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)  # 利用opencv保存网络对验证集超分后的图像

                    # calculate PSNR and SSIM 计算PSNR 和 SSIM
                    crop_size = opt['scale']  # 4
                    gt_img = gt_img / 255.  # 将图像归一化到0~1
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_ssim += util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)  # myself 加入计算SSIM的方法

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx  # myself 计算平均的ssim
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx

                if avg_psnr > best_psnr:  # 记住最好的PSNR和iter
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))  # myself, 打印ssim
                logger.info('# Validation # Best-PSNR: {:.4e}'.format(best_psnr))  # myself, 打印best_psnr
                logger.info('# Validation # Best-iter: {:8,d}'.format(best_iter))  # myself, 打印best_iter
                logger_val = logging.getLogger('val')  # validation logger

                # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                #     epoch, current_step, avg_psnr))   # original
                logger_val.info(
                    '<epoch:{:6d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e} Best psnr: {:.4e} Best iter: {:8,d}'.format(
                        epoch, \
                        current_step, avg_psnr, avg_ssim, best_psnr, best_iter))  # myself , 加入ssim的结果
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)  # myself， tensorboard记录ssim
                    tb_logger.add_scalar('val_pix_err_f', val_pix_err_f, current_step)
                    tb_logger.add_scalar('val_pix_err_nf', val_pix_err_nf, current_step)
                    tb_logger.add_scalar('val_mean_color_err', val_mean_color_err, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()