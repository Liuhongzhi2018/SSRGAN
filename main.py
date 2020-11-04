import torch
# import torch.nn as nn
import argparse
# import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
import random
from utils.dataset import HyperDatasetValid, HyperDatasetTrain  # Clean Data set
from models.models import create_model
# from utils.util import AverageMeter, initialize_logger, save_checkpoint, record_loss, LossTrainCSS, Loss_valid
from utils.visualizer import Visualizer
from utils.util import initialize_logger
from utils.EvalMetrics import computeMRAE
# from collections import OrderedDict
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


parser = argparse.ArgumentParser(description="SSRGAN")
parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
parser.add_argument("--SEED", type=int, default=2020, help="random seed")
parser.add_argument("--end_epoch", type=int, default=200+1, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--trade_off", type=float, default=10, help="trade_off")
parser.add_argument("--max_iter", type=float, default=300000, help="max_iter")  # patch48:380x450/32x100-534375; patch96:82x450/32x100-113906
parser.add_argument("--outf", type=str, default="SSRGAN_Results", help='path log files')
# custom options
parser.add_argument('--name', type=str, default='RGB2HSI', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--isTrain', type=bool, default=True, help='isTrain')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
parser.add_argument('--display_freq', type=int, default=5, help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')
parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--save_latest_freq', type=int, default=300, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
parser.add_argument('--no_flip', action='store_true', default=False, help='if specified, do not flip the images for data argumentation')
parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=31, help='# of output HSI channels')

parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

torch.manual_seed(opt.SEED)
torch.cuda.manual_seed(opt.SEED)
np.random.seed(opt.SEED)
random.seed(opt.SEED)


def main():
    cudnn.benchmark = True

    # load dataset
    print("\nloading dataset ...")
    train_data = HyperDatasetTrain(mode='train', opt=opt)
    print("Train datasets: %d " % (len(train_data)))
    val_data = HyperDatasetValid(mode='valid', opt=opt)
    print("Validation set samples: ", len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1,  shuffle=False, num_workers=2, pin_memory=True)

    # model
    print("\nbuilding models_baseline ...")
    model = create_model(opt)
    print('Parameters number is %.3f M' % (sum(param.numel() for param in model.parameters())/1e6))
    visualizer = Visualizer(opt)

    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    # Parameters, Loss and Optimizer
    start_epoch = 1
    epoch_iter = 0
    # iteration = 0
    total_steps = (start_epoch-1) * len(train_data) + epoch_iter
    max_PSNR, max_epoch = 0, 0

    # visualzation
    os.makedirs(opt.outf, exist_ok=True)
    log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'train_val_metric.log')
    logger = initialize_logger(log_dir)

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    # # Resume
    # resume_file = opt.outf + '/net_9epoch.pth'
    # resume_file = ''
    # if resume_file:
    #     if os.path.isfile(resume_file):
    #         print("=> loading checkpoint '{}'".format(resume_file))
    #         checkpoint = torch.load(resume_file)
    #         start_epoch = checkpoint['epoch']
    #         iteration = checkpoint['iter']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])

    # start epoch
    for epoch in range(start_epoch, opt.end_epoch):
        start_time = time.time()
        train_PSNR, train_SSIM, val_PSNR, val_SSIM, val_MRAE, max_PSNR, max_epoch = train(train_loader, val_loader, model,
                                                                                          optimizer_G, optimizer_D, epoch,
                                                                                          total_steps, start_time, visualizer,
                                                                                          iter_path, max_PSNR, max_epoch)
        end_time = time.time()
        epoch_time = end_time - start_time
        logger.info(" Epoch [%02d], Time:%.9f, Train PSNR: %.4f SSIM: %.4f Val PSNR: %.4f SSIM: %.4f MRAE: %.4f"
                    % (epoch, epoch_time, train_PSNR, train_SSIM, val_PSNR, val_SSIM, val_MRAE))


# Training
def train(train_loader, val_loader, model, optimizer_G, optimizer_D, epoch, total_steps, start_time, visualizer, iter_path, max_PSNR, max_epoch):
    save_fake = True if opt.display_freq else False
    psnr_list = []
    ssim_list = []
    mrae_list = []
    for i, (rgb, hyper) in enumerate(train_loader):
        total_steps += 1
        # print("rgb shape {} hyper shape {}".format(rgb.shape, hyper.shape))
        # rgb shape torch.Size([1, 3, 482, 512]) hyper shape torch.Size([1, 31, 482, 512])
        # print("rgb max {} rgb min {} hyper max {} hyper min {}".format(rgb.max(), rgb.min(), hyper.max(), hyper.min()))
        # rgb max 1.0 rgb min -1.0 hyper max 1.0 hyper min -0.9982895851135254

        # Forward Pass
        losses, generated = model(Variable(rgb),
                                  Variable(hyper),
                                  infer=save_fake)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))
        # print("loss_dict: ", loss_dict)
        # loss_dict:  {'G_GAN': tensor(12.2738, device='cuda:0', grad_fn=<MeanBackward0>),
        # 'G_GAN_Feat': tensor(15.5414, device='cuda:0', grad_fn=<MeanBackward0>),
        # 'G_CSS': tensor(4.9444, device='cuda:0', grad_fn=<MeanBackward0>),
        # 'D_real': tensor(5.7572, device='cuda:0', grad_fn=<MeanBackward0>),
        # 'D_fake': tensor(7.2036, device='cuda:0', grad_fn=<MeanBackward0>)}

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_CSS', 0)

        # Backward Pass #
        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Display results and errors #
        psnr, ssim, l1, mrae = metrics(generated.cuda(), hyper.cuda())
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        mrae_list.append(mrae.item())
        # print out errors
        if i % opt.display_freq == 0:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - start_time) / opt.display_freq
            visualizer.print_current_errors(epoch, i, errors, t)
            # visualizer.plot_current_errors(errors, total_steps)

        # display output images
        if save_fake and i % opt.display_freq == 0:
            visualizer.display_samples(rgb, hyper, generated, epoch, total_steps, i, mode=True)

        # # save latest model
        # if total_steps % opt.save_latest_freq == 0:
        #     print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
        #     model.save('latest')
        #     np.savetxt(iter_path, (epoch, i), delimiter=',', fmt='%d')

    avg_psnr = np.average(psnr_list)
    avg_ssim = np.average(ssim_list)
    avg_mrae = np.average(mrae_list)
    print('End of epoch %d / %d \t PSNR %.4f SSIM %.4f MARE %.4f Time Taken: %d sec' %
          (epoch, opt.end_epoch, avg_psnr, avg_ssim, avg_mrae, time.time() - start_time))

    print("--- Start Validation ---")
    val_PSNR, val_SSIM, val_MRAE = validate(val_loader, model, epoch, total_steps, visualizer)

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0 and val_PSNR > max_PSNR:
        max_PSNR = val_PSNR
        max_epoch = epoch
        # print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        print('Saving the model at the end of epoch %d' % (epoch))
        model.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
    model.save('latest')

    print('The best epoch is %d when max_PSNR is %.4f' % (max_epoch, max_PSNR))

    # instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.update_fixed_params()

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()

    return avg_psnr, avg_ssim, val_PSNR, val_SSIM, val_MRAE, max_PSNR, max_epoch


# Validate
def validate(val_loader, model, epoch, total_steps, visualizer):
    start_time = time.time()
    psnr_list = []
    ssim_list = []
    mrae_list = []
    for i, (rgb, hyper) in enumerate(val_loader):
        # print("rgb max {} rgb min {} hyper max {} hyper min {}".format(rgb.max(), rgb.min(), hyper.max(), hyper.min()))
        generated = model.inference(Variable(rgb), Variable(hyper))
        visualizer.display_samples(rgb, hyper, generated, epoch, total_steps, i, mode=False)
        psnr, ssim, l1, mrae = metrics(generated.cuda(), hyper.cuda())
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        mrae_list.append(mrae.item())

    avg_psnr = np.average(psnr_list)
    avg_ssim = np.average(ssim_list)
    avg_mrae = np.average(mrae_list)
    print('Epoch %d \t Validation PSNR %.4f SSIM %.4f MRAE %.4f Time Taken: %d sec' %
          (epoch, avg_psnr, avg_ssim, avg_mrae, time.time() - start_time))

    return avg_psnr, avg_ssim, avg_mrae


def metrics(inputs, gts):
    # inputs = self.postprocess(inputs)
    # gts = self.postprocess(gts)
    psnr_value = []
    l1_value = torch.mean(torch.abs(inputs-gts))

    [b, n, w, h] = inputs.size()
    inputs = (inputs*255.0).int().float()/255.0
    gts = (gts*255.0).int().float()/255.0

    for i in range(inputs.size(0)):
        inputs_p = inputs[i, :, :, :].cpu().numpy().astype(np.float32).transpose(1, 2, 0)
        gts_p = gts[i, :, :, :].cpu().numpy().astype(np.float32).transpose(1, 2, 0)
        psnr_value.append(compare_psnr(inputs_p, gts_p, data_range=1))

    psnr_value = np.average(psnr_value)
    inputs = inputs.view(b*n, w, h).cpu().numpy().astype(np.float32).transpose(1, 2, 0)
    gts = gts.view(b*n, w, h).cpu().numpy().astype(np.float32).transpose(1, 2, 0)
    ssim_value = compare_ssim(inputs, gts, data_range=1, win_size=51, multichannel=True)
    # print("inputs value: ", inputs.max(), inputs.min(), "gts value: ", gts.max(), gts.min())
    norm_inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
    norm_gts = (gts - gts.min()) / (gts.max() - gts.min())
    # print("norm_gts value: ", norm_inputs.max(), norm_inputs.min(), "norm_gts value: ", norm_gts.max(), norm_gts.min())
    # print("norm_gts value: ", norm_inputs, "norm_gts value: ", norm_gts)
    mrae = computeMRAE(norm_inputs, norm_gts)

    return psnr_value, ssim_value, l1_value, mrae


if __name__ == '__main__':
    print(torch.__version__)
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    # set gpu ids
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    main()
