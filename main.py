import torch
# import torch.nn as nn
import argparse
# import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
# import random
from dataset import HyperDatasetValid, HyperDatasetTrain  # Clean Data set
from models.models import create_model
# from utils.util import AverageMeter, initialize_logger, save_checkpoint, record_loss, LossTrainCSS, Loss_valid
from utils.visualizer import Visualizer
# from collections import OrderedDict
import numpy as np


parser = argparse.ArgumentParser(description="SSRGAN")
parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
parser.add_argument("--end_epoch", type=int, default=100+1, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--trade_off", type=float, default=10, help="trade_off")
parser.add_argument("--max_iter", type=float, default=300000, help="max_iter")  # patch48:380x450/32x100-534375; patch96:82x450/32x100-113906
parser.add_argument("--outf", type=str, default="SSRResults", help='path log files')
# custom options
parser.add_argument('--name', type=str, default='RGB2HSI', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--isTrain', type=bool, default=True, help='isTrain')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
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
parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
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
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids


def main():
    cudnn.benchmark = True

    # load dataset
    # print("\nloading dataset ...")
    # train_data1 = HyperDatasetTrain1(mode='train')
    # train_data2 = HyperDatasetTrain2(mode='train')
    # train_data3 = HyperDatasetTrain3(mode='train')
    # train_data4 = HyperDatasetTrain4(mode='train')
    # print("Train1:%d,Train2:%d,Train3:%d,Train4:%d," % (len(train_data1), len(train_data2), len(train_data3), len(train_data4),))
    # val_data = HyperDatasetValid(mode='valid')
    # print("Validation set samples: ", len(val_data))
    print("\nloading dataset ...")
    train_data = HyperDatasetTrain(mode='train', opt=opt)
    print("Train datasets: %d " % (len(train_data)))
    val_data = HyperDatasetValid(mode='valid', opt=opt)
    print("Validation set samples: ", len(val_data))

    # Data Loader (Input Pipeline)
    # train_loader1 = DataLoader(dataset=train_data1, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # train_loader2 = DataLoader(dataset=train_data2, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # train_loader3 = DataLoader(dataset=train_data3, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # train_loader4 = DataLoader(dataset=train_data4, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # train_loader = [train_loader1, train_loader2, train_loader3, train_loader4]
    # val_loader = DataLoader(dataset=val_data, batch_size=1,  shuffle=False, num_workers=2, pin_memory=True)

    train_loader = DataLoader(dataset=train_data, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    # val_loader = DataLoader(dataset=val_data, batch_size=1,  shuffle=False, num_workers=2, pin_memory=True)

    # model
    print("\nbuilding models_baseline ...")
    # model = AWAN(3, 31, 200, 8)
    model = create_model(opt)
    print('Parameters number is %.3f M' % (sum(param.numel() for param in model.parameters())/1e6))
    visualizer = Visualizer(opt)

    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    # total_steps = (start_epoch-1) * dataset_size + epoch_iter

    # Loss function
    # criterion_train = LossTrainCSS()
    # criterion_valid = Loss_valid()
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)  # batchsize integer times
    # if torch.cuda.is_available():
    #     model.cuda()
    #     criterion_train.cuda()
    #     criterion_valid.cuda()

    # Parameters, Loss and Optimizer
    start_epoch = 1
    epoch_iter = 0
    # iteration = 0
    # record_val_loss = 1000
    # optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    total_steps = (start_epoch-1) * len(train_data) + epoch_iter

    # visualzation
    # os.makedirs(opt.outf, exist_ok=True)
    # loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+')
    # log_dir = os.path.join(opt.outf, 'train.log')
    # logger = initialize_logger(log_dir)
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
        train_loss, iteration, lr = train(train_loader, model, optimizer_G, optimizer_D, epoch, total_steps, start_time, visualizer, iter_path)
        # val_loss = validate(val_loader, model, criterion_valid)

        # train_loss, iteration, lr = train(train_loader, model, criterion_train, optimizer, epoch, iteration, opt.init_lr, opt.decay_power, opt.trade_off)
        # val_loss = validate(val_loader, model, criterion_valid)
        # # Save model
        # if torch.abs(val_loss - record_val_loss) < 0.0001 or val_loss < record_val_loss:
        #     save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
        #     if val_loss < record_val_loss:
        #         record_val_loss = val_loss
        # # print loss
        # end_time = time.time()
        # epoch_time = end_time - start_time
        # print("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f "
        #       % (epoch, iteration, epoch_time, lr, train_loss, val_loss))
        # # save loss
        # record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, val_loss)
        # logger.info("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f "
        #             % (epoch, iteration, epoch_time, lr, train_loss, val_loss))


# Training
def train(train_loader, model, optimizer_G, optimizer_D, epoch, total_steps, start_time, visualizer, iter_path):
    # random.shuffle(train_loader)
    # whether to collect output images
    save_fake = True if opt.display_freq else False

    for i, (rgb, hyper) in enumerate(train_loader):
        total_steps += 1
        # print("rgb shape {} hyper {}: ".format(rgb.shape, hyper.shape))
        # Forward Pass
        losses, generated = model(Variable(rgb),
                                  Variable(hyper),
                                  infer=save_fake)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

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
        # print out errors
        if i % opt.display_freq == 0:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - start_time) / opt.display_freq
            visualizer.print_current_errors(epoch, i, errors, t)
            # visualizer.plot_current_errors(errors, total_steps)

        # display output images
        if save_fake:
            # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
            #                        ('synthesized_image', util.tensor2im(generated.data[0])),
            #                        ('real_image', util.tensor2im(data['image'][0]))])
            # visualizer.display_current_results(visuals, epoch, total_steps)
            visualizer.display_samples(rgb, hyper, generated, epoch, total_steps)

        # save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, i), delimiter=',', fmt='%d')

        # if epoch_iter >= dataset_size:
        #     break
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.end_epoch, time.time() - start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    # instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()


# # Validate
# def validate(val_loader, model, criterion):
#     model.eval()
#     for i, (input, target) in enumerate(val_loader):
#         input = input.cuda()
#         target = target.cuda()

#         generated = model.inference(data['label'], data['inst'], data['image'])
#         visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
#                             ('synthesized_image', util.tensor2im(generated.data[0]))])
#         img_path = data['path']
#         print('process image... %s' % img_path)
#         visualizer.save_images(webpage, visuals, img_path)

#     return losses.avg

# # Training
# def train(train_loader, model, criterion, optimizer, epoch, iteration, init_lr, decay_power, trade_off):
#     model.train()
#     random.shuffle(train_loader)
#     losses = AverageMeter()
#     losses_rgb = AverageMeter()
#     for k, train_data_loader in enumerate(train_loader):
#         for i, (images, labels) in enumerate(train_data_loader):
#             labels = labels.cuda()
#             images = images.cuda()
#             images = Variable(images)
#             labels = Variable(labels)

#             # Decaying Learning Rate
#             lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=opt.max_iter, power=decay_power)
#             iteration = iteration + 1

#             # Forward + Backward + Optimize
#             output = model(images)
#             loss, loss_rgb = criterion(output, labels, images)
#             loss_all = loss + trade_off * loss_rgb
#             optimizer.zero_grad()
#             loss_all.backward()

#             # Calling the step function on an Optimizer makes an update to its parameters
#             optimizer.step()

#             #  record loss
#             losses.update(loss.data)
#             losses_rgb.update(loss_rgb.data)
#             print('[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f, rgb_train_losses.avg=%.9f'
#                   % (epoch, k+1, len(train_loader), iteration, lr, losses.avg, losses_rgb.avg))

#     return losses.avg, iteration, lr


# # Validate
# def validate(val_loader, model, criterion):
#     model.eval()
#     losses = AverageMeter()
#     for i, (input, target) in enumerate(val_loader):
#         input = input.cuda()
#         target = target.cuda()
#         with torch.no_grad():
#             # compute output
#             output = model(input)
#             loss = criterion(output, target)
#         # record loss
#         losses.update(loss.data)

#     return losses.avg


# # Learning rate
# def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
#     """Polynomial decay of learning rate
#         :param init_lr is base learning rate
#         :param iter is a current iteration
#         :param lr_decay_iter how frequently decay occurs, default is 1
#         :param max_iter is number of maximum iterations
#         :param power is a polymomial power

#     """
#     if iteraion % lr_decay_iter or iteraion > max_iter:
#         return optimizer

#     lr = init_lr*(1 - iteraion/max_iter)**power
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     return lr


if __name__ == '__main__':
    print(torch.__version__)
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    # print("opt.gpu_ids: ", opt.gpu_ids)
    # set gpu ids
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    # print("opt.gpu_ids: ", opt.gpu_ids)

    main()
    # print(torch.__version__)
