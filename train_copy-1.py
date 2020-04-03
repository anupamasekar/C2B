'''
-----------------------------------
TRAINING CODE - SHIFTVARCONV + UNET
-----------------------------------
'''
import os 
import numpy as np
import torch
import torch.nn as nn
import logging
import glob
import argparse
# import time
from torch.utils import data
# from skimage.measure import compare_psnr, compare_ssim

from logger import Logger
from dataloader_v2 import Dataset_load
from sensor import C2B
from unet_v3 import UNet
from shift_var_conv import ShiftVarConv2D
import utils


# set random seed
torch.manual_seed(0)
np.random.seed(0)


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, required=True, help='expt name')
parser.add_argument('--epochs', type=int, default=200, help='num epochs to train')
parser.add_argument('--batch', type=int, required=True, help='batch size for training and validation')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--code', type=str, default='seq', help='code to use for c2b "rand" or "seq"')
parser.add_argument('--blocksize', type=int, default=8, help='tile size for code default 3x3')
parser.add_argument('--subframes', type=int, default=16, help='num sub frames')
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')
parser.add_argument('--patchsize', type=int, default=128, help='size of image patches default 240x240')
parser.add_argument('--npatches', type=int, default=5, help='num of patches to take from each input vid')
parser.add_argument('--gpu', type=str, required=True, help='GPU ID')
args = parser.parse_args()
# print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# params for DataLoader
train_params = {'batch_size': args.batch,
                'shuffle': True,
                'num_workers': 40
                }
val_params = {'batch_size': args.batch,
              'shuffle': False,
              'num_workers': 40
              }
input_params = {'height': 720,
                'width': 1280,
                'patch_size': args.patchsize,
                'num_patches': args.npatches,
                'sub_frames': args.subframes
                }

# num_patches = (input_params['height']//input_params['patch_size'])*(input_params['width']//input_params['patch_size'])
lr = args.lr
num_epochs = args.epochs

save_path = os.path.join('/media/data/prasan/C2B/anupama/', args.expt)
utils.create_dirs(save_path)


# tensorboard summary logger
logger = Logger(os.path.join(save_path, 'logs'))


# configure runtime logging
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(save_path, 'logs', 'logfile.log'), 
                    format='%(asctime)s - %(message)s', 
                    filemode='a')
# logger=logging.getLogger()#.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger('').addHandler(console)
logging.info(args)



# initializing training and validation data loaders
data_path = '/media/data/prasan/datasets'

val_path = sorted(glob.glob(data_path + '/test_set/GOPR*/'))
val_images = []
for path in val_path:
    val_images.extend(sorted(glob.glob(path + '*.png'))[:500]) 
    # 500 frames for each video sequence

test_folders = [f[-15:-1] for f in val_path]

training_path = sorted(glob.glob(data_path + '/GOPR*/'))
training_path = [f for f in training_path if f[-15:-1] not in test_folders]
training_images = []
for path in training_path:
    training_images.extend(sorted(glob.glob(path + '*.png'))[:500])



# initializing training and validation data generators
training_set = Dataset_load(training_images, **input_params)
training_generator = data.DataLoader(training_set, **train_params)
logging.info('Loaded training set: %d sequences %d images %d videos'
            %(len(training_images)//500, len(training_images), len(training_set)))

validation_set = Dataset_load(val_images, **input_params)
validation_generator = data.DataLoader(validation_set, **val_params)
logging.info('Loaded validation set: %d sequences %d images %d videos'
            %(len(val_images)//500, len(val_images), len(validation_set)))



c2b = C2B(block_size=args.blocksize, sub_frames=args.subframes, patch_size=args.patchsize).cuda()
invNet = ShiftVarConv2D(out_channels=args.subframes, block_size=args.blocksize).cuda()
netG = UNet(in_channel=args.subframes, out_channel=args.subframes, batch_norm=False).cuda()

# optimizer
optimizer = torch.optim.Adam(list(netG.parameters()) + list(invNet.parameters()),
                             lr=lr, weight_decay=1e-5)

# load if checkpoint given
if args.ckpt is None:
    start_epoch = 0
    logging.info('No checkpoint, initialized net')
else:
    ckpt = torch.load(os.path.join(save_path, 'model', args.ckpt))
    c2b.load_state_dict(ckpt['c2b_state_dict'])
    c2b.train()
    invNet.load_state_dict(ckpt['invnet_state_dict'])
    invNet.train()
    netG.load_state_dict(ckpt['unet_state_dict'])
    netG.train()
    optimizer.load_state_dict(ckpt['opt_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    logging.info('Loaded checkpoint from epoch {}'.format(start_epoch-1))
torch.save(c2b.code, os.path.join(save_path, 'logs', 'exposure_code.pth'))

# define losses
l1 = nn.L1Loss()


logging.info('Starting training')
for i in range(start_epoch, start_epoch+num_epochs):

    ## TRAINING
    train_iter = 0
    l1_interm_loss_sum = 0.
    l1_final_loss_sum = 0.
    tv_loss_sum = 0.
    loss_sum = 0.
    psnr_sum = 0.
    # ssim_sum = 0.

    for gt_vid in training_generator:   

        # t2 = time.time()
        gt_vid = gt_vid.cuda()
        gt_vid = gt_vid.reshape(-1, input_params['sub_frames'], input_params['patch_size'], input_params['patch_size'])

        b1 = c2b(gt_vid) # (N,1,H,W)
        interm_vid = invNet(b1)        
        highRes_vid = netG(interm_vid) # (N,9,H,W)

        psnr_sum += utils.compute_psnr(highRes_vid, gt_vid).item()

        ## LOSSES
        b1_est = c2b(interm_vid)
        l1_interm_loss = l1(b1, b1_est)
        l1_interm_loss_sum += l1_interm_loss.item()

        l1_final_loss = l1(highRes_vid, gt_vid)
        l1_final_loss_sum += l1_final_loss.item()
        tv_loss = utils.gradx(highRes_vid).abs().mean() + utils.grady(highRes_vid).abs().mean()
        tv_loss_sum += tv_loss.item()

        loss = l1_interm_loss + l1_final_loss + 0.1*tv_loss
        loss_sum += loss.item()

        ## BACKPROP
        optimizer.zero_grad()
        loss.backward()       
        optimizer.step()

        if train_iter % 100 == 0:
            logging.info('epoch: %3d \t iter: %6d \t loss: %.4f'%(i, train_iter, loss.item()))
        
        train_iter += 1
        # t1 = time.time()

    # logging.info(time.strftime('T12: %H:%M:%S', time.gmtime(T12))) 
    # logging.info(time.strftime('T23: %H:%M:%S', time.gmtime(T23))) 
    # logging.info(time.strftime('T34: %H:%M:%S', time.gmtime(T34))) 
    # logging.info(time.strftime('T45: %H:%M:%S', time.gmtime(T45))) 
    # logging.info(time.strftime('T56: %H:%M:%S', time.gmtime(T56))) 
    # logging.info(time.strftime('T67: %H:%M:%S', time.gmtime(T67))) 
    # logging.info(time.strftime('T78: %H:%M:%S', time.gmtime(T78)))
    # logging.info(time.strftime('T89: %H:%M:%S', time.gmtime(T89)))

    logging.info('Total train iterations: %d'%(train_iter))
    logging.info('Finished epoch %3d with loss: %.4f psnr: %.4f'
                %(i, loss_sum/train_iter, psnr_sum/(len(training_set)*input_params['num_patches'])))


    # dump tensorboard summaries
    logger.scalar_summary(tag='training/psnr', value=psnr_sum/(len(training_set)*input_params['num_patches']), step=i)
    logger.scalar_summary(tag='training/loss', value=loss_sum/train_iter, step=i)
    logger.scalar_summary(tag='training/l1_interm_loss', value=l1_interm_loss_sum/train_iter, step=i)
    logger.scalar_summary(tag='training/l1_final_loss', value=l1_final_loss_sum/train_iter, step=i)
    logger.scalar_summary(tag='training/tv_loss', value=tv_loss_sum/train_iter, step=i)
    logging.info('Dumped tensorboard summaries for epoch %4d'%(i))


    ## VALIDATION
    if ((i+1) % 5 == 0) or ((i+1) == (start_epoch+num_epochs)):
        
        logging.info('Starting validation')
        val_iter = 0
        val_loss_sum = 0.
        val_psnr_sum = 0.
        val_ssim_sum = 0.

        with torch.no_grad():
            for gt_vid in validation_generator:
                gt_vid = gt_vid.cuda()
                gt_vid = gt_vid.reshape(-1, input_params['sub_frames'], input_params['patch_size'], input_params['patch_size'])
                b1 = c2b(gt_vid) # (N,1,H,W)
                interm_vid = invNet(b1)               
                highRes_vid = netG(interm_vid) # (N,9,H,W)

                val_psnr_sum += utils.compute_psnr(highRes_vid, gt_vid).item()
                val_ssim_sum += utils.compute_ssim(highRes_vid, gt_vid).item()
                    
                ## loss
                b1_est = c2b(interm_vid)
                l1_interm_loss = l1(b1, b1_est)
                l1_final_loss = l1(highRes_vid, gt_vid)
                tv_loss = utils.gradx(highRes_vid).abs().mean() + utils.grady(highRes_vid).abs().mean()

                val_loss_sum += (l1_interm_loss + l1_final_loss + 0.1*tv_loss).item()

                if val_iter % 100 == 0:
                    print('In val iter {}'.format(val_iter))

                val_iter += 1

        logging.info('Total val iterations: {}'.format(val_iter))
        logging.info('Finished validation with loss: %.4f psnr: %.4f ssim: %.4f'
                    %(val_loss_sum/val_iter, val_psnr_sum/(len(validation_set)*input_params['num_patches']), 
                        val_ssim_sum/(len(validation_set)*input_params['num_patches'])))

        # dump tensorboard summaries
        logger.scalar_summary(tag='validation/loss', value=val_loss_sum/val_iter, step=i)
        logger.scalar_summary(tag='validation/psnr', value=val_psnr_sum/(len(validation_set)*input_params['num_patches']), step=i)
        logger.scalar_summary(tag='validation/ssim', value=val_ssim_sum/(len(validation_set)*input_params['num_patches']), step=i)
    
    ## CHECKPOINT
    if ((i+1) % 10 == 0) or ((i+1) == (start_epoch+num_epochs)):
        utils.save_checkpoint(state={'epoch': i, 
                                    'invnet_state_dict': invNet.state_dict(),
                                    'unet_state_dict': netG.state_dict(),
                                    'c2b_state_dict': c2b.state_dict(),
                                    'opt_state_dict': optimizer.state_dict()},
                            save_path=os.path.join(save_path, 'model'),
                            filename='model_%.6d.pth'%(i))
        logging.info('Saved checkpoint for epoch {}'.format(i))

logging.info('Finished training')