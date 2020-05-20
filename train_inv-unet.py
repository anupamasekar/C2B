'''
-----------------------------------
TRAINING CODE - SHIFTVARCONV + UNET
-----------------------------------
'''
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import glob
import argparse
import time
from torch.utils import data
# from skimage.measure import compare_psnr, compare_ssim


## set random seed
torch.manual_seed(12)
np.random.seed(12)


from logger import Logger
from dataloader_v4 import Dataset_load
from sensor import C2B
from unet_v3 import UNet
# from unet_model import UNet
from shift_var_conv import ShiftVarConv2D
import utils


## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, required=True, help='expt name')
parser.add_argument('--epochs', type=int, default=500, help='num epochs to train')
parser.add_argument('--batch', type=int, required=True, help='batch size for training and validation')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--blocksize', type=int, default=8, help='tile size for code default 3x3')
parser.add_argument('--subframes', type=int, default=16, help='num sub frames')
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')
parser.add_argument('--ckptpath', type=str, default=None, help='load ckpt from another expt')
parser.add_argument('--gpu', type=str, required=True, help='GPU ID')
parser.add_argument('--mask', type=str, default='random', help='"impulse" or "random" or "opt" or "train"')
parser.add_argument('--two_bucket', action='store_true', help='1 bucket or 2 buckets')
args = parser.parse_args()
# print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

## params for DataLoader
train_params = {'batch_size': args.batch,
                'shuffle': True,
                'num_workers': 20,
                'pin_memory': True}
val_params = {'batch_size': args.batch,
              'shuffle': False,
              'num_workers': 20,
              'pin_memory': True}


lr = args.lr
num_epochs = args.epochs

save_path = os.path.join('/data/prasan/anupama/', args.expt)
utils.create_dirs(save_path)


## tensorboard summary logger
logger = Logger(os.path.join(save_path, 'logs'))


## configure runtime logging
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



## dataloaders using hdf5 file
data_path = '/data/prasan/anupama/dataset/GoPro_patches_ds2_s16-8_p64-32.hdf5'


## initializing training and validation data generators
training_set = Dataset_load(data_path, dataset='train', num_samples='all')
training_generator = data.DataLoader(training_set, **train_params)
logging.info('Loaded training set: %d videos'%(len(training_set)))

validation_set = Dataset_load(data_path, dataset='test', num_samples=60000)
validation_generator = data.DataLoader(validation_set, **val_params)
logging.info('Loaded validation set: %d videos'%(len(validation_set)))



## initialize nets
c2b = C2B(block_size=args.blocksize, sub_frames=args.subframes, mask=args.mask, two_bucket=args.two_bucket).cuda()
invNet = ShiftVarConv2D(out_channels=args.subframes, block_size=args.blocksize, two_bucket=args.two_bucket).cuda()
uNet = UNet(in_channel=args.subframes, out_channel=args.subframes, instance_norm=False).cuda()
# uNet = UNet(n_channels=16, n_classes=16).cuda()

## optimizer
optimizer = torch.optim.Adam(list(invNet.parameters())+list(uNet.parameters())+list(c2b.parameters()),
                             lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, 
                                                        patience=5, min_lr=1e-6, verbose=True)

## load checkpoint
if args.ckpt is None:
    start_epoch = 0
    logging.info('No checkpoint, initialized net')
elif args.ckptpath is None:
    ckpt = torch.load(os.path.join(save_path, 'model', args.ckpt))
    c2b.load_state_dict(ckpt['c2b_state_dict'])
    c2b.train()
    invNet.load_state_dict(ckpt['invnet_state_dict'])
    invNet.train()
    uNet.load_state_dict(ckpt['unet_state_dict'])
    uNet.train()
    optimizer.load_state_dict(ckpt['opt_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    logging.info('Loaded checkpoint from epoch %d'%(start_epoch-1))
else:
    ckpt_path = os.path.join('/data/prasan/anupama/', args.ckptpath, 'model', args.ckpt)
    ckpt = torch.load(ckpt_path)
    if 'c2b_state_dict' in ckpt:
        c2b.load_state_dict(ckpt['c2b_state_dict'])
        c2b.train()
    if 'invnet_state_dict' in ckpt:
        invNet.load_state_dict(ckpt['invnet_state_dict'])
        invNet.train()
    if 'unet_state_dict' in ckpt:
        uNet.load_state_dict(ckpt['unet_state_dict'])
        uNet.train()
    start_epoch = 0
    logging.info('Loaded checkpoint %d from %s'%(ckpt['epoch'], ckpt_path))
torch.save(c2b.code, os.path.join(save_path, 'model', 'exposure_code.pth'))

## define losses
# L1loss = nn.L1Loss()

logging.info('Starting training')
for i in range(start_epoch, start_epoch+num_epochs):

    ## TRAINING
    train_iter = 0
    interm_loss_sum = 0.
    final_loss_sum = 0.
    tv_loss_sum = 0.
    loss_sum = 0.
    psnr_sum = 0.
    for gt_vid in training_generator:   

        gt_vid = gt_vid.cuda()
        if not args.two_bucket:
            b1 = c2b(gt_vid) # (N,1,H,W)
            interm_vid = invNet(b1)  
        else:
            b1, b0 = c2b(gt_vid)
            interm_vid = invNet(torch.cat([b1,b0], dim=1))
        highres_vid = uNet(interm_vid) # (N,16,H,W)
        
        psnr_sum += utils.compute_psnr(highres_vid, gt_vid).item()

        ## LOSSES
        # b1_est = c2b(interm_vid)
        # interm_loss = utils.weighted_L1loss(b1_est, b1)
        # blurred = torch.mean(gt_vid, dim=1, keepdim=True)
        # blurred_est = torch.mean(interm_vid, dim=1, keepdim=True)
        # interm_loss = L1loss(blurred_est, blurred)
        interm_loss = utils.weighted_L1loss(interm_vid, gt_vid)
        interm_loss_sum += interm_loss.item()

        final_loss = utils.weighted_L1loss(highres_vid, gt_vid)
        final_loss_sum += final_loss.item()

        tv_loss = utils.gradx(highres_vid).abs().mean() + utils.grady(highres_vid).abs().mean()
        tv_loss_sum += tv_loss.item()

        loss = final_loss + 0.1*tv_loss + 0.5*interm_loss
        # loss = final_loss + 0.1*tv_loss
        loss_sum += loss.item()

        ## BACKPROP
        print('before backprop', c2b.code[0,0,0,:].data)
        optimizer.zero_grad()
        loss.backward()       
        optimizer.step()
        print('after backprop', c2b.code[0,0,0,:].data)
        if args.mask == 'train':
            c2b.update_mask()
            print('after update', c2b.code[0,0,0,:].data)

        if train_iter % 1000 == 0:
            logging.info('epoch: %3d \t iter: %5d \t loss: %.4f'%(i, train_iter, loss.item()))
            # print(l1_interm_loss.item(), l1_final_loss.item(), tv_loss.item())
        # if (i % 5 == 0) and (train_iter % 500 == 0):
        #     highres_np = highres_vid[0,...].data.cpu().numpy()
        #     interm_np = interm_vid[0,...].data.cpu().numpy()
        #     gt_np = gt_vid[0,...].data.cpu().numpy()
        #     # for frame in range(gt_np.shape[0]):
        #     #     utils.save_image(interm_np[frame,...], 
        #     #                     os.path.join(save_path, 'images', 'interm_%.3d_%.5d_%.2d.png'%(i, train_iter, frame)))
        #     utils.save_gif(interm_np, os.path.join(save_path, 'gifs', 'interm_%.3d_%.5d.gif'%(i, train_iter)))            
        #     utils.save_gif(highres_np, os.path.join(save_path, 'gifs', 'highres_%.3d_%.5d.gif'%(i, train_iter)))
        #     utils.save_gif(gt_np, os.path.join(save_path, 'gifs', 'gt_%.3d_%.5d.gif'%(i, train_iter)))

        train_iter += 1


    logging.info('Total train iterations: %d'%(train_iter))
    logging.info('Finished epoch %3d with loss: %.4f psnr: %.4f'
                %(i, loss_sum/train_iter, psnr_sum/len(training_set)))


    ## dump tensorboard summaries
    logger.scalar_summary(tag='training/loss', value=loss_sum/train_iter, step=i)
    logger.scalar_summary(tag='training/interm_loss', value=interm_loss_sum/train_iter, step=i)
    logger.scalar_summary(tag='training/final_loss', value=final_loss_sum/train_iter, step=i)
    logger.scalar_summary(tag='training/tv_loss', value=tv_loss_sum/train_iter, step=i)
    logger.scalar_summary(tag='training/psnr', value=psnr_sum/len(training_set), step=i)
    logging.info('Dumped tensorboard summaries for epoch %4d'%(i))


    ## VALIDATION
    if ((i+1) % 2 == 0) or ((i+1) == (start_epoch+num_epochs)):        
        logging.info('Starting validation')
        val_iter = 0
        val_loss_sum = 0.
        val_psnr_sum = 0.
        val_ssim_sum = 0.
        invNet.eval()
        uNet.eval()
        c2b.eval()
        with torch.no_grad():
            for gt_vid in validation_generator:
                
                gt_vid = gt_vid.cuda()
                if not args.two_bucket:
                    b1 = c2b(gt_vid) # (N,1,H,W)
                    interm_vid = invNet(b1)   
                else:
                    b1, b0 = c2b(gt_vid)
                    interm_vid = invNet(torch.cat([b1,b0], dim=1))            
                highres_vid = uNet(interm_vid) # (N,9,H,W)

                val_psnr_sum += utils.compute_psnr(highres_vid, gt_vid).item()
                val_ssim_sum += utils.compute_ssim(highres_vid, gt_vid).item()
                
                psnr = utils.compute_psnr(highres_vid, gt_vid).item() / gt_vid.shape[0]
                ssim = utils.compute_ssim(highres_vid, gt_vid).item() / gt_vid.shape[0]

                ## loss
                # b1_est = c2b(interm_vid)
                # interm_loss = utils.weighted_L1loss(b1_est, b1)
                # blurred = torch.mean(gt_vid, dim=1, keepdim=True)
                # blurred_est = torch.mean(interm_vid, dim=1, keepdim=True)
                interm_loss = utils.weighted_L1loss(interm_vid, gt_vid) 
                final_loss = utils.weighted_L1loss(highres_vid, gt_vid)
                tv_loss = utils.gradx(highres_vid).abs().mean() + utils.grady(highres_vid).abs().mean()

                val_loss_sum += (final_loss + 0.1*tv_loss + 0.5*interm_loss).item()
                # val_loss_sum += (final_loss + 0.1*tv_loss).item()

                if val_iter % 1000 == 0:
                    print('In val iter %d'%(val_iter))

                val_iter += 1

        logging.info('Total val iterations: %d'%(val_iter))
        logging.info('Finished validation with loss: %.4f psnr: %.4f ssim: %.4f'
                    %(val_loss_sum/val_iter, val_psnr_sum/len(validation_set), val_ssim_sum/len(validation_set)))

        scheduler.step(val_loss_sum/val_iter)
        invNet.train()
        uNet.train()
        
        ## dump tensorboard summaries
        logger.scalar_summary(tag='validation/loss', value=val_loss_sum/val_iter, step=i)
        logger.scalar_summary(tag='validation/psnr', value=val_psnr_sum/len(validation_set), step=i)
        logger.scalar_summary(tag='validation/ssim', value=val_ssim_sum/len(validation_set), step=i)

    
    ## CHECKPOINT
    if ((i+1) % 10 == 0) or ((i+1) == (start_epoch+num_epochs)):
        utils.save_checkpoint(state={'epoch': i, 
                                    'invnet_state_dict': invNet.state_dict(),
                                    'unet_state_dict': uNet.state_dict(),
                                    'c2b_state_dict': c2b.state_dict(),
                                    'opt_state_dict': optimizer.state_dict()},
                            save_path=os.path.join(save_path, 'model'),
                            filename='model_%.6d.pth'%(i))
        torch.save(c2b.code, os.path.join(save_path, 'model', 'exposure_code.pth'))
        logging.info('Saved checkpoint for epoch {}'.format(i))

logger.writer.flush()
logging.info('Finished training')