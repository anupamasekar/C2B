'''
------------------------------------------------------
RUN INFERENCE ON SAMPLE TEST SEQUENCES - ATTENTION NET
------------------------------------------------------
'''
import os 
import numpy as np
import torch
import torch.nn.functional as F
import logging
import glob
import argparse
from skimage.measure import compare_psnr, compare_ssim
from natsort import natsorted
# from PIL import Image

# from sensor import C2B
from unet_v3 import UNet
from shift_var_conv import ShiftVarConv2D
import utils


## set random seed
torch.manual_seed(0)
np.random.seed(0)


## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, required=True, help='expt name for inference')
parser.add_argument('--export', type=str, required=True, help='export dir name to dump results')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint file name')
parser.add_argument('--blocksize', type=int, default=8, help='tile size for code default 3x3')
parser.add_argument('--subframes', type=int, default=16, help='num sub frames')
parser.add_argument('--gpu', type=str, required=True, help='GPU ID')
args = parser.parse_args()
# print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

expt_name = os.path.join('/data/prasan/anupama/', args.expt)
if not os.path.exists(expt_name):
    raise ValueError('Give valid expt name')

save_path = os.path.join(expt_name, args.export)
if not os.path.exists(save_path):
  os.mkdir(save_path)
  os.mkdir(os.path.join(save_path, 'frames'))


## configure runtime logging
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(save_path, 'logfile.log'), 
                    format='%(asctime)s - %(message)s', 
                    filemode='w')
# logger=logging.getLogger()#.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger().addHandler(console)
logging.info(args)


## input params
## GoPro 720x1200 for 9x3x3 code
## GoPro 704x1280 for 16x4x4 code
## ECCV 256x256 for 16x8x8 code
input_params = {'height': 256,
                'width': 256}



## loading test sequences
## GoPro videos
# data_path = '/media/data/prasan/C2B/anupama/test_sequences_16'
# image_paths = sorted(glob.glob(data_path+'/seq*/*.png'))
## ECCV videos
data_path = '/data/prasan/anupama/dataset/eccv18/test_video_14' 
# image_paths = sorted(glob.glob(data_path+'/*/*.png'), key=lambda i: int((os.path.basename(i)[:-4]).split('_')[-1]))
# image_paths = natsorted(glob.glob(data_path+'/*/*_[1-9].png')+glob.glob(data_path+'/*/video[1-9].png'))
image_paths = natsorted(glob.glob(data_path+'/*/*.png'))
logging.info('Test images found: %d'%(len(image_paths)))

# assert len(image_paths) == 15*args.subframes # for GoPro test
assert len(image_paths) == 14*args.subframes # for eccv comparison


invNet = ShiftVarConv2D(out_channels=args.subframes, block_size=args.blocksize).cuda()
uNet = UNet(in_channel=args.subframes, out_channel=args.subframes, instance_norm=False).cuda()


## loading checkpoint
ckpt = torch.load(os.path.join(expt_name, 'model', args.ckpt))
invNet.load_state_dict(ckpt['invnet_state_dict'])
invNet.eval()
uNet.load_state_dict(ckpt['unet_state_dict'])
uNet.eval()

c2b_code = ckpt['c2b_state_dict']['code']
code_repeat = c2b_code.repeat(1, 1, input_params['height']//args.blocksize, input_params['width']//args.blocksize)
# code_repeat = c2b_code.repeat(1,1,128//8,128//8)

logging.info('Starting inference')
psnr_sum = 0.
ssim_sum = 0.
with torch.no_grad():
    # print('\n\nExposure code:\n', c2b.code, '\n\n')
    for seq in range(len(image_paths)//args.subframes):

        vid = np.zeros((args.subframes, input_params['height'], input_params['width'])) # (16,H,W)
        for sub_frame in range(args.subframes):
            img = utils.read_image(image_paths[seq*args.subframes+sub_frame], input_params['height'], input_params['width'])
            vid[sub_frame] = img
        vid = torch.cuda.FloatTensor(vid).unsqueeze(0) # (1,16,H,W)

        b1 = torch.sum(code_repeat*vid, dim=1, keepdim=True) / torch.sum(code_repeat, dim=1, keepdim=True)
        interm_vid = invNet(b1)        
        highres_vid = uNet(interm_vid) # (1,16,H,W)
        assert highres_vid.shape == vid.shape
        
        highres_vid = highres_vid.clamp(0,1)
        # highres_vid = (highres_vid - torch.min(highres_vid)) / (torch.max(highres_vid) - torch.min(highres_vid))
        # for sf in range(highres_vid.shape[1]):
        #     highres_vid[:,sf,:,:] = (highres_vid[:,sf,:,:]-torch.min(highres_vid[:,sf,:,:]))\
        #                                 /(torch.max(highres_vid[:,sf,:,:])-torch.min(highres_vid[:,sf,:,:]))

        ## converting tensors to numpy arrays
        b1_np = b1.squeeze().data.cpu().numpy() # (H,W)
        vid_np = vid.squeeze().data.cpu().numpy() # (9,H,W)
        interm_np = interm_vid.squeeze().data.cpu().numpy()
        highres_np = highres_vid.squeeze().data.cpu().numpy() # (9,H,W)
        code_np = c2b_code.squeeze().data.cpu().numpy()

        ## psnr
        # psnr = utils.compute_psnr(highres_vid, vid).item()
        psnr = compare_psnr(highres_np, vid_np)
        psnr_sum += psnr

        ## ssim
        # ssim = utils.compute_ssim(highres_vid, vid).item()
        ssim = 0.
        for sf in range(vid_np.shape[0]):
            ssim += compare_ssim(highres_np[sf,:,:], vid_np[sf,:,:], gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        ssim = ssim / vid_np.shape[0]
        ssim_sum += ssim

        logging.info('Seq %.2d PSNR: %.2f SSIM: %.3f'%(seq+1, psnr, ssim))

        ## saving images and gifs
        utils.save_image(b1_np, os.path.join(save_path, 'seq_%.2d_coded.png'%(seq+1)))
        utils.save_gif(vid_np, os.path.join(save_path, 'seq_%.2d_groundTruth.gif'%(seq+1)))
        utils.save_gif(interm_np, os.path.join(save_path, 'seq_%.2d_intermediate.gif'%(seq+1)))
        utils.save_gif(highres_np, os.path.join(save_path, 'seq_%.2d_highRes.gif'%(seq+1)))

        # for sub_frame in range(vid_np.shape[0]):
        #     utils.save_image(img=highres_np[sub_frame,:,:], path=os.path.join(save_path, 'frames', 'seq_%.2d_highRes_%.2d.png'%(seq+1, sub_frame+1)))
            # utils.save_image(img=vid_np[sub_frame,:,:], path=os.path.join(output_path, 'frames', 'seq_%.2d_gt_%.1d.png'%(seq+1, sub_frame+1)))

    # logging.info('Total PSNR: %.4f'%(psnr_sum))
    logging.info('Average PSNR: %.2f'%(psnr_sum/(len(image_paths)//args.subframes)))
    logging.info('Average SSIM: %.3f'%(ssim_sum/(len(image_paths)//args.subframes)))
    logging.info('Saved images and gifs for all sequences')
    
    code_np = c2b_code.squeeze().data.cpu().numpy()
    np.save(os.path.join(save_path, 'exposure_code'), code_np)
    code_vis = np.repeat(np.repeat(code_np, 50, axis=2), 50, axis=1)
    utils.save_gif(code_vis, os.path.join(save_path, 'exposure_code.gif'))


logging.info('Finished inference')