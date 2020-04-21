'''
-----------------------
DEFINE HELPER FUNCTIONS
-----------------------
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import scipy.misc
from matplotlib import cm
import matplotlib.pyplot as plt
from math import exp


def gradx(tensor):
    return tensor[:,:,:,2:] - tensor[:,:,:,:-2]

def grady(tensor):
    return tensor[:,:,2:,:] - tensor[:,:,:-2,:]


def gradxy(tensor):
    # grad_filter = np.zeros((1,1,3,3))
    filt = np.array([[0.,-1.,0.],[0.,2.,-1.],[0.,0.,0.]])/2.
    # grad_filter[0,0,...] = filt
    grad_filter = torch.cuda.FloatTensor(filt).unsqueeze(0).unsqueeze(0)
    # grad_filter = grad_filter.repeat(tensor.shape[1], 1, 1, 1)
    # grad = F.conv2d(tensor, grad_filter, padding=1, stride=1, groups=tensor.shape[1])
    N,S,H,W = tensor.size()
    tensor = tensor.view(N*S,1,H,W)
    grad = F.conv2d(tensor, grad_filter, padding=1, stride=1)
    grad = grad.view(N,S,H,W)
    return grad


def weighted_L2loss(pred, target):
    grad_weight = gradxy(target).abs() + 1.
    loss = ((pred - target)*grad_weight).pow(2).mean()
    return loss


def weighted_L1loss(pred, target):
    grad_weight = gradxy(target).abs() + 1.
    loss = ((pred-target)*grad_weight).abs().mean()
    return loss


def solve_constraints(b1 ,b0, code, normalize=False):
    # code (1,9,3,3)
    code_size = code.shape[-1]

    vec_filter = np.eye(code_size**2) # 9x9
    vec_filter = vec_filter.reshape([code_size**2,code_size,code_size]) # 9x3x3
    vec_filter = vec_filter[:,None,:,:] # 9x1x3x3
    vec_filter = torch.cuda.FloatTensor(vec_filter) # (9,1,3,3)

    N,_,H,W = b0.size() # (N,1,H,W)
    # reshaping b1 and b0
    vec_b1 = F.conv2d(b1, vec_filter, stride=code_size) # (N,9,H/3,W/3)
    vec_b0 = F.conv2d(b0, vec_filter, stride=code_size) # (N,9,H/3,W/3)
    b1_b0 = torch.cat([vec_b1,vec_b0], dim=1) # (N,18,H/3,W/3)
    b1_b0 = b1_b0.view(N,2*code_size**2,-1) # (N,18,H*W/9)
    b1_b0 = b1_b0.unsqueeze(1) # (N,1,18,H*W/9)
    
    # code_mat = code.reshape([1,code_size**2,-1]).squeeze(0) # (9,9)
    code_mat = code.contiguous().view(code_size**2, code_size**2).transpose(0,1)
    comp_mat = 1. - code_mat
    code_concat = torch.cat([code_mat,comp_mat], dim=0) # (18,9)
    code_concat = code_concat / torch.sum(code_concat, dim=1, keepdim=True)
    # code_concat = code_concat / code.shape[1]
    code_pinv = torch.pinverse(code_concat, rcond=1e-3) # (9,18)
    # print(code_pinv.min(), code_pinv.max())
    inverse_filter = code_pinv.unsqueeze(1).unsqueeze(3) # (9,1,18,1)
    
    lowRes_vid = F.conv2d(b1_b0, inverse_filter, stride=1) # (N,9,1,H*W/9)
    lowRes_vid = lowRes_vid.view(N,code_size**2,H//code_size,W//code_size) # (N,9,H/3,W/3)

    # normalize
    if normalize:
        for i in range(N):
            lowRes_vid[i] = (lowRes_vid[i]-torch.min(lowRes_vid[i]))/(torch.max(lowRes_vid[i])-torch.min(lowRes_vid[i]))
    else:
        lowRes_vid = lowRes_vid.clamp(min=0, max=1)
    # print(torch.max(lowRes_vid), torch.min(lowRes_vid))
    return lowRes_vid


def single_bucket_to_lowRes(b1, code, normalize=False):
    # code (1,9,3,3)
    code_size = code.shape[-1]

    vec_filter = np.eye(code_size**2) # 9x9
    vec_filter = vec_filter.reshape([code_size**2,code_size,code_size]) # 9x3x3
    vec_filter = vec_filter[:,None,:,:] # 9x1x3x3
    vec_filter = torch.cuda.FloatTensor(vec_filter) # (9,1,3,3)

    N,_,H,W = b1.size() # (N,1,H,W)
    # reshaping b1
    vec_b1 = F.conv2d(b1, vec_filter, stride=code_size) # (N,9,H/3,W/3)
    vec_b1 = vec_b1.view(N,code_size**2,-1) # (N,9,H*W/9)
    vec_b1 = vec_b1.unsqueeze(1) # (N,1,9,H*W/9)
    
    # code_mat = code.reshape([1,code_size**2,-1]).squeeze(0) # (9,9)
    code_mat = code.contiguous().view(code_size**2, code_size**2).transpose(0,1)
    code_mat = code_mat / torch.sum(code_mat, dim=1, keepdim=True)
    # code_mat = code_mat / code.shape[1]
    code_pinv = torch.pinverse(code_mat) # (9,9)
    inverse_filter = code_pinv.unsqueeze(1).unsqueeze(3) # (9,1,9,1)
    
    lowRes_vid = F.conv2d(vec_b1, inverse_filter, stride=1) # (N,9,1,H*W/9)
    lowRes_vid = lowRes_vid.view(N,code_size**2,H//code_size,W//code_size) # (N,9,H/3,W/3)

    # normalize
    if normalize:
        for i in range(N):
            lowRes_vid[i] = (lowRes_vid[i]-torch.min(lowRes_vid[i]))/(torch.max(lowRes_vid[i])-torch.min(lowRes_vid[i]))
    else:
        lowRes_vid = lowRes_vid.clamp(min=0, max=1)
    # print(torch.max(lowRes_vid), torch.min(lowRes_vid))
    return lowRes_vid


def inverse_pixel_shuffle(img, downscale_factor):
    # img (N,1,H,W)
    vec_filter = np.eye(downscale_factor**2)
    vec_filter = vec_filter.reshape([downscale_factor**2,downscale_factor,downscale_factor])
    vec_filter = torch.cuda.FloatTensor(vec_filter[:,None,:,:])

    shuffled = F.conv2d(img, vec_filter, stride=downscale_factor) # (N,9,H/3,W/3)
    return shuffled
    # output (N,9,H/3,W/3)


# def coded_pixel_shuffle(img, code):
#     # input (N,1,H,W) (1,9,3,3)
#     vec_filter = code.squeeze(0).unsqueeze(1)
#     shuffled = F.conv2d(img, vec_filter, stride=code.shape[-1]) / code.sum(dim=[2,3], keepdim=True)
#     return shuffled
#     # output (N,9,H/3,W/3)


# def get_patches(vid, size):
#     # input video tensor (N,9,H,W)
#     # return list of patches of size (size x size)
#     patched_vid = vid.unfold(2,size,size).unfold(3,size,size) # (N,9,3,5,240,240)
#     patches = []
#     for i in range(patched_vid.shape[2]):
#         for j in range(patched_vid.shape[3]):
#             patches.append(patched_vid[:,:,i,j,:,:])
#     return patches


def compute_psnr(vid1, vid2):
    # (N,9,H,W)
    assert vid1.shape == vid2.shape
    mse = ((vid1 - vid2)**2).mean(dim=[1,2,3])
    psnr = (20.*torch.log10(1./torch.sqrt(mse))).sum()
    return psnr


def compute_ssim(vid1, vid2):
    # imput video tensors (N,9,H,W)
    # return ssim value sum over batch
    assert vid1.shape == vid2.shape
    sigma = 1.5
    window_size = 11
    channel = vid1.shape[1]

    gauss = torch.cuda.FloatTensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss/gauss.sum()

    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.cuda()
   
    if torch.max(vid1) > 128:
        max_val = 255
    else:
        max_val = 1

    if torch.min(vid1) < -0.5:
        min_val = -1
    else:
        min_val = 0
    L = max_val - min_val

    padd = 0

    mu1 = F.conv2d(vid1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(vid2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(vid1 * vid1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(vid2 * vid2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(vid1 * vid2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # ret = ssim_map.mean(1).mean(1).mean(1)
    ret = ssim_map.mean(dim=[1,2,3]).sum()
    # print(ret.shape)
    return ret


# def adjust_learning_rate(optimizer, i, lr):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def create_dirs(save_path):
    """create 
       save_path/logs/
       save_path/model/
    """
    if not os.path.exists(os.path.join(save_path)):
        os.mkdir(os.path.join(save_path))
    if not os.path.exists(os.path.join(save_path,'images')):
        os.mkdir(os.path.join(save_path,'images'))
    if not os.path.exists(os.path.join(save_path,'gifs')):
        os.mkdir(os.path.join(save_path,'gifs'))
    if not os.path.exists(os.path.join(save_path,'logs')):
        os.mkdir(os.path.join(save_path,'logs'))
    if not os.path.exists(os.path.join(save_path,'model')):
        os.mkdir(os.path.join(save_path,'model'))
    return


def save_checkpoint(state, save_path, filename):
    torch.save(state, os.path.join(save_path, filename))
    return


def read_image(path, height_img, width_img):
    # img = scipy.misc.imread(path, mode='L') # grayscale
    # img = scipy.misc.imresize(img, size=(height_img,width_img))
    img = Image.open(path).convert(mode='L')
    img = img.resize((width_img,height_img), Image.ANTIALIAS)
    img = np.array(img)
    img = img/255.
    return img


def save_image(img, path, normalize=False):
    # img = scipy.misc.toimage(img, cmin=np.amin(img), cmax=np.amax(img))
    # print(np.amin(img), np.amax(img))
    if normalize:
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img = img.clip(0,1)
    img = Image.fromarray((img*255.).astype('uint8'))
    img.save(path)
    return


def save_gif(arr, path, normalize=False):
    # (9,H,W)
    frames = []
    for sub_frame in range(arr.shape[0]):
        img = arr[sub_frame,...]
        # print(np.amax(img), np.amin(img))
        if normalize:
            img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        img = img.clip(0,1)
        img = Image.fromarray((img*255.).astype('uint8'))
        frames.append(img)
    frame1 = frames[0]
    frame1.save(path, save_all=True, append_images=frames[1:], duration=500, loop=0)
    return