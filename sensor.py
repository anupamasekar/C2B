'''
-----------------------------------
DEFINE C2B SENSOR FOR BUCKET IMAGES
-----------------------------------
'''
import torch
import torch.nn as nn
import numpy as np
import scipy.io


class C2B(nn.Module):
    
    def __init__(self, block_size, sub_frames, patch_size):
        super(C2B, self).__init__()

        ## random code 16x8x8
        code = torch.empty(1, sub_frames, block_size, block_size).cuda()
        code = code.bernoulli_(p=0.5)

        self.block_size = block_size
        self.patch_size = patch_size
        # self.sub_frames = sub_frames
        self.code = nn.Parameter(code, requires_grad=False)
        # self.code_repeat = code.repeat(1, 1, patch_size//block_size, patch_size//block_size)


    def forward(self, x):

        # code_repeat_comp = 1 - self.code_repeat
        code_repeat = self.code.repeat(1, 1, self.patch_size//self.block_size, self.patch_size//self.block_size)
        b1 = torch.sum(code_repeat*x, dim=1, keepdim=True) / torch.sum(code_repeat, dim=1, keepdim=True)
        # b0 = torch.sum(code_repeat_comp*x, dim=1, keepdim=True) / torch.sum(code_repeat_comp, dim=1, keepdim=True)
        # blurred = torch.mean(x, dim=1, keepdim=True)
        return b1 # (N,1,H,W)