import torch
import torch.nn as nn
import torch.nn.functional as F



class ShiftVarConv2D(nn.Module):


    def window_shuffle(self, img, kernel_size):
    ## input (N,1,H,W)
        vec_filter = torch.eye(kernel_size**2).cuda()
        vec_filter = vec_filter.view(kernel_size**2, kernel_size, kernel_size).unsqueeze(1)

        shuffled = F.conv2d(img, vec_filter, stride=1, padding=(kernel_size-1)//2) # (N,k*k,H,W)
        return shuffled
    ## output (N,k*k,H,W)


    def reverse_pixel_shuffle(self, img, kernel_size):
    ## input (N,1,H,W)
        vec_filter = torch.eye(kernel_size**2).cuda()
        vec_filter = vec_filter.view(kernel_size**2, kernel_size, kernel_size).unsqueeze(1)

        shuffled = F.conv2d(img, vec_filter, stride=kernel_size, padding=0) # (N,k*k,H/k,W/k)
        return shuffled
    ## output (N,k*k,H/k,W/k)


    def conv_layer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(self.window**2,1,1), stride=1, padding=0, groups=in_channels))
        # layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)


    def __init__(self, out_channels=16, block_size=8):
        super(ShiftVarConv2D, self).__init__()

        self.block_size = block_size
        self.sub_frames = out_channels
        self.window = 3

        self.inverse_layer = self.conv_layer(in_channels=block_size**2, out_channels=out_channels*(block_size**2))
        self.ps_layer = nn.PixelShuffle(upscale_factor=block_size)
        
        # print(self.inverse_layer._modules)
        with torch.no_grad():
            init_weight = torch.empty(16,1,9,1,1)
            nn.init.kaiming_normal_(init_weight)
            self.inverse_layer._modules['0'].weight.data.copy_(init_weight.repeat(64,1,1,1,1))
            nn.init.zeros_(self.inverse_layer._modules['0'].bias)
        
        
    def forward(self, coded):
    ## input (N,1,H,W)
        coded_inp = self.window_shuffle(coded, kernel_size=self.window) # (N,9,H,W)
        N = coded_inp.size(0)
        shuffled = []
        for i in range(coded_inp.shape[1]):
            coded_shuff = self.reverse_pixel_shuffle(coded_inp[:,i:i+1,:,:], kernel_size=self.block_size)
            shuffled.append(coded_shuff) # (N,64,H/8,W/8)
        coded_inp = torch.stack(shuffled, dim=2) # (N,64,9,H/8,W/8)
        
        inverse_out = self.inverse_layer(coded_inp) # (N,16*64,1,H/8,W/8)
        inverse_out = torch.reshape(inverse_out,[N,64,16,16,16])
        inverse_out = torch.transpose(inverse_out,1,2)
        inverse_out = torch.reshape(inverse_out,[N,64*16,16,16])
        final_out = self.ps_layer(inverse_out) # (N,16,H,W)

        return final_out
    ## output (N,16,H,W)