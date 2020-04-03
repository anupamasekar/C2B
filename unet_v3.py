'''
-------------------------------------------------
DEFINE THE UNET ARCHITECTURE FOR VIDEO REGRESSION
-------------------------------------------------
made batchnorm optional
changed convtranspose layers to upsample+conv layers
removed pixelshuffle layer in final block
added 1x1 conv layer instead
added residual connection between input and output
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):

    def contracting_block(self, in_channels, out_channels, kernel_size=3, batch_norm=False):
        # input (N,in_channels,Hi,Wi)
        layers = []
        
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
        						kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))     
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
        						kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)
        # output (N,out_channels,Hi,Wi)
    

    def bottleneck_block(self, in_channels, mid_channels, out_channels, kernel_size=3, batch_norm=False):
        # input (N,in_channels,Hi,Wi)
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, 
        						kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))                  
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, 
        						kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        # layers.append(nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
        #                                  kernel_size=kernel_size, stride=2, padding=1, output_padding=1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=1, padding=1))
        
        return nn.Sequential(*layers)
        # output (N,out_channels,2*Hi,2*Wi)


    def expansive_block(self, in_channels, mid_channels, out_channels, kernel_size=3, batch_norm=False):
        # input (N,in_channels,Hi,Wi)
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, 
                                kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, 
                                kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        # layers.append(nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, 
        #                                  kernel_size=kernel_size, stride=2, padding=1, output_padding=1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=1, padding=1))

        return nn.Sequential(*layers)
        # output (N,out_channels,2*Hi,2*Wi)

    
    def final_block(self, in_channels, mid_channels, out_channels, kernel_size=3, batch_norm=False):
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, 
                                kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))                    
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, 
                                kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))                    
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, 
                                kernel_size=1, stride=1, padding=0))                
        return nn.Sequential(*layers)
    
    
    def __init__(self, in_channel, out_channel, batch_norm=False):
        super(UNet, self).__init__()

        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64, batch_norm=batch_norm)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(in_channels=64, out_channels=128, batch_norm=batch_norm)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(in_channels=128, out_channels=256, batch_norm=batch_norm)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.bottleneck = self.bottleneck_block(in_channels=256, mid_channels=512, out_channels=256, batch_norm=batch_norm)
        # Decode
        self.conv_decode3 = self.expansive_block(in_channels=512, mid_channels=256, out_channels=128, batch_norm=batch_norm)
        self.conv_decode2 = self.expansive_block(in_channels=256, mid_channels=128, out_channels=64, batch_norm=batch_norm)
        self.final_layer = self.final_block(in_channels=128, mid_channels=64, out_channels=out_channel, batch_norm=batch_norm)
                
    
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), dim=1)
    
    
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)

        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)

        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)

        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)

        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_out = self.final_layer(decode_block1) + x

        return final_out