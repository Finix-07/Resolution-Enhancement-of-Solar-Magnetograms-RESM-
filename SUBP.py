import torch
import torch.nn as nn
from RLFB import RLFB
from scipy.signal import wiener

class SubPixelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(SubPixelConvBlock, self).__init__()
        # Sub-Pixel Convolutional Layer
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        # Sub-pixel reorganization
        self.subpixel = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.subpixel(x)
        # x= self.apply_wiener_filter(x)

        return self.relu(x)

    # def apply_wiener_filter(self, x):
    #     x_np = x.detach().cpu().numpy()
    #     filtered_np = wiener(x_np,mysize=None, noise=None)
    #     filtered_tensor = torch.tensor(filtered_np,dtype=x.dtype).to(x.device)
    #     return filtered_tensor