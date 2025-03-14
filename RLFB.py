import torch.nn as nn
from conv_layer import conv_layer
from activation import activation
from CBAM import CBAM

class RLFB(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=None):
        super(RLFB, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.cbam = CBAM(out_channels)
        self.act = activation('silu')
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x  # Skip connection
        out = self.c1_r(x)
        out = self.act(out)
        out = self.dropout(out)

        out = self.c2_r(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.c3_r(out)
        out = self.act(out)
        out = self.dropout(out)

        out = out + residual  # Add skip connection
        out = self.cbam(self.c5(out))
        return out