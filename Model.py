"""
A U-net Generator.
"""
from torch import nn
from torch.nn import init

# sequential generator model:
import torch
import torch.nn as nn
import torch.nn.functional as F

class gen_double_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder part
        self.encoder1 = gen_double_conv_block(1, 64)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.encoder2 = gen_double_conv_block(64, 128)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.encoder3 = gen_double_conv_block(128, 256)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.encoder4 = gen_double_conv_block(256, 512)
        self.pool4 = nn.MaxPool2d((2, 2))
        self.bottleneck = gen_double_conv_block(512, 1024)

        # Decoder part
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = gen_double_conv_block(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = gen_double_conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = gen_double_conv_block(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = gen_double_conv_block(128, 64)

        self.outputs = nn.Conv2d(64, 2, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)

        # Decoding path
        d1 = self.upconv1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.decoder1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.decoder2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d4 = self.upconv4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.decoder4(d4)

        outputs = self.outputs(d4)
        return outputs

# class double_conv_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.sequence = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, inputs):
#         return self.sequence(inputs)
#
#
# class encoder_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = double_conv_block(in_channels, out_channels)
#         self.maxpool = nn.MaxPool2d((2, 2))
#
#     def forward(self, inputs):
#         out = self.conv(inputs)
#         p = self.maxpool(out)
#         return out, p
#
#
# class decoder_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
#         self.conv = double_conv_block(2*out_channels, out_channels)
#
#     def forward(self, inputs, skip):
#         out = self.up(inputs)
#         out = torch.cat([out, skip], dim=1)
#         out = self.conv(out)
#         return out
#
#
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#          # Encoder blocks
#         self.e1 = encoder_block(1, 64)
#         self.e2 = encoder_block(64, 128)
#         self.e3 = encoder_block(128, 256)
#         self.e4 = encoder_block(256, 512)
#
#         # Bottleneck
#         self.b = double_conv_block(512, 1024)
#
#         # Decoder blocks
#         self.d1 = decoder_block(1024, 512)
#         self.d2 = decoder_block(512, 256)
#         self.d3 = decoder_block(256, 128)
#         self.d4 = decoder_block(128, 64)
#
#         self.outputs = nn.Conv2d(64, 2, kernel_size=1, padding=0)
#
#     def forward(self, inputs):
#         # Encoder blocks
#         s1, p1 = self.e1(inputs)
#         s2, p2 = self.e2(p1)
#         s3, p3 = self.e3(p2)
#         s4, p4 = self.e4(p3)
#
#         # Bottleneck
#         b = self.b(p4)
#
#         # Decoder blocks
#         d1 = self.d1(b, s4)
#         d2 = self.d2(d1, s3)
#         d3 = self.d3(d2, s2)
#         d4 = self.d4(d3, s1)
#
#         outputs = self.outputs(d4)
#         return outputs

"""
The critic is based upon the discriminator used in Pix2Pix paper - a "Patch Discriminator".
"""

class conv_block(nn.Module):
    """
    Each block is composed of 3 elements:
    Convolutional layer, instance norm and leaky relu.
    If norm=True, then instance norm is used.
    """

    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.ins_norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

        self.norm = norm

    def forward(self, inputs):
        if self.norm:
            return self.lrelu(self.ins_norm(self.conv(inputs)))
        else:
            return self.lrelu(self.conv(inputs))

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.block1 = conv_block(3, 32, norm=False)
        self.block2 = conv_block(32, 64, norm=True)
        self.block3 = conv_block(64, 128, norm=True)
        self.block4 = conv_block(128, 256, norm=True)
        self.block5 = conv_block(256, 512, norm=True)

        self.output = nn.Conv2d(512, 1, kernel_size=4, bias=False)

        # Initialize parameters with normal distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        output = self.output(x)
        return output

