import torch
import torch.nn as nn

class S1Net(nn.Module):
    def __init__(self):
        super(S1Net, self).__init__()

        self.in_dim = 3
        self.out_dim = 64
        self.final_out_dim = 2

        self.down_layers = nn.ModuleList([
            ConvResidualConv(self.in_dim, self.out_dim),
            ConvResidualConv(self.out_dim, self.out_dim * 2),
            ConvResidualConv(self.out_dim * 2, self.out_dim * 4),
            ConvResidualConv(self.out_dim * 4, self.out_dim * 8)
        ])

        self.bridge = ConvResidualConv(self.out_dim * 8, self.out_dim * 16)

        self.up_layers = nn.ModuleList([
            ConvTransBlock(self.out_dim * 16, self.out_dim * 8),
            ConvResidualConv(self.out_dim * 8, self.out_dim * 8),
            ConvTransBlock(self.out_dim * 8, self.out_dim * 4),
            ConvResidualConv(self.out_dim * 4, self.out_dim * 4),
            ConvTransBlock(self.out_dim * 4, self.out_dim * 2),
            ConvResidualConv(self.out_dim * 2, self.out_dim * 2),
            ConvTransBlock(self.out_dim * 2, self.out_dim),
            ConvResidualConv(self.out_dim, self.out_dim)
        ])

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Tanh()

    def forward(self, input):
        down_outputs = []
        for down_layer in self.down_layers:
            input = down_layer(input)
            down_outputs.append(input)
            input = nn.MaxPool2d(kernel_size=2, stride=2)(input)

        input = self.bridge(input)

        for i, up_layer in enumerate(self.up_layers):
            input = up_layer(input)
            input = input + down_outputs[-(i+1)]

        out = self.out(input)
        out = self.out_2(out)
        return out

class ConvResidualConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvResidualConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv_1 = nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = ConvBlock(self.out_dim, self.out_dim)
        self.conv_3 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.act_fn = nn.ReLU()

    def forward(self, input):
        conv = self.conv(input)
        out = self.act_fn(conv)
        return out

class ConvTransBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvTransBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.act_fn = nn.ReLU()

    def forward(self, input):
        deconv = self.deconv(input)
        out = self.act_fn(deconv)
        return out
