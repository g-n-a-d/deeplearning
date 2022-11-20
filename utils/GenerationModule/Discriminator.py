import torch
from utils.Modules import DownBlock

class Discriminator(torch.nn.Module):
    def __init__(self, num_channels=3, layer_xp=64, num_layers=4, max_channel=512, sn=False):
        super().__init__()
        layers_list = []
        layers_dim = [num_channels]
        for i in range(1, num_layers + 1):
            layers_dim.append(min(max_channel, (2**i)*layer_xp))
        for i in range(len(layers_dim) - 1):
            layers_list.append(DownBlock(layers_dim[i], layers_dim[i + 1], kernel_size=4, padding=0))
        # for i in range(num_layers):
        #     down_blocks.append(
        #         DownBlock(num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i)),
        #                     min(max_features, block_expansion * (2 ** (i + 1))),
        #                     norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.block = torch.nn.ModuleList(layers_list)
        self.discriminator = torch.nn.Conv2d(self.down_blocks[-1].conv.out_channels, 1, kernel_size=1)
        if sn:
            self.conv = torch.nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        outs = [x]
        for layer in self.block:
            outs.append(layer(outs[-1]))
        pred = self.discriminator(outs[-1])
        return outs, pred