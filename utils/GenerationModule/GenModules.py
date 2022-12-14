import torch
from utils.Modules import UpBlock, DownBlock

class Encoder_gen(torch.nn.Module):
    def __init__(self, input_dim, layer_xp, num_layers, max_channel=256, downscale=2):
        super().__init__()
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, min(max_channel, layer_xp), kernel_size=7, padding=3),
            torch.nn.BatchNorm2d(min(max_channel, layer_xp))
        )
        layers_list = []
        layers_dim = []
        for i in range(0, num_layers + 1):
            layers_dim.append(min(max_channel, (2**i)*layer_xp))
        for i in range(len(layers_dim) - 1):
            layers_list.append(DownBlock(layers_dim[i], layers_dim[i + 1], downscale=downscale))
        self.block_2 = torch.nn.ModuleList(layers_list)

    def forward(self, x):
        out = self.block_1(x)
        for layer in self.block_2:
            out = layer(out)
        return out

class Decoder_gen(torch.nn.Module):
    def __init__(self, input_dim, layer_xp, num_layers, num_reslayers, max_channel=256, upscale=2):
        super().__init__()
        layers_list = []
        num_channelres = min(max_channel, layer_xp*(2**num_layers))
        for i in range(num_reslayers):
            layers_list.append(torch.nn.Sequential(
                torch.nn.BatchNorm2d(num_channelres),
                torch.nn.ReLU(),
                torch.nn.Conv2d(num_channelres, num_channelres, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(num_channelres),
                torch.nn.ReLU(),
                torch.nn.Conv2d(num_channelres, num_channelres, kernel_size=3, padding=1)
            ))
        self.block_1 = torch.nn.ModuleList(layers_list)
        layers_list = []
        layers_dim = []
        for i in range(num_layers + 1):
            layers_dim.append(min(max_channel, (2**(num_layers - i))*layer_xp))
        for i in range(len(layers_dim) - 1):
            layers_list.append(UpBlock(layers_dim[i], layers_dim[i + 1], upscale=upscale))
        self.block_2 = torch.nn.ModuleList(layers_list)
        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(min(max_channel, layer_xp), input_dim, kernel_size=7, padding=3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x
        for layer in self.block_1:
            out = layer(out) + out
        for layer in self.block_2:
            out = layer(out)
        out = self.block_3(out)
        return out

class Encoder_gen_(torch.nn.Module):
    def __init__(self, input_dim, layer_xp, num_layers, max_channel=256, downscale=2):
        super().__init__()
        layers_list = []
        layers_dim = [input_dim]
        for i in range(1, num_layers + 1):
            layers_dim.append(min(max_channel, (2**i)*layer_xp))
        for i in range(len(layers_dim) - 1):
            layers_list.append(DownBlock(layers_dim[i], layers_dim[i + 1], downscale=downscale))
        self.block = torch.nn.ModuleList(layers_list)

    def forward(self, x):
        outs = [x]
        for layer in self.block:
            outs.append(layer(outs[-1]))
        return outs

class Decoder_gen_(torch.nn.Module):
    def __init__(self, input_dim, heatmap_dim, layer_xp, num_layers, num_reslayers, max_channel=256, upscale=2):
        super().__init__()
        layers_list = []
        layers_dim = [min(max_channel, (2**num_layers)*layer_xp)//2]
        for i in range(1, num_layers + 1):
            layers_dim.append(min(max_channel, (2**(num_layers - i))*layer_xp))
        for i in range(len(layers_dim) - 1):
            layers_list.append(UpBlock(2*layers_dim[i] + heatmap_dim, layers_dim[i + 1], upscale=upscale))
        self.block_1 = torch.nn.ModuleList(layers_list)
        reslayers_list = []
        num_channelres = layer_xp + input_dim + heatmap_dim
        for i in range(num_reslayers):
            reslayers_list.append(torch.nn.Sequential(
                torch.nn.BatchNorm2d(num_channelres),
                torch.nn.ReLU(),
                torch.nn.Conv2d(num_channelres, num_channelres, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(num_channelres),
                torch.nn.ReLU(),
                torch.nn.Conv2d(num_channelres, num_channelres, kernel_size=3, padding=1)
            ))
        self.block_2 = torch.nn.ModuleList(reslayers_list)
        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(num_channelres, input_dim, kernel_size=7, padding=3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.pop()
        for layer in self.block_1:
            out = layer(out)
            out = torch.cat((out, x.pop()), dim=1)
        for layer in self.block_2:
            out = layer(out)
        out = self.block_3(out)
        return out