import torch
from utils.Modules import UpBlock, DownBlock, meshgrid

class Encoder_kp(torch.nn.Module):
    def __init__(self, input_dim, layer_xp, num_layers=3, max_channel=256):
        super().__init__()
        layers_list = []
        layers_dim = [input_dim]
        for i in range(1, num_layers + 1):
            layers_dim.append(min(max_channel, (2**i)*layer_xp))
        for i in range(len(layers_dim) - 1):
            layers_list.append(DownBlock(layers_dim[i], layers_dim[i + 1]))
        self.block = torch.nn.ModuleList(layers_list)

    def forward(self, x):
        outs = [x]
        for layer in self.block:
            outs.append(layer(outs[-1]))
        return outs

class Decoder_kp(torch.nn.Module):
    def __init__(self, input_dim, layer_xp, num_layers=3, max_channel=256):
        super().__init__()
        layers_list = []
        layers_dim = 2*[min(max_channel, (2**num_layers)*layer_xp)]
        for i in range(1, num_layers + 1):
            layers_dim.append(min(max_channel, (2**(num_layers - i))*layer_xp))
        for i in range(len(layers_dim) - 2):
            layers_list.append(UpBlock(layers_dim[i], layers_dim[i + 2]))
        self.block = torch.nn.ModuleList(layers_list)

    def forward(self, x):
        out = x.pop()
        for layer in self.block:
            out = layer(out)
            out = torch.cat((out, x.pop()), dim=1)
        return out

class BottleNeck(torch.nn.Module):
    def __init__(self, input_dim, layer_xp, num_layers=3, max_channel=256):
        super().__init__()
        self.encoder = Encoder_kp(input_dim, layer_xp, num_layers, max_channel)
        self.decoder = Decoder_kp(input_dim, layer_xp, num_layers, max_channel)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def heatmap2kp(heatmap):
    heatmap = heatmap.unsqueeze(-1) #(b, heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], 1)
    grid = meshgrid(heatmap.shape[2:]).unsqueeze_(0).unsqueeze_(0) #(1, 1, heatmap.shape[2], heatmap.shape[3], 2)
    coordinate = (heatmap*grid).sum(dim=(2, 3)) #(b, heatmap.shape[1], 2)
    return coordinate

def heatmap_prob(kp, size, std=0.1):
    coordinate = meshgrid(size).unsqueeze(0).unsqueeze(0).repeat(kp.shape[0], kp.shape[1], 1, 1, 1) #(b, kp.shape[1], size[0], size[1], 2)
    z_score = (coordinate - kp.view(kp.shape[0], kp.shape[1], 1, 1, 2))/std #(b, kp.shape[1], size[0], size[1], 2)
    prob = torch.exp(-0.5*(z_score**2).sum(-1)) #(b, kp.shape[1], size[0], size[1])
    return prob

def heatmap_diff(kp_source, kp_driving, size):
    heatmap = heatmap_prob(kp_driving['kp'], size) - heatmap_prob(kp_source['kp'], size) #(b, kp_driving['kp'].shape[1], size[0], size[1])
    zeros = torch.zeros(heatmap.shape[0], 1, size[0], size[1]) #(b, 1, size[0], size[1])
    heatmap = torch.cat((zeros, heatmap), dim=1) #(b, kp_driving['kp'].shape[1] + 1, size[0], size[1])
    return heatmap.unsqueeze(2)

def sparse_motions(frame_source, kp_source, kp_driving):
    b, c, h, w = frame_source.shape
    coordinate = meshgrid((h, w)).unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1, 1) #(b, 1, h, w, 2)
    flow = coordinate - kp_driving['kp'].unsqueeze(2).unsqueeze(2) #(b, kp_driving['kp'].shape[1], h, w, 2)
    if 'jacobian' in kp_source:
        jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian'])) #(b, kp_source['jacobian'].shape[1], 2, 2)
        jacobian = jacobian.unsqueeze(-3).unsqueeze(-3) #(b, kp_source['jacobian'].shape[1], 1, 1, 2, 2)
        flow = torch.matmul(jacobian, flow.unsqueeze(-1)) #(b, kp_source['jacobian'].shape[1], h, w, 2, 1)
        flow = flow.squeeze(-1) #(b, kp_source['jacobian'].shape[1], h, w, 2)
    DtoS = flow + kp_source['kp'].view(b, -1, 1, 1, 2) #(b, kp_source['kp'].shape[1], h, w, 2)
    motion = torch.cat((coordinate, DtoS), dim=1) #(b, kp_driving['kp'].shape[1] + 1, h, w, 2)
    return motion

def deform_source(frame_source, flow, num_kp):
    b, c, h, w = frame_source.shape
    frame_undeformed = frame_source.unsqueeze(1).unsqueeze(1).repeat(1, num_kp + 1, 1, 1, 1, 1) #(b, num_kp + 1, 1, c, h, w)
    frame_undeformed = frame_undeformed.view(b*(num_kp + 1), c, h, w) #(b*(num_kp + 1), c, h, w)
    frame_deformed = torch.nn.functional.grid_sample(frame_undeformed, flow.view((b*(num_kp + 1), h, w, 2)), align_corners=False).view((b, num_kp + 1, c, h, w)) #(b, num_kp + 1, c, h, w)
    return frame_deformed