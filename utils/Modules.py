import torch

class UpBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=1):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        out = torch.nn.functional.interpolate(x, scale_factor=2)
        out = self.layer(out)
        return out


class DownBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=1):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Encoder_kp(torch.nn.Module):
    def __init__(self, input_dim, layer_xp, num_layers=3, max_channel=256):
        super().__init__()
        layers_list = []
        layers_dim = [input_dim]
        for i in range(1, num_layers + 1):
            layers_dim.append(min(max_channel, (2**i)*layer_xp))
        for i in range(len(layers_dim) - 1):
            layers_list.append(DownBlock(layers_dim[i], layers_dim[i + 1]))
        # for i in range(num_blocks):
        #     down_blocks.append(DownBlock(input_dim if i == 0 else min(max_dim, layer_xp * (2 ** i)),
        #                                    min(max_dim, layer_xp * (2 ** (i + 1))),
        #                                    kernel_size=3, padding=1))
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
        # for i in range(num_blocks)[::-1]:
        #     in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, layer_xp * (2 ** (i + 1)))
        #     out_filters = min(max_features, layer_xp * (2 ** i))
        #     up_blocks.append(UpBlock(in_filters, out_filters, kernel_size=3, padding=1))

        self.block = torch.nn.ModuleList(layers_list)
        # self.out_filters = layer_xp + input_dim

    def forward(self, x):
        out = x.pop()
        for layer in self.block:
            out = layer(out)
            out = torch.cat((out, x.pop()), dim=1)
            # if len(x) > 1:
            #     out = out + x.pop()
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

def meshgrid(size):
    x = torch.linspace(-1, 1, steps=size[1]) #(size[1])
    y = torch.linspace(-1, 1, steps=size[0]) #(size[0])
    x_, y_ = torch.meshgrid(x, y, indexing='xy') #(size[1]), (size[0])
    return torch.cat((x_.unsqueeze(-1), y_.unsqueeze(-1)), dim=2) #(size[0], size[1], 2)

# def heatmap2kp(heatmap):
#     shape = heatmap.shape
#     heatmap = heatmap.unsqueeze(-1)
#     grid = meshgrid(shape[2:]).unsqueeze_(0).unsqueeze_(0)
#     coordinate = (heatmap*grid).sum(dim=(2, 3))
#     return coordinate

# def heatmap_prob(kp, size, std=0.1):
#     coordinate = meshgrid(size).unsqueeze(0).unsqueeze(0).repeat(kp['kp'].shape[0], kp['kp'].shape[1], 1, 1, 1)
#     z_score = (coordinate - kp['kp'].view(kp['kp'].shape[0], kp['kp'].shape[1], 1, 1, 2))/std
#     prob = torch.exp(-0.5*(z_score**2).sum(-1))
#     return prob

# def heatmap_diff(kp_source, kp_driving, size):
#     heatmap = heatmap_prob(kp_driving, size) - heatmap_prob(kp_source, size)
#     zeros = torch.zeros(heatmap.shape[0], 1, size[0], size[1])
#     heatmap = torch.cat((zeros, heatmap), dim=1)
#     return heatmap.unsqueeze(2)

# def sparse_motions(frame_source, kp_source, kp_driving):
#     b, c, h, w = frame_source.shape
#     flow = (kp_source['value'] - kp_driving['value']).view(b, -1, 1, 1, 2)
#     coordinate = meshgrid((h, w)).unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1, 1)
#     # if 'jacobian' in kp_driving:
#     #     jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
#     #     jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
#     #     jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
#     #     coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
#     #     coordinate_grid = coordinate_grid.squeeze(-1)
#     DtoS = coordinate + flow
#     return torch.cat((coordinate, DtoS), dim=1)

# def deform_source(frame_source, flow, num_kp):
#     b, c, h, w = frame_source.shape
#     frame_undeformed = frame_source.unsqueeze(1).unsqueeze(1).repeat(1, num_kp + 1, 1, 1, 1, 1)
#     frame_undeformed = frame_undeformed.view(b*(num_kp + 1), c, h, w)
#     frame_deformed = torch.nn.functional.grid_sample(frame_undeformed, flow.view((b*(num_kp + 1), h, w, 2))).view((b, num_kp + 1, c, h, w))
#     return frame_deformed