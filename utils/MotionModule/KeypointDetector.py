import torch
from utils.Modules import BottleNeck
from MotionModules import heatmap2kp

class KeypointDetector(torch.nn.Module):
    def __init__(self, num_channels, num_kp, layer_xp, num_layers, max_channel, temperature, jacobian=False, scale_factor=1):
        super().__init__()
        self.num_kp = num_kp
        self.jacobian = jacobian
        self.bottle_neck = BottleNeck(num_channels, layer_xp, num_layers, max_channel)
        self.heatmap = torch.nn.Conv2d(num_channels + layer_xp, num_kp, kernel_size=7, padding=3)
        self.softmax = torch.nn.Softmax(dim=2)
        if self.jacobian:
            self.jacobian = torch.nn.Conv2d(self.bottle_neck.out_filters, 4*num_kp, kernel_size=7, padding=3)
            # self.jacobian.weight.data.zero_()
            # self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))

        self.temperature = temperature
        self.scale_factor = scale_factor
        # if self.scale_factor != 1:
        #     self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def forward(self, x):
        # if self.scale_factor != 1:
        #     x = self.down(x)
        out = {}
        x = self.bottle_neck(x) #(b, num_channels + layer_xp, h, w)
        heatmap = self.heatmap(x) #(b, num_kp, h, w)
        heatmap = self.softmax(heatmap.view(heatmap.shape[0], heatmap.shape[1], -1)/self.temperature).view(*heatmap.shape) #(b, num_kp, h, w)
        out['kp'] = heatmap2kp(heatmap) #(b, num_kp, 2)
        if self.jacobian:
            jacobian = self.jacobian(x) #(b, 4*num_kp, h, w)
            jacobian = jacobian.view(heatmap.shape[0], self.num_kp, 4, heatmap.shape[2], heatmap.shape[3]) #(b, num_kp, 4, h, w)
            jacobian = heatmap.unsqueeze(2)*jacobian #(b, num_kp, 4, h, w)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 4, -1) #(b, num_kp, 4, h*w)
            jacobian = jacobian.sum(dim=-1) #(b, num_kp, 4)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) #(b, num_kp, 2, 2)
            out['jacobian'] = jacobian #(b, num_kp, 2, 2)
        return out