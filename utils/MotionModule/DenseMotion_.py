import torch
from utils.Modules import meshgrid
from utils.MotionModule.MotionModules import BottleNeck, heatmap_diff, sparse_motions, deform_source

class DenseMotion_(torch.nn.Module):
    def __init__(self, num_channels, num_kp, layer_xp, num_layers, max_channel=256):
        super().__init__()
        self.num_kp = num_kp
        self.bottle_neck = BottleNeck((num_kp + 1)*(num_channels + 1), layer_xp, num_layers, max_channel)
        self.mask = torch.nn.Sequential(
            torch.nn.Conv2d(layer_xp + (num_kp + 1)*(num_channels + 1), num_kp + 1, kernel_size=7, padding=3),
            torch.nn.Softmax(dim=1)
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(layer_xp + (num_kp + 1)*(num_channels + 1), 2, kernel_size=7, padding=3),
            torch.nn.Tanh()
        )

    def forward(self, frame_source, kp_source, kp_driving):
        out = {}
        b, c, h, w = frame_source.shape
        H_dot = heatmap_diff(kp_source, kp_driving, (h, w)) #(b, num_kp + 1, 1, h, w)
        out['heatmap'] = H_dot
        S = sparse_motions(frame_source, kp_source, kp_driving) #(b, num_kp + 1, h, w, 2)
        source_df = deform_source(frame_source, S, self.num_kp) #(b, num_kp + 1, c, h, w)
        map = self.bottle_neck(torch.cat([H_dot, source_df], dim=2).view(b, (self.num_kp + 1)*(c + 1), h, w)) #(b, layer_xp + (num_kp + 1)*(c + 1), h, w)
        mask = self.mask(map) #(b, num_kp + 1, h, w)
        motion_coarse = (mask.unsqueeze(2)*S.permute(0, 1, 4, 2, 3)).sum(dim=1).permute(0, 2, 3, 1) #(b, h, w, 2)
        motion_residual = self.residual(map)
        out['motion'] = motion_coarse + motion_residual #(b, h, w, 2)
        return out