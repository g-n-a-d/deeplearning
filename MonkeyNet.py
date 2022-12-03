import torch

class MonkeyNet(torch.nn.Module):
    def __init__(self, kpextractor, densemotion, generator):
        super().__init__()
        self.kpextractor = kpextractor
        self.densemotion = densemotion
        self.generator = generator

    def forward(self, frame_source, frame_driving):
        kp_source = self.kpextractor(frame_source) #kp(b, num_kp, 2), jacobian(b, num_kp, 2, 2)
        kp_driving = self.kpextractor(frame_driving) #kp(b, num_kp, 2), jacobian(b, num_kp, 2, 2)
        motion = self.densemotion(frame_source, kp_source, kp_driving) #motion(b, h, w, 2), occlusion(b, 1, h, w)
        frame_generated = self.generator(frame_source, motion) #frame_generated(b, num_channels, h, w), mean&logvar(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        out = {
            'kp_source':kp_source, #kp(b, num_kp, 2), jacobian(b, num_kp, 2, 2)
            'kp_driving':kp_driving, #kp(b, num_kp, 2), jacobian(b, num_kp, 2, 2)
            'motion':motion, #motion(b, h, w, 2), occlusion(b, 1, h, w)
            'frame_generated':frame_generated['frame_generated'] #(b, num_channels, h, w)
        }
        if 'mean' in frame_generated.keys() and 'logvar' in frame_generated.keys():
            out['mean'] = frame_generated['mean'] #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
            out['logvar'] = frame_generated['logvar'] #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        return out