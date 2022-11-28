import torch
from utils.GenerationModule.GenModules import Encoder_gen, Decoder_gen

class GAN_Generator(torch.nn.Module):
    def __init__(self, num_channels, layer_xp, num_layers, num_reslayers, max_channel=256):
        super().__init__()
        self.encoder = Encoder_gen(num_channels, layer_xp, num_layers, max_channel)
        self.decoder = Decoder_gen(num_channels, layer_xp, num_layers, num_reslayers, max_channel)

    def forward(self, frame_source, motion):
        out = self.encoder(frame_source) #(b, min(max_channel, layer_xp), h/2**num_layers, w/2**num_layers)
        motion_flow = motion['motion'] #(b, h, w, 2)
        motion_flow = torch.nn.functional.interpolate(motion_flow.permute(0, 3, 1, 2), size=out.shape[2:], mode='bilinear').permute(0, 2, 3, 1) #(b, h/2**num_layers, w/2**num_layers, 2)
        out = torch.nn.functional.grid_sample(out, motion_flow, align_corners=False) #(b, min(max_channel, layer_xp), h/2**num_layers, w/2**num_layers)
        if 'occlusion' in motion.keys():
            occlusion = motion['occlusion'] #(b, 1, h, w)
            occlusion = torch.nn.functional.interpolate(occlusion, size=out.shape[2:], mode='bilinear') #(b, 1, h/2**num_layers, w/2**num_layers)
            out = out*occlusion #(b, min(max_channel, layer_xp), h/2**num_layers, w/2**num_layers)
        out = self.decoder(out) #(b, num_channels, h, w)
        return out