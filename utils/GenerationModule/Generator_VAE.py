import torch
from utils.GenerationModule.GenModules import Encoder_gen, Decoder_gen

class Generator_VAE(torch.nn.Module):
    def __init__(self, num_channels, layer_xp, num_layers, num_reslayers, max_channel=256, scalestep=2):
        super().__init__()
        self.encoder = Encoder_gen(num_channels, layer_xp, num_layers, max_channel, scalestep)
        self.decoder = Decoder_gen(num_channels, layer_xp, num_layers, num_reslayers, max_channel, scalestep)
        self.mean = torch.nn.Conv2d(min(max_channel, (2**num_layers)*layer_xp), min(max_channel, (2**num_layers)*layer_xp), kernel_size=3, padding=1)
        self.logvar = torch.nn.Conv2d(min(max_channel, (2**num_layers)*layer_xp), min(max_channel, (2**num_layers)*layer_xp), kernel_size=3, padding=1)
        self.upsampler = torch.nn.Upsample(scale_factor=1/scalestep**num_layers, mode='bilinear')

    def forward(self, frame_source, motion):
        out = {}
        latent = self.encoder(frame_source) #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        mean = self.mean(latent) #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        out['mean'] = mean #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        logvar = self.logvar(latent) #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        out['logvar'] = logvar #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        z_randn = torch.randn(mean.shape).to(frame_source.device) #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        latent = mean + torch.exp(0.5*logvar)*z_randn #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        motion_flow = motion['motion'] #(b, h, w, 2)
        motion_flow = self.upsampler(motion_flow.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) #(b, h/scalestep**num_layers, w/scalestep**num_layers, 2)
        latent = torch.nn.functional.grid_sample(latent, motion_flow, align_corners=True) #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        if 'occlusion' in motion.keys():
            occlusion = motion['occlusion'] #(b, 1, h, w)
            occlusion = self.upsampler(occlusion) #(b, 1, h/scalestep**num_layers, w/scalestep**num_layers)
            latent = latent*occlusion #(b, min(max_channel, (2**num_layers)*layer_xp), h/2**num_layers, w/2**num_layers)
        out['frame_generated'] = self.decoder(latent) #(b, num_channels, h, w)
        return out