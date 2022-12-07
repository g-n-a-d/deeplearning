import torch
from utils.GenerationModule.GenModules import Encoder_gen_, Decoder_gen_

class Generator_Unet(torch.nn.Module):
    def __init__(self, num_channels, num_kp, layer_xp, num_layers, num_reslayers, max_channel=256, scalestep=2):
        super().__init__()
        self.num_kp = num_kp
        self.encoder = Encoder_gen_(num_channels, layer_xp, num_layers, max_channel, scalestep)
        self.decoder = Decoder_gen_(num_channels, num_kp + 1, layer_xp, num_layers, num_reslayers, max_channel, scalestep)

    def forward(self, frame_source, motion):
        out = {}
        latent = self.encoder(frame_source) #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
        motion_flow = motion['motion'] #(b, h, w, 2)
        if 'heatmap' in motion.keys():
            heatmap = motion['heatmap'] #(b, num_kp + 1, h, w)
        else:
            heatmap = torch.zeros((frame_source.shape[0], self.num_kp + 1, frame_source.shape[2], frame_source.shape[3])) #(b, num_kp + 1, h, w)
        for i in range(len(latent)):
            motion_flow_ = torch.nn.functional.interpolate(motion_flow.permute(0, 3, 1, 2), size=(latent[i].shape[2], latent[i].shape[3])).permute(0, 2, 3, 1) #(b, h/scalestep**num_layers, w/scalestep**num_layers, 2)
            heatmap_ = torch.nn.functional.interpolate(heatmap, size=(latent[i].shape[2], latent[i].shape[3])) #(b, num_kp + 1, h/scalestep**num_layers, w/scalestep**num_layers)
            latent[i] = torch.nn.functional.grid_sample(latent[i], motion_flow_, align_corners=True) #(b, min(max_channel, (2**num_layers)*layer_xp), h/scalestep**num_layers, w/scalestep**num_layers)
            latent[i] = torch.cat((latent[i], heatmap_), dim=1) #(b, min(max_channel, (2**num_layers)*layer_xp) + num_kp + 1, h/scalestep**num_layers, w/scalestep**num_layers)
        out['frame_generated'] = self.decoder(latent) #(b, num_channels, h, w)
        return out