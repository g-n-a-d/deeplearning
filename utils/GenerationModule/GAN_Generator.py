import torch
from GenModules import Encoder_gen, Decoder_gen

class GAN_Generator(torch.nn.Module):
    def __init__(self, num_channels, num_kp, layer_xp, num_layers=3, num_reslayers=6, max_channel=256):
        super().__init__()
        self.encoder = Encoder_gen(num_channels, layer_xp, num_layers, max_channel)
        self.decoder = Decoder_gen(num_channels, layer_xp, num_layers, num_reslayers, max_channel)
        # if dense_motion_params is not None:
        #     self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
        #                                                    estimate_occlusion_map=estimate_occlusion_map,
        #                                                    **dense_motion_params)
        # else:
        #     self.dense_motion_network = None

        # self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        # down_blocks = []
        # for i in range(num_down_blocks):
        #     in_features = min(max_features, block_expansion * (2 ** i))
        #     out_features = min(max_features, block_expansion * (2 ** (i + 1)))
        #     down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # self.down_blocks = torch.nn.ModuleList(down_blocks)

        # up_blocks = []
        # for i in range(num_down_blocks):
        #     in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
        #     out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
        #     up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # self.up_blocks = torch.nn.ModuleList(up_blocks)

        # self.bottleneck = torch.nn.Sequential()
        # in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        # for i in range(num_bottleneck_blocks):
        #     self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        # self.final = torch.nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

    # def deform_input(self, inp, deformation):
    #     _, h_old, w_old, _ = deformation.shape
    #     _, _, h, w = inp.shape
    #     if h_old != h or w_old != w:
    #         deformation = deformation.permute(0, 3, 1, 2)
    #         deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
    #         deformation = deformation.permute(0, 2, 3, 1)
    #     return F.grid_sample(inp, deformation)

    def forward(self, frame_source, motion):
        # out = self.first(frame_source)
        # for i in range(len(self.down_blocks)):
        #     out = self.down_blocks[i](out)
        # outs = {}
        out = self.encoder(frame_source) #(b, min(max_channel, layer_xp), h/2**num_layers, w/2**num_layers)
        # Transforming feature representation according to deformation and occlusion
        # dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
        #                                          kp_source=kp_source)
        # output_dict['mask'] = motion['mask']
        # outs['sparse_deformed'] = motion['source_df']
        motion_flow = motion['motion'] #(b, h, w, 2)
        motion_flow = torch.nn.functional.interpolate(motion_flow.permute(0, 3, 1, 2), size=out.shape[2:], mode='bilinear').permute(0, 2, 3, 1) #(b, h/2**num_layers, w/2**num_layers, 2)
        out = torch.nn.functional.grid_sample(out, motion_flow) #(b, min(max_channel, layer_xp), h/2**num_layers, w/2**num_layers)
        if 'occlusion' in motion.keys():
            occlusion = motion['occlusion'] #(b, 1, h, w)
            # outs['occlusion_map'] = occlusion_map
            occlusion = torch.nn.functional.interpolate(occlusion, size=out.shape[2:], mode='bilinear') #(b, 1, h/2**num_layers, w/2**num_layers)
            out = out*occlusion #(b, min(max_channel, layer_xp), h/2**num_layers, w/2**num_layers)

        # outs['source_df'] = self.deform_input(frame_source, motion_flow)
        # if self.dense_motion_network is not None:
        #     dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
        #                                              kp_source=kp_source)
        #     output_dict['mask'] = dense_motion['mask']
        #     output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

        #     if 'occlusion_map' in dense_motion:
        #         occlusion_map = dense_motion['occlusion_map']
        #         output_dict['occlusion_map'] = occlusion_map
        #     else:
        #         occlusion_map = None
        #     deformation = dense_motion['deformation']
        #     out = self.deform_input(out, deformation)

        #     if occlusion_map is not None:
        #         if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
        #             occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        #         out = out * occlusion_map

        #     output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out = self.decoder(out) #(b, num_channels, h, w)
        # out = self.bottleneck(out)
        # for i in range(len(self.up_blocks)):
        #     out = self.up_blocks[i](out)
        # out = self.final(out)
        # out = F.sigmoid(out)

        # outs['frame_gen'] = out

        return out