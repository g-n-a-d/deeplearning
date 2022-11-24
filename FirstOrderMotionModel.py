import torch

class FirstOrderMotionModel(torch.nn.Module):
    def __init__(self, kp_extractor, dense_motion, generator):
        super().__init__()
        self.kp_extractor = kp_extractor
        self.dense_motion = dense_motion
        self.generator = generator

    def forward(self, frame_source, frame_driving):
        kp_source = self.kp_extractor(frame_source) #kp(b, num_kp, 2), jacobian(b, num_kp, 2, 2)
        kp_driving = self.kp_extractor(frame_driving) #kp(b, num_kp, 2), jacobian(b, num_kp, 2, 2)
        motion = self.dense_motion(frame_source, kp_source, kp_driving) #motion(b, h, w, 2), occlusion(b, 1, h, w)
        frame_generated = self.generator(frame_source, motion) #(b, num_channels, h, w)
        out = {
            'kp_source':kp_source,
            'kp_driving':kp_driving,
            'motion':motion,
            'frame_generated':frame_generated
        }
        return out