import torch
from torchvision import models
from utils.Modules import meshgrid

class Vgg19_pretrained(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).requires_grad_(False).features
        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.block3 = torch.nn.Sequential()
        self.block4 = torch.nn.Sequential()
        for x in range(0, 9):
            self.block1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.block2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.block3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.block4.add_module(str(x), vgg_pretrained_features[x])
        self.mean = torch.nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
    
    def forward(self, x):
        x = (x - self.mean)/self.std
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        outs = [out1, out2, out3, out4]
        return outs

class Transform(torch.nn.Module):
    def __init__(self, b, kpdetector, scale_std=0.5, shift_std=0.3):
        super().__init__()
        self.kp_detector = kpdetector
        self.scale = torch.nn.Parameter(scale_std*(torch.rand((b, 1)) - 0.5) + 1, requires_grad=False) #(b, 1)
        theta = torch.nn.Parameter(6.28318530718*torch.rand((b, 1)), requires_grad=False) #(b, 1)
        costheta = torch.cos(theta).unsqueeze(-1) #(b, 1, 1)
        sintheta = torch.sin(theta).unsqueeze(-1) #(b, 1, 1)
        rotationmatrix = torch.cat((torch.cat((costheta, sintheta), dim=-1), torch.cat((-sintheta, costheta), dim=-1)), dim=-2) #(b, 2, 2)
        self.rotation = torch.nn.Parameter(rotationmatrix, requires_grad=False) #(b, 2, 2)
        self.shift = torch.nn.Parameter(shift_std*torch.rand((b, 2)), requires_grad=False) #(b, 2)

    def warp_kp(self, kp):
        kp_tf = self.scale.unsqueeze(1)*kp #(b, num_kp, 2)
        kp_tf = torch.matmul(kp_tf, self.rotation)#(b, num_kp, 2)
        kp_tf = kp_tf + self.shift.unsqueeze(1) #(b, num_kp, 2)
        return kp_tf

    def tf_frame(self, frame):
        b, c, h, w = frame.shape
        grid = meshgrid((h, w)).to(frame.device).unsqueeze(0).repeat(b, 1, 1, 1) #(b, h, w, 2)
        grid = grid.view(b, h*w, 2) #(b, h*w, 2)
        grid = self.warp_kp(grid).view(b, h, w, 2) #(b, h, w, 2)
        return torch.nn.functional.grid_sample(frame, grid, align_corners=True) #(b, c, h, w)

    def jacobian(self, kp):
        kp_warped = self.warp_kp(kp) #(b, num_kp, 2)
        grad_x = torch.autograd.grad(kp_warped[:, :, 0].sum(), kp, create_graph=True) #(b, num_kp, 2)
        grad_y = torch.autograd.grad(kp_warped[:, :, 1].sum(), kp, create_graph=True) #(b, num_kp, 2)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2) #(b, num_kp, 2, 2)
        return jacobian

    def forward(self, frame):
        loss = {}
        kp = self.kp_detector(frame)
        frame_transformed = self.tf_frame(frame) #(b, c, h, w)
        kp_transformed = self.kp_detector(frame_transformed) #kp(b, num_kp, 2), jacobian(b, num_kp, 2, 2)
        ecv = self.warp_kp(kp_transformed['kp']) #(b, num_kp, 2)
        loss['ecv'] = torch.abs(kp['kp'] - ecv).mean() #(1)
        Xgrad_inv = torch.inverse(kp['jacobian']) #(b, num_kp, 2, 2)
        ecj = torch.matmul(Xgrad_inv, torch.matmul(self.jacobian(kp_transformed['kp']), kp_transformed['jacobian'])) #(b, num_kp, 2, 2)
        I = torch.eye(2).to(frame.device) #(2, 2)
        loss['ecj'] = torch.abs(ecj - I).mean() #(1)
        return loss

class L1Loss(torch.nn.Module):
    def __init__(self, scales=(1.,), equivariance_constraint_value=False, equivariance_constraint_jacobian=False, weight_loss=[1., 1., 1.], kpdetector=None, scale_std=0.5, shift_std=0.3):
        super().__init__()
        self.scales = scales
        self.ecv = equivariance_constraint_value
        self.ecj = equivariance_constraint_jacobian
        self.weight_loss = weight_loss
        self.kpdetector = kpdetector
        self.scale_std = scale_std
        self.shift_std = shift_std
        self.vgg = Vgg19_pretrained()

    def forward(self, pred, frame_driving):
        loss = {}
        loss['total'] = 0
        for scale in self.scales:
            frame_pred_ = torch.nn.functional.interpolate(pred['frame_generated'], scale_factor=scale, mode='bilinear', antialias=True)
            frame_target = torch.nn.functional.interpolate(frame_driving, scale_factor=scale, mode='bilinear', antialias=True)
            pred_vgg = self.vgg(frame_pred_)
            target_vgg = self.vgg(frame_target)
            for i in range(4):
                loss['total'] += self.weight_loss[0]*torch.abs(pred_vgg[i] - target_vgg[i]).mean()
        if self.ecv or self.ecj:
            tf = Transform(frame_driving.shape[0], self.kpdetector, self.scale_std, self.shift_std).to(frame_driving.device)
            loss_ec = tf(frame_driving)
            loss['ec'] = 0
            if self.ecv:
                loss['ec'] += loss_ec['ecv']
                loss['total'] += self.weight_loss[1]*loss_ec['ecv']
            if self.ecj:
                loss['ec'] += loss_ec['ecj']
                loss['total'] += self.weight_loss[2]*loss_ec['ecj']
        return loss

class VAELoss(torch.nn.Module):
    def __init__(self, scales=(1.,), equivariance_constraint_value=False, equivariance_constraint_jacobian=False, weight_loss=[1., 1., 1.], kpdetector=None, scale_std=0.5, shift_std=0.3):
        super().__init__()
        self.scales = scales
        self.ecv = equivariance_constraint_value
        self.ecj = equivariance_constraint_jacobian
        self.weight_loss = weight_loss
        self.kpdetector = kpdetector
        self.scale_std = scale_std
        self.shift_std = shift_std
        self.vgg = Vgg19_pretrained()
        self.mse = torch.nn.MSELoss()

    def kld(self, mean, logvar):
        return 0.5*torch.mean(torch.exp(logvar) - logvar + mean**2 - 1)

    def forward(self, pred, frame_driving):
        loss = {}
        loss['total'] = 0
        for scale in self.scales:
            frame_pred_ = torch.nn.functional.interpolate(pred['frame_generated'], scale_factor=scale, mode='bilinear', antialias=True)
            frame_target = torch.nn.functional.interpolate(frame_driving, scale_factor=scale, mode='bilinear', antialias=True)
            pred_vgg = self.vgg(frame_pred_)
            target_vgg = self.vgg(frame_target)
            for i in range(4):
                loss['total'] += self.mse(pred_vgg[i], target_vgg[i]) + self.kld(pred['mean'], pred['logvar'])
        if self.ecv or self.ecj:
            tf = Transform(frame_driving.shape[0], self.kpdetector, self.scale_std, self.shift_std).to(frame_driving.device)
            loss_ec = tf(frame_driving)
            loss['ec'] = 0
            if self.ecv:
                loss['ec'] += loss_ec['ecv']
                loss['total'] += self.weight_loss[1]*loss_ec['ecv']
            if self.ecj:
                loss['ec'] += loss_ec['ecj']
                loss['total'] += self.weight_loss[2]*loss_ec['ecj']
        return loss
