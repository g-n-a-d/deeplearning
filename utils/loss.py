import torch
from torchvision import models
import numpy as np
from utils.Modules import meshgrid

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Transform:
    def __init__(self, b, sigma_affine, sigma_tps=None, points_tps=None):
        noise = torch.normal(mean=0, std=sigma_affine*torch.ones((b, 2, 3))) #(b, 2, 3)
        self.theta = noise + torch.eye(2, 3).unsqueeze(0) #(b, 2, 3)
        self.b = b
        self.tps = True
        self.control_points = meshgrid((points_tps, points_tps)).unsqueeze(0) #(1, points_tps, points_tps, 2)
        self.control_params = torch.normal(mean=0, std=sigma_tps*torch.ones((b, 1, points_tps**2))) #(b, 1, points_tps**2)
    def transform_frame(self, frame):
        b, c, h, w = frame.shape
        grid = meshgrid((h, w)).unsqueeze(0) #(1, h, w, 2)
        grid = grid.view(1, h*w, 2) #(1, h*w, 2)
        grid = self.warp_kp(grid).view(self.b, h, w, 2) #(b, h, w, 2)
        return torch.nn.functional.grid_sample(frame, grid, padding_mode="reflection", mode='nearest') #(b, c, h, w)
    def warp_kp(self, kp):
        theta = self.theta.unsqueeze(1) #(b, 1, 2, 3)
        transformed = torch.matmul(theta[:, :, :, :2], kp.unsqueeze(-1)) + theta[:, :, :, 2:] #(b, num_kp, 2, 1)
        transformed = transformed.squeeze(-1) #(b, num_kp, 2)
        distances = kp.view(kp.shape[0], -1, 1, 2) - self.control_points.view(1, 1, -1, 2) #(b, num_kp, point_tps**2, 2)
        distances = torch.abs(distances).sum(-1) #(b, num_kp, point_tps**2)
        result = distances**2 #(b, num_kp, point_tps**2)
        result = result*torch.log(distances + 1e-6) #(b, num_kp, point_tps**2)
        result = result*self.control_params #(b, num_kp, point_tps**2)
        result = result.sum(dim=2).view(self.b, kp.shape[1], 1) #(b, num_kp, 1)
        transformed += result #(b, num_kp, 2)
        return transformed
    def jacobian(self, kp):
        new_kp = self.warp_kp(kp) #(b, num_kp, 2)
        grad_x = torch.autograd.grad(new_kp[..., 0].sum(), kp, create_graph=True)
        grad_y = torch.autograd.grad(new_kp[..., 1].sum(), kp, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian

class L1Loss(torch.nn.Module):
    def __init__(self, equivariance_constraint_value=False, equivariance_constraint_jacobian=False):
        super().__init__()
        self.ecv = equivariance_constraint_value
        self.ecj = equivariance_constraint_jacobian
        self.vgg = Vgg19()

    def forward(self, frame_pred, frame_target, frame_driving, kp_extractor, kp_driving):
# if sum(self.loss_weights['perceptual']) != 0:
        loss_total = 0
        x_vgg = self.vgg(frame_pred)
        y_vgg = self.vgg(frame_target)
        for i in range(5):
            loss_total += torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
        if self.ecv or self.ecj:
            transform = Transform(frame_driving.shape[0], **self.train_params['transform_params'])
            frame_transformed = transform.transform_frame(frame_driving) #(b, c, h, w)
            kp_transformed = kp_extractor(frame_transformed) #kp(b, num_kp, 2), jacobian()

            # generated['transformed_frame'] = transformed_frame
            # generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.ecv:
                value = torch.abs(kp_driving['kp'] - transform.warp_coordinates(kp_transformed['kp'])).mean()
                loss_total += value

            ## jacobian loss part
            if self.ecj:
                jacobian_transformed = torch.matmul(transform.jacobian(kp_transformed['kp']), kp_transformed['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                I = torch.eye(2).unsqueeze(0).unsqueeze(0)

                # value = torch.abs(eye - value).mean()
                # loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
                loss_total += torch.abs(value - I).mean()
        return loss_total