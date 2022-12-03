import torch

class UpBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=1, upscale=2):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=upscale)
        )

    def forward(self, x):
        out = self.block(x)
        return out

class DownBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=1, downscale=2):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=downscale)
        )

    def forward(self, x):
        out = self.block(x)
        return out

def meshgrid(size):
    x = torch.linspace(-1, 1, steps=size[1]) #(size[1])
    y = torch.linspace(-1, 1, steps=size[0]) #(size[0])
    x_, y_ = torch.meshgrid(x, y, indexing='xy') #(size[1]), (size[0])
    return torch.cat((x_.unsqueeze(-1), y_.unsqueeze(-1)), dim=2) #(size[0], size[1], 2)