import torch
from torchvision import models
import face_alignment

def reconstructionloss(model, data):
    data = torch.Tensor(data)
    rloss = 0
    l1 = torch.nn.L1Loss()
    pred = model(data[:, 0, :, :, :], data[:, 1, :, :, :])['frame_generated']
    return l1(pred, data[:, 1, :, :, :]) 

def akd(model, data):
    data = torch.Tensor(data)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)
    score = 0.
    pred = model(data[:, 0, :, :, :], data[:, 1, :, :, :])['frame_generated']
    for i in range(pred.shape[0]):
        s0 = fa.get_landmarks(pred[i].permute(1, 2, 0))
        s1 = fa.get_landmarks(data[i][1].permute(1, 2, 0))
        if s0 == None or s1 == None:
            score += 128.
        else:
            tmp = 0.
            for i in range(len(s0[0])):
                tmp += ((s0[i][0] - s1[i][0])**2 + (s0[i][1] - s1[i][1])**2)**0.5
            score += tmp/len(s0[0])
    return score/pred.shape[0]

class Inception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = models.inception_v3(weights='IMAGENET1K_V1')
        self.inception.dropout = torch.nn.Identity()
        self.inception.fc = torch.nn.Identity()

    def forward(self, x):
        return self.inception(x)

def fid(model, data):
    data = torch.Tensor(data)
    inception = Inception().eval()
    pred = model(data[:, 0, :, :, :], data[:, 1, :, :, :])['frame_generated']
    meanp = inception(pred).mean(dim=0)
    covp = inception(pred).T.cov()
    meant = inception(data[:, 1, :, :, :]).mean(dim=0)
    covt = inception(data[:, 1, :, :, :]).T.cov()
    return torch.sum((meanp - meant)**2) + torch.trace(covp + covt - 2*torch.sqrt(covp*covt))
