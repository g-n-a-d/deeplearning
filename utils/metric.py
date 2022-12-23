import torch
from torchvision import models
import face_alignment
from torch.utils.data import DataLoader

def reconstructionloss(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    rloss = 0.
    for x, y in dataloader:
        pred = model(x, y)['frame_generated'].detach()
        rloss += torch.abs(y - pred).mean()
    return rloss.item()/len(dataloader)

def akd(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)
    score = 0.
    err = 0
    for x, y in dataloader:
        pred = model(x, y)['frame_generated'].detach()
        s0 = fa.get_landmarks(x.squeeze(0).permute(1, 2, 0)*255)
        s1 = fa.get_landmarks(pred.squeeze(0).permute(1, 2, 0)*255)
        if s0 == None or s1 == None:
            err += 0
        else:
            tmp = 0.
            for i in range(len(s0[0])):
                tmp += ((s0[0][i][0] - s1[0][i][0])**2 + (s0[0][i][1] - s1[0][i][1])**2)**0.5/(x.shape[-1]/2)
            score += tmp.item()/len(s0[0])
    return score/(len(dataloader) - err + 1e-9)

class Inception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = models.inception_v3(weights='IMAGENET1K_V1')
        self.inception.dropout = torch.nn.Identity()
        self.inception.fc = torch.nn.Identity()

    def forward(self, x):
        return self.inception(x)

def fid(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    inception = Inception().eval()
    p = []
    t = []
    for x, y in dataloader:
        pred = model(x, y)['frame_generated'].detach()
        pred = torch.nn.functional.interpolate(pred, size=(299, 299))
        y = torch.nn.functional.interpolate(y, size=(299, 299))
        p.append(inception(pred).detach())
        t.append(inception(y).detach())
    meanp = torch.cat(p, dim=0).mean(dim=0)
    covp = torch.cat(p, dim=0).T.cov()
    meant = torch.cat(t, dim=0).mean(dim=0)
    covt = torch.cat(t, dim=0).T.cov()
    score = (torch.sum((meanp - meant)**2) + torch.trace(covp + covt - 2*torch.sqrt(covp*covt))).item()
    return score
