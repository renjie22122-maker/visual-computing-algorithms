import torch
import torch.nn as nn

class DenoisingCNN(nn.Module):

    def __init__(self):
        super(DenoisingCNN, self).__init__()
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 3, 3, padding=1))

    def forward(self, x):
        return self.encoder(x)

def loss_function(prediction, target):
    loss = torch.mean((prediction - target) ** 2)
    return loss

def get_vgg_gram_matrix(input):
    pass
    b, c, h, w = input.size()
    features = input.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)

def get_vgg_mean(input):
    pass
    mean = input.mean(dim=[2, 3], keepdim=True)
    return mean

def get_vgg_var(input):
    pass
    var = input.var(dim=[2, 3], keepdim=True, unbiased=False)
    return var

class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        pass
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        pass
        return (img - self.mean) / self.std
