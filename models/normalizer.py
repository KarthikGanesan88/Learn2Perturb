import torch
import torch.nn as nn

class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        tmp_mean = torch.Tensor(mean).unsqueeze(1).unsqueeze(1).cuda()
        tmp_std = torch.Tensor(std).unsqueeze(1).unsqueeze(1).cuda()
        self.mean = nn.Parameter(tmp_mean, requires_grad=False)
        self.std = nn.Parameter(tmp_std, requires_grad=False)
        # self.mean, self.std = self.mean.cuda(), self.std.cuda()

    def forward(self, x):
        return x.sub(self.mean).div(self.std)
