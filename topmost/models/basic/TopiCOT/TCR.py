import torch
from torch import nn
import torch.nn.functional as F

class TCR(nn.Module):
    def __init__(self, cluster_center, weight_loss_TCR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2, init_a_dist=None, init_b_dist=None):
        super().__init__()
        
        self.cluster_center = cluster_center
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_TCR = weight_loss_TCR
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist

        if init_a_dist is not None:
            self.a_dist = init_a_dist

        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, topic_emb):
        if self.weight_loss_TCR <= 1e-6:
            return 0.
        device = self.cluster_center.device
        M = torch.cdist(topic_emb, self.cluster_center)

        # Sinkhorn's algorithm
        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)

        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)

        loss_TCR = torch.sum(transp * M)
        loss_TCR *= self.weight_loss_TCR

        return loss_TCR
