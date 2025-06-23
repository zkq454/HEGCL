import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

class Classifier(nn.Module):
    def __init__(self, in_fea, hid_fea, out_fea, drop_out=0.5):
        super(Classifier, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, out_fea))

    def forward(self, doc_fea):
        z = F.normalize(self.projector(doc_fea),dim=1)
        return z


class UCL(nn.Module):
    def __init__(self, in_fea, out_fea, temperature):
        super(UCL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, out_fea),
            nn.BatchNorm1d(out_fea),
            nn.ReLU(inplace=True),
            nn.Linear(out_fea, out_fea))
        self.tem = temperature
        self.auto_iou = 0.5
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def forward(self, doc_fea, doc_fea_svd):
        out, out_svd = self.projector(doc_fea), self.projector(doc_fea_svd)
        out_1, out_2 = F.normalize(out, dim=1), F.normalize(out_svd, dim=1)
        
        out = torch.cat([out_1, out_2], dim=0)
        dim = out.shape[0]
        sim_matrix = torch.exp(torch.mm(out, out.t()) / self.tem)
        mask = (torch.ones_like(sim_matrix) - torch.eye(dim, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(dim, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.tem)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        b1 = pos_sim <= self.auto_iou - 0.1
        a1 = torch.ones_like(pos_sim)
        b2 = (pos_sim > (self.auto_iou - 0.1)) & (pos_sim < self.auto_iou)
        a2 = torch.exp(torch.ones_like(pos_sim) * (1.0 - self.auto_iou))
        b3 = pos_sim >= self.auto_iou
        a3 = torch.exp(-pos_sim + 1.0)
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3

        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1)) * modulating_weight).mean()

        return loss


class WCL(nn.Module):
    def __init__(self, in_fea, hid_fea, temperature):
        super(WCL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, hid_fea))
        self.projector_2 = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, hid_fea))
        self.tem = temperature
   
    @torch.no_grad()
    def build_connected_component(self, dist):
        b = dist.size(0)
        dist = dist.fill_diagonal_(-2)
        device = dist.device
        x = torch.arange(b).unsqueeze(1).repeat(1,1).flatten().to(device)
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])
        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels).to(device)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is not None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * diagnal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, doc_fea, doc_fea_svd, pro, labels, share=True):
        out, out_svd = self.projector(doc_fea), self.projector(doc_fea_svd)
        out_1 = F.normalize(out, dim=1)
        out_2 = F.normalize(out_svd, dim=1)
        weight_decay = 5e-3

        pro = F.normalize(pro, dim=1)
        b = out_1.shape[0]

        if share:
            labels = torch.tensor(labels).to(doc_fea.device)
            mask1 = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
            mask2 = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)

        else:
            out = self.projector_2(doc_fea)
            out = F.normalize(out, dim=1)
            out_1, out_2 = out[even], out[odd]
            mask1 = self.build_connected_component(out_1 @ out_1.T).float()
            mask2 = self.build_connected_component(out_2 @ out_2.T).float()
        diagnal_mask = torch.eye(b).to(doc_fea.device)
        graph_loss = self.sup_contra(out_1 @ out_1.T / self.tem, mask2, diagnal_mask)
        graph_loss += self.sup_contra(out_2 @ out_2.T / self.tem, mask1, diagnal_mask)
        graph_loss /= 2

        reg_loss = weight_decay * (torch.norm(pro) ** 2)
        total_loss = graph_loss + reg_loss

        return total_loss

class PNCL(nn.Module):
    def __init__(self, in_features, out_features, temperature, alpha):
        super(PNCL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features)
        )
        self.temperature = temperature
        self.alpha = alpha

    def feature_enhancement(self, features):
        noise = torch.randn_like(features) * self.alpha
        enhanced_features = features + noise
        return enhanced_features

    def similarity_matrix(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def forward(self, input1, input2):
        enhanced_input1 = self.feature_enhancement(input1)
        enhanced_input2 = self.feature_enhancement(input2)

        projected1 = self.projector(enhanced_input1)
        projected2 = self.projector(enhanced_input2)


        sim_matrix = self.similarity_matrix(projected1, projected2)
        pos_sim = torch.diag(sim_matrix)
        neg_sim = sim_matrix.sum(dim=1) - pos_sim

        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()

        return loss