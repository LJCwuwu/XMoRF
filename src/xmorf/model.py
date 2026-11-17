"""Global-summary predictor compatible with the trained checkpoint keys.

The checkpoint keys ``spectral.spec`` refer to the one-point, zero-frequency
summary used by every final checkpoint. The saved scalar gate and epsilon are
retained to reproduce the trained network exactly.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class InputFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.names = ['prott5', 'prostt5']
        self.p_mod = 0.1
        self.proj = nn.ModuleDict({n: nn.Sequential(nn.LayerNorm(1024), nn.Dropout(0.2),
                                                  nn.Linear(1024, 128)) for n in self.names})
        self.mix = nn.Sequential(nn.Linear(256, 128), nn.GELU())

    def forward(self, feats, mask):
        zs = [self.proj[n](feats[n].float()) for n in self.names]
        if self.training and self.p_mod > 0:
            keep = (torch.rand(zs[0].shape[0], 2, device=zs[0].device) > self.p_mod).float()
            dead = keep.sum(1) == 0
            if dead.any():
                keep[dead, torch.randint(0, 2, (int(dead.sum()),), device=keep.device)] = 1.0
            zs = [z * keep[:, i, None, None] / (1.0-self.p_mod) for i,z in enumerate(zs)]
        return self.mix(torch.cat(zs, -1)) * mask[..., None]


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.conv1 = nn.Conv1d(128,128,5,padding=2)
        self.conv2 = nn.Conv1d(128,128,1)
        self.drop = nn.Dropout(0.3)

    def forward(self,x,m):
        y=(self.norm(x)*m).transpose(1,2)
        y=self.conv2(F.gelu(self.conv1(y))).transpose(1,2)
        return (x+self.drop(y))*m


class WholeProteinSummary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('theta',torch.linspace(0.0,math.pi,1))
        self.gate_logit=nn.Parameter(torch.zeros(1,1))

    def forward(self,x,mask):
        batch, channels, length=x.shape
        m=mask.to(x.dtype)
        x=x*m[:,None,:]
        n=torch.arange(length,device=x.device,dtype=x.dtype)
        ang=n[:,None]*self.theta.to(x.dtype)[None,:]
        c,s=x@torch.cos(ang),x@torch.sin(ang)
        mag=torch.sqrt(c**2+s**2+1e-8)/m.sum(-1).clamp(min=1)[:,None,None]
        feat=torch.log1p(mag)*torch.sigmoid(self.gate_logit)[0].view(1,1,-1)
        return feat.reshape(batch,channels,1).expand(batch,channels,length)*m[:,None,:]


class SummaryBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,16))
        self.spec=WholeProteinSummary()
        self.out=nn.Sequential(nn.LayerNorm(16),nn.Linear(16,128),nn.GELU(),nn.Dropout(0.3))
        self.smooth=ResidualBlock()

    def forward(self,h,mask):
        with torch.autocast(device_type=h.device.type,enabled=False):
            sig=self.proj(h.float()).float().transpose(1,2)
            summary=self.spec(sig,mask.float())
        s=self.out(summary.transpose(1,2).to(h.dtype))
        m=mask[...,None].to(h.dtype)
        return self.smooth((h+s)*m,m)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp=InputFusion()
        self.spectral=SummaryBranch()
        self.out=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU(),
                               nn.Dropout(0.3),nn.Linear(128,1))

    def forward(self,feats,mask):
        h=self.inp(feats,mask)
        logits=self.out(self.spectral(h,mask)).squeeze(-1)
        return logits.masked_fill(mask==0,0.0)


class NestedPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.r=Predictor()
        self.q=Predictor()

    def forward(self,feats,mask):
        return self.r(feats,mask),self.q(feats,mask)
