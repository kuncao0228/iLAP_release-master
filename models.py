import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        modules=list(self.model.children())[:-1]
        self.classifier =nn.Sequential(*modules)
        self.fc = nn.Linear(512, num_classes)
        self.alpha = nn.Parameter(torch.ones(num_classes, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(num_classes, requires_grad=True, device="cuda"))
        
    def forward(self, x):
        out = self.classifier(x)
        out = torch.flatten(out, 1)
        if self.fc:
            out = self.fc(out)
            out = self.alpha * out + self.beta
        return out
    
    def changeFC3(self, length):
        self.fc = nn.Linear(512, length)
        self.alpha = nn.Parameter(torch.ones(length, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(length, requires_grad=True, device="cuda"))