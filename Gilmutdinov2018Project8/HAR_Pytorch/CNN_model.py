
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
# print(torch.__version__)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.3),
            nn.BatchNorm1d(32))

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32))
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.3),
            nn.BatchNorm1d(64))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.3),
            nn.MaxPool1d(2),
            nn.Dropout(0.3))

        self.layer5 = nn.Sequential(
            nn.Linear(32*64, 64),
            nn.LeakyReLU(0.3),
   #        nn.Dropout(0.5),
            nn.Linear(64, 6))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        return out


