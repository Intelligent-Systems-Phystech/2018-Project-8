# cnn model

import torch
import torchvision
import h5py
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
print(torch.__version__)

from HARread import *


use_cuda = torch.cuda.is_available()



trainX, trainy, testX, testy = load_dataset()
trainX, testX = scale_data(trainX, testX, standartize=True)


class MyDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.labels = labels.flatten()
        if self.transform is not None:
            self.data = data

    def __getitem__(self, item):
        label, row = self.labels[item], self.data[item]
        if self.transform is not None:
            row = self.transform(row)
        return row, label

    def __len__(self):
        return len(self.labels)


def toTensor(data):
    return torch.from_numpy(data.transpose((1, 0))).float()
    

transform = transforms.Compose([toTensor])


BATCH_SIZE = 256

Train_data = MyDataset(trainX, trainy, transform=transform)
Test_data = MyDataset(testX, testy, transform=transform)
trainloader = DataLoader(dataset=Train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              )
testloader = DataLoader(dataset=Test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              )


# ## Determine model


LR = 7e-3
NUM_CLASS = 6
RAW_SIZE = 128
CHANNEL = 9
NUM_EPOCHS = 600


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(CHANNEL, 32, kernel_size=5, padding=2),
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


torch.cuda.empty_cache()
model = Model()
if use_cuda:
    model = model.cuda()


# Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# Train network

loss_list = []
acc = []
running_loss = 0.0

lambda2 = lambda epoch: 0.996 ** epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])

for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    for i, (inputs, labels) in enumerate(trainloader, 0):
        
        # get the inputs
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
# print statistics
        #if i % 60 == 59 :
    loss_list.append(running_loss)
    # if epoch % 5 == 4:
    model.eval()
    correct = 0
    total = 0
    # Iterate through test dataset
    scheduler.step()
    with torch.no_grad():
        for (inputs, labels) in testloader:
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Epoch : %d; Iteration : %d; Accuracy : %.3f; loss : %.3f' 
                            %(epoch, i, correct/total, running_loss))
    acc.append(correct/total)
    model.train()
    running_loss = 0
        
print('Finished Training')


with h5py.File('acc_loss.hdf5', 'w') as f:
    f.create_dataset("accuracy", data=acc)
    f.create_dataset("loss", data=loss_list)

torch.save(model.state_dict(), 'saved_model_state.pt')