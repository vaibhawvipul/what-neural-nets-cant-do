import pandas as pd
import numpy as np 
import torch 
#import visdom

import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt

device = torch.device("cuda")

class MultiplicationDataset(Dataset):
    """Multiplication dataset."""

    def __init__(self, csv_file):
        
        self.multiply_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.multiply_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x1 = self.multiply_frame.iloc[idx, 0]
        x2 = self.multiply_frame.iloc[idx, 1]
        y = self.multiply_frame.iloc[idx, 2]

        sample = {'x1': x1, 'x2': x2, 'y': y}

        return sample

traindataloader = MultiplicationDataset(csv_file="data.csv")

# Uncomment the following code for sample data
"""for i in range(len(traindataloader)):
    sample = traindataloader[i]

    print(sample)   

    if i == 3 :
        break"""

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2, 100)  
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device=device)
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
loss_function = nn.L1Loss()

#visualizations: Todo
# viz = visdom.Visdom()

epochs = 100
for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(traindataloader, 0):

            inputs, gt = [data["x1"], data["x2"]], data["y"]
            inputs = torch.tensor(inputs).to(device=device)
            
            gt = torch.tensor([gt]).to(device=device)

            optimizer.zero_grad()
            
            #forward + backprop
            output = net(inputs)

            loss = loss_function(output, gt)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")
