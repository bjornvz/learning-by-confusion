import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom1DCNN(nn.Module):
    def __init__(self, regression=False):
        super(Custom1DCNN, self).__init__()
        self.regression = regression
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # self.pool = nn.MaxPool2d(2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.fc1 = nn.Linear(128 * 37, 256)  # 300 → 150 → 75 → 37
        self.fc2 = nn.Linear(256, 128)
        
        # regression output:
        if self.regression:
            self.fc3 = nn.Linear(128, 1) 
        else:
            self.fc3 = nn.Linear(128, 2) # NOTE: no softmax here; included in CrossEntropyLoss
 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation for regression

        return x



