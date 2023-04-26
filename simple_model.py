import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, max_state):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 1)
        self.conv2 = nn.Conv1d(6, 16, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 2 * 1, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, max_state)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        # If the size is a square, you can specify with a single number
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        # x = torch.argmax(x, dim=
        return x

class ConvNet(nn.Module):
    def __init__(self, output_size, hidden_size=64, kernel_size=2):
        super(ConvNet, self).__init__()
        # self.linear = nn.Linear(1, hidden_size)
        self.conv1d_1 = nn.Conv1d(1, hidden_size, kernel_size)
        self.conv1d_2 = nn.Conv1d(hidden_size, output_size, kernel_size)
        
    def forward(self, x, t):
        x = self.conv1d_1(x)
        x = nn.functional.relu(x)
        x = self.conv1d_2(x)
        x = nn.functional.relu(x)
        # incorporate time here somehow...
        x = F.softmax(x)
        x = torch.argmax(x,dim=2)
        return x

