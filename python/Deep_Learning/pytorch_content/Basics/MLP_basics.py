
from torch import nn, optim
from collections import OrderedDict

# declare network parameters 
n_input = 784
n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 64
n_output = 10 


#### WAYS TO CREATE MLP NETWORKS WITH DIFFERENT APPROACHES #####

#### 1. Creating Network with an ordered Dicitonary via nn.Sequential
net_struct = OrderedDict([
    ("input", nn.Linear(n_input, n_hidden_1)),
    ("relu1", nn.ReLU()),
    ("hidden_1", nn.Linear(n_hidden_1, n_hidden_2)),
    ("relu2", nn.ReLU()),
    ("hidden_2", nn.Linear(n_hidden_2, n_hidden_3)),
    ("relu3", nn.ReLU()),
    ("hidden_3", nn.Linear(n_hidden_3, n_output)),
    ("log_softmax", nn.LogSoftmax(dim=1))
])

model = nn.Sequential(net_struct)

##### 2. Creating a class with the given parameters

from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, n_output) # n_output different classes
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
    
###### 3. Creating directly with nn.Sequential

# Build a feed-forward network
model = nn.Sequential(nn.Linear(n_input, n_hidden_1),
                      nn.ReLU(),
                      nn.Linear(n_hidden_1, n_hidden_2),
                      nn.ReLU(),
                      nn.Linear(n_hidden_2, n_hidden_3),
                      nn.ReLU(),
                      nn.Linear(n_hidden_3, n_output),
                      nn.LogSoftmax(dim=1))