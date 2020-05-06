  
## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # first convolutional layer:
        self.conv1 = nn.Conv2d(1, 32, 5) 
        # (32,92,92) output tensor # (W-F)/S + 1 = 96-5/1 +1=92 
        
        #First Max pooling layer:
        self.pool1 = nn.MaxPool2d(2,2) # (32,46,46) output tensor
        
        #Second convelutional layer:
        self.conv2 = nn.Conv2d(32, 64, 5) # (W-F)/S +1 = (46-5)/1 + 1=42 #(64,42,42) output tensor 
        
        #Second Max pooling layer:
        self.pool2 = nn.MaxPool2d(2,2) # (64,21,21) output tensor
        
        #Fully connected layer:
        self.fc1 = nn.Linear(64*21*21,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,136)
        self.drop1 = nn.Dropout(p=0.5)


        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)           
        x = self.drop1(x)
        
        #Flatten before passing to the fully connected layer
        x = x.view(x.size(0),-1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x