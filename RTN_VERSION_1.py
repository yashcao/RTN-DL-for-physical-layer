# standard library
import os
from math import sqrt
# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)   # reproducible

# Hyper Parameters
EPOCH = 50             # train the training data e times
BATCH_SIZE = 8
LR = 0.001              # learning rate
USE_CUDA = False

class_num = 16         # (n=7,k=4)  m=16
label = torch.range(1, class_num)-1
train_labels = label.long()
# train_labels = (torch.rand(5) * class_num).long()
train_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=train_labels)
torch_dataset = Data.TensorDataset(data_tensor = train_data, target_tensor = train_labels)
loader = Data.DataLoader(                          
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    )                  

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.transmitter = nn.Sequential(         # input shape (2^4=16, 16)
            nn.Linear(in_features=16, out_features=16, bias=True),      
            nn.ReLU(inplace = True),              # activation
            nn.Linear(16,7),
        )

        self.reciever = nn.Sequential(
            nn.Linear(7,16),
            nn.ReLU(inplace = True), 
            nn.Linear(16,16),
        )

    def forward(self, x):
        x = self.transmitter(x)

        # Normalization layer norm2(x)^2 = n
        n = (x.norm(dim=-1)[:,None].view(-1,1).expand_as(x))
        x = sqrt(7)*(x / n)

        """channel noise layer""" 
        training_SNR = 5.01187     # 7dBW to SNR. 10**(7/10)
        communication_rate = 4/7   # bit / channel_use
        # 1/(2*communication_rate*training_SNR)   fixed variance
        noise = Variable(torch.randn(x.size()) / ((2*communication_rate*training_SNR) ** 0.5))
        x += noise

        x = self.reciever(x)
        return x              # return x for visualization

net = AutoEncoder()
print(net)        # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                      # the target label is not one-hotted

if __name__ == '__main__':
    # training and testing
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader):   # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)   # batch x
            b_y = Variable(y)   # batch y

            output = net(b_x)             # output
            b_y = (b_y.long()).view(-1)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])