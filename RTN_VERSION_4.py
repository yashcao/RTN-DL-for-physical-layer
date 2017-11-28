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
# USE_CUDA = False
class_num = 16         # (n=7,k=4)  m=16

test_labels = (torch.rand(5) * class_num).long()
test_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=test_labels)

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

model = AutoEncoder()
print(model)        # net architecture

# model = torch.load('net.pkl')
model.load_state_dict(torch.load('net_params.pkl'))

if __name__ == '__main__':
    test_data = Variable(test_data)
    # test_labels = Variable(test_labels)
    # training and testing

    test_output = model(test_data)
    pred_labels = torch.max(test_output, 1)[1].data.squeeze()
    print(test_labels)
    print(pred_labels)
    accuracy = sum(pred_labels == test_labels) / float(test_labels.size(0))
    print('test accuracy: %.2f' % accuracy)