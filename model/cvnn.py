import numpy as np
import torch
from torch import nn
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear, Dropout, Parameter
from torch.nn import ReLU, Softmax
from model.complexcnn import ComplexConv
import torch.nn.functional as F
from config import Dataset


class base_complex_model(nn.Module):
    def __init__(self):
        super(base_complex_model, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=3)
        self.batchnorm1 = BatchNorm1d(num_features=128)
        self.maxpool1 = MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=3)
        self.batchnorm2 = BatchNorm1d(num_features=128)
        self.maxpool2 = MaxPool1d(kernel_size=2)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = BatchNorm1d(num_features=128)
        self.maxpool3 = MaxPool1d(kernel_size=2)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = BatchNorm1d(num_features=128)
        self.maxpool4 = MaxPool1d(kernel_size=2)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = BatchNorm1d(num_features=128)
        self.maxpool5 = MaxPool1d(kernel_size=2)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = BatchNorm1d(num_features=128)
        self.maxpool6 = MaxPool1d(kernel_size=2)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = BatchNorm1d(num_features=128)
        self.maxpool7 = MaxPool1d(kernel_size=2)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = BatchNorm1d(num_features=128)
        self.maxpool8 = MaxPool1d(kernel_size=2)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = BatchNorm1d(num_features=128)
        self.maxpool9 = MaxPool1d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = LazyLinear(1024)
        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(1024)))
        self.linear2 = LazyLinear(Dataset.classes)


    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)
        x = self.flatten(x)
        x = self.linear1(x)
        lamda = self.lamda
        x *= lamda
        embedding = F.relu(x)
        #output = self.dropout(embedding)
        output = self.linear2(embedding)
        return embedding, output

class prune_complex_model(nn.Module):
    def __init__(self,m):
        super(prune_complex_model, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=3)
        self.batchnorm1 = BatchNorm1d(num_features=128)
        self.maxpool1 = MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=3)
        self.batchnorm2 = BatchNorm1d(num_features=128)
        self.maxpool2 = MaxPool1d(kernel_size=2)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = BatchNorm1d(num_features=128)
        self.maxpool3 = MaxPool1d(kernel_size=2)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = BatchNorm1d(num_features=128)
        self.maxpool4 = MaxPool1d(kernel_size=2)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = BatchNorm1d(num_features=128)
        self.maxpool5 = MaxPool1d(kernel_size=2)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = BatchNorm1d(num_features=128)
        self.maxpool6 = MaxPool1d(kernel_size=2)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = BatchNorm1d(num_features=128)
        self.maxpool7 = MaxPool1d(kernel_size=2)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = BatchNorm1d(num_features=128)
        self.maxpool8 = MaxPool1d(kernel_size=2)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = BatchNorm1d(num_features=128)
        self.maxpool9 = MaxPool1d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = LazyLinear(m)
        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(m)))
        self.linear2 = LazyLinear(Dataset.classes)


    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)
        x = self.flatten(x)
        x = self.linear1(x)
        lamda = self.lamda
        x *= lamda
        embedding = F.relu(x)
        output = self.linear2(embedding)
        return embedding, output

def main():
    model = base_complex_model()
    # r1_output = [p for n, p in model.named_parameters() if 'lamda' in n]
    # #r1_output = [{"params": [p for n, p in model.named_parameters() if 'lamda' in n]}]
    # print( "1:", np.array(r1_output).shape)
    # print("2:",r1_output)
    # print("3:",torch.from_numpy(np.array(r1_output)))
    # target = torch.zeros(np.array(r1_output).shape)
    # print("4:",target)
    # print("5:",target.shape)
    #print({'params': model.parameters()})
    # print("1:", model.lamda)
    # zero_data = torch.zeros(model.lamda.size())
    # if torch.cuda.is_available():
    #     zero_data = zero_data.to(device)
    # print("2:", zero_data)


    #print(r1_output.shape)
if __name__ == '__main__':
    main()