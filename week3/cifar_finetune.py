'''
This is starter code for Assignment 2 Problem 1 of CMPT 726 Fall 2020.
The file is adapted from the repo https://github.com/chenyaofo/CIFAR-pretrained-models
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
NUM_EPOCH = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

######################################################
####### Do not modify the code above this line #######
######################################################
import os

class cifar_resnet20(nn.Module):
    def __init__(self):
        super(cifar_resnet20, self).__init__()
        ResNet20 = CifarResNet(BasicBlock, [3, 3, 3])
        url ='https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt'
		
        ResNet20.load_state_dict(model_zoo.load_url(url))
        modules = list(ResNet20.children())[:-1]
        backbone = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)  # backbone: feature extraction from input layer
        self.fc = nn.Linear(64, 10)   # fc: fully-connected layer
                                      # creates single layer feed forward network with m inputs and n output.

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.shape[0], -1)  # .view: reshape the tensor.  
                                          # -1: situation that we don't know how many rows wanted, but sure of the number of columns
        return self.fc(out)


best_losses = float('inf')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cifar_resnet20().to(device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, 
                                          shuffle=True, num_workers=2)  

    # add test data for validation
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, 
                                          shuffle=True, num_workers=2) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9, weight_decay=1e-3)   # add L2 regularization

    ## Do the training
    
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):   # index from 0. data contains inputs and labels (tensor type). i: # of batch 
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()  # accumulates gradient for each parameter (w.grad += dloss/dw)
            optimizer.step()  # update parameter (w += -lr*w.grad)
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('[%d,%5d] Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)'
                    % (epoch + 1, i + 1, running_loss/20, 100*correct/total, correct, total))
                running_loss = 0
                correct = 0
                total = 0
        PATH = '/content/drive/My Drive/train_model'
        torch.save(model.state_dict(), PATH)
        print('Finished Training')


        ## Do the validation
        v_loss = 0.0
        t_correct = 0
        t_total = 0
        best_epoch = 1
        # best_loss = 0
        t_model = cifar_resnet20().to(device)
        t_model.load_state_dict(torch.load(PATH))
        with torch.no_grad():
            for i, test_data in enumerate(testloader, 0):   # index from 0. data contains inputs and labels (tensor type).
                # get the inputs
                t_inputs, t_labels = test_data[0].to(device), test_data[1].to(device)
                # forward + backward + optimize
                t_outputs = t_model(t_inputs)
                t_loss = criterion(t_outputs, t_labels)
                v_loss += t_loss.item()
                t_, t_predicted = torch.max(t_outputs.data, 1)
                t_total += t_labels.size(0)
                t_correct += (t_predicted == t_labels).sum().item()
                
        print('[%d] Validation Loss: %.3f | Validation Acc: %.3f%% (%d/%d)'
            % (epoch+1, v_loss/i, 100 * t_correct/t_total, t_correct, t_total))
        losses = v_loss/i
        # acc = 100 * t_correct/t_total
        if losses < best_losses:
            best_acc = 100 * t_correct/t_total
            best_epoch = epoch+1
            best_losses = losses
            best_correct = t_correct
            best_total = t_total
            BEST_PATH = '/content/drive/My Drive/best_model'
            torch.save(t_model.state_dict(), BEST_PATH)

        print('Finished Validation')

    print('Best Validation Loss: %.3f | Best Validation Accuracy: %.3f%% (%d/%d)'
        % (best_losses, best_acc, best_correct, best_total))

