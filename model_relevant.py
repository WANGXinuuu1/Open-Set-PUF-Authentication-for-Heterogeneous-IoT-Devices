import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)     

class MLP(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=8, 
            kernel_size=(3, 3),
            stride=1, 
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, 
            out_channels=8, 
            kernel_size=(3, 3),
            stride=1, 
            padding=0
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5) 
        self.fc = nn.Linear(in_features=8 * 98 * 108, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.conv2(x)
        x = F.relu(x) 
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ModifiedResNet18(nn.Module):
    def __init__(self, model, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, features


class ModifiedResNet34(nn.Module):
    def __init__(self, model, num_classes=10):
        super(ModifiedResNet34, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        features = x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, features


class ModifiedResNet50(nn.Module):
    def __init__(self, model, num_classes=10):
        super(ModifiedResNet50, self).__init__()
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        
        in_features = model.fc.in_features 
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        features = x 
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, features

    
class Discriminator(nn.Module):
    def __init__(self, nc=512, ndf=256):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf*8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.ndf*8, self.ndf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.ndf*4, self.ndf*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=256, nc=512):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d( self.nz, self.ngf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ngf*8) x 4 x 4
            nn.Conv2d( self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ngf*4) x 8 x 8
            nn.Conv2d( self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ngf*2) x 16 x 16
            nn.Conv2d( self.ngf * 2, self.ngf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ngf) x 32 x 32
            nn.Conv2d( self.ngf*4, self.nc, 1, 1, 0, bias=True),
            # nn.Tanh()
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class BCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(BCNN, self).__init__()
        self.conv1 = bnn.BayesianConv2d(
            in_channels=1, 
            out_channels=8, 
            kernel_size=(16, 16),
            stride=1, 
            padding=1
        )
        self.conv2 = bnn.BayesianConv2d(
            in_channels=8, 
            out_channels=8, 
            kernel_size=(9, 9),
            stride=1, 
            padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8) 
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5) 
        self.fc1 = bnn.BayesianLinear(in_features=4400, out_features=512)
        self.fc2 = bnn.BayesianLinear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.conv2(x)
        x = F.relu(x) 
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        return x
