import torch
import torch.nn as nn
import os
from torchvision.models import resnet18, resnet34
from model_relevant import ModifiedResNet18, Generator, Discriminator

D_weight_path = ''
# Define the path for saving the ONNX file
onnx_path = ''

# input dimension for closed-set classifier following your settings
dummy_input = torch.randn(1, 512, 1, 1)
nc = 512
ngf = ndf = 256
netD = Discriminator(nc=nc, ndf=ndf)
netD.load_state_dict(torch.load(D_weight_path, weights_only=True))
netD.eval()

# Use the torch.onnx.export function
torch.onnx.export(netD,
                  dummy_input,             # input dimension for closed-set classifier following your settings
                  onnx_path,               # saving path
                  export_params=True,      # export all model parameters
                  opset_version=18,        # ONNX opset version
                  do_constant_folding=True, 
                  input_names=['input'],
                  output_names=['output']
)
